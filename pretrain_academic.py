import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    TrainerCallback
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from datasets import load_dataset
from tree_sitter import Parser
from tree_sitter_languages import get_language

import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from collections import Counter

# 1. 从 ModelArguments 中移除 use_flash_attention_2
@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "Name for the trained model."})
    hidden_size: int = field(default=768, metadata={"help": "Dimension of the hidden states."})
    num_hidden_layers: int = field(default=8, metadata={"help": "Number of hidden layers."})
    num_attention_heads: int = field(default=12, metadata={"help": "Number of attention heads."})
    intermediate_size: int = field(default=3072, metadata={"help": "Dimension of the FFN."})
    max_position_embeddings: int = field(default=1024, metadata={"help": "Maximum sequence length."})
    symbol_vocab_size: int = field(default=10000, metadata={"help": "Maximum number of symbols in the fixed vocabulary."})
    max_symbols_per_file: int = field(default=128, metadata={"help": "Maximum number of symbols to consider per code file."})

@dataclass
class DataArguments:
    dataset_name: str = field(default="codeparrot/codeparrot-clean-valid", metadata={"help": "Dataset name from Hugging Face Hub."})
    tokenizer_name: str = field(default="gpt2", metadata={"help": "Tokenizer to use."})
    symbol_vocab_file: str = field(default="symbol_vocab.json", metadata={"help": "Path to the pre-built symbol vocabulary file."})
    code_language: str = field(default="python", metadata={"help": "The programming language for Tree-sitter parsing."})

# 2. 从 CodeCCATConfig 中移除 use_flash_attention_2
class CodeCCATConfig(PretrainedConfig):
    model_type = "codeccat"
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        symbol_vocab_size=10000,
        max_symbols_per_file=128,
        pad_token_id=50256,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.symbol_vocab_size = symbol_vocab_size
        self.max_symbols_per_file = max_symbols_per_file
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

class CoherenceFusionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        self.seq_self_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.struct_cross_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)

    def forward(self, seq_hidden_states, struct_hidden_states, attention_mask=None, symbol_attention_mask=None):
        seq_len = seq_hidden_states.size(1)
        device = seq_hidden_states.device
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None

        residual = seq_hidden_states
        attn_output, _ = self.seq_self_attention(
            seq_hidden_states, seq_hidden_states, seq_hidden_states,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
            need_weights=False
        )
        seq_hidden_states = self.norm1(residual + attn_output)

        residual = seq_hidden_states
        ca_symbol_attention_mask = (symbol_attention_mask == 0) if symbol_attention_mask is not None else None
        cross_attn_output, _ = self.struct_cross_attention(
            query=seq_hidden_states,
            key=struct_hidden_states,
            value=struct_hidden_states,
            key_padding_mask=ca_symbol_attention_mask,
            need_weights=False
        )
        seq_hidden_states = self.norm2(residual + cross_attn_output)
        
        residual = seq_hidden_states
        ffn_output = self.ffn(seq_hidden_states)
        seq_hidden_states = self.norm3(residual + ffn_output)
        
        return seq_hidden_states

class CodeCCATPreTrainedModel(PreTrainedModel):
    config_class = CodeCCATConfig
    base_model_prefix = "codeccat"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _post_init(self):
        super()._post_init()
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self, "lm_head") and hasattr(self, "token_embed"):
            self.lm_head.weight = self.token_embed.weight

class CodeCCATForCausalLM(CodeCCATPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.symbol_embed = nn.Embedding(config.symbol_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CoherenceFusionBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embed

    def set_input_embeddings(self, value):
        self.token_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids,
        symbol_ids,
        attention_mask=None,
        symbol_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        token_embeddings = self.token_embed(input_ids)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.pos_embed(position_ids)
        seq_hidden_states = token_embeddings + position_embeddings
        struct_hidden_states = self.symbol_embed(symbol_ids)
        
        for layer in self.layers:
            seq_hidden_states = layer(
                seq_hidden_states, 
                struct_hidden_states, 
                attention_mask=attention_mask,
                symbol_attention_mask=symbol_attention_mask
            )
            
        seq_hidden_states = self.final_norm(seq_hidden_states)
        logits = self.lm_head(seq_hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

class SymbolVocabulary:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, vocab_file, max_vocab_size, language='python'):
        self.vocab_file = vocab_file
        self.max_vocab_size = max_vocab_size
        self.parser = self._setup_parser(language)
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None

    def _setup_parser(self, language):
        lang_obj = get_language(language)
        parser = Parser()
        parser.set_language(lang_obj)
        combined_query_str = """
        (function_definition name: (identifier) @name)
        (class_definition name: (identifier) @name)
        (assignment left: (identifier) @name)
        (for_statement left: (identifier) @name)
        (aliased_import name: (dotted_name (identifier) @alias))
        (parameters (identifier) @param)
        (call function: (identifier) @function_call)
        """
        self.compiled_query = lang_obj.query(combined_query_str)
        return parser

    def _extract_symbols_from_code(self, code):
        if not code: return []
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            symbols = set()
            captures = self.compiled_query.captures(tree.root_node)
            for node, _ in captures:
                symbols.add(node.text.decode('utf8'))
            return list(symbols)
        except Exception as e:
            logging.warning(f"Failed to parse code and extract symbols. Error: {e}")
            return []

    def build_from_dataset(self, dataset):
        print(f"Building symbol vocabulary from dataset...")
        symbol_counts = Counter()
        for example in tqdm(dataset, desc="Scanning for symbols"):
            symbols = self._extract_symbols_from_code(example["content"])
            symbol_counts.update(symbols)

        most_common_symbols = symbol_counts.most_common(self.max_vocab_size - 2)
        self.id_to_token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.token_to_id = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        
        for i, (symbol, _) in enumerate(most_common_symbols):
            idx = i + 2
            self.id_to_token[idx] = symbol
            self.token_to_id[symbol] = idx
        
        print(f"Built vocabulary with {len(self.token_to_id)} symbols.")
        self.save()
        
    def save(self):
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        print(f"Symbol vocabulary saved to {self.vocab_file}")

    def load(self):
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}. Please build it first.")
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        print(f"Loaded symbol vocabulary with {len(self.token_to_id)} symbols from {self.vocab_file}")

class CodePreprocessor:
    def __init__(self, tokenizer, model_args, symbol_vocab: SymbolVocabulary):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.symbol_vocab = symbol_vocab
        self.unk_symbol_id = self.symbol_vocab.token_to_id[self.symbol_vocab.UNK_TOKEN]
        self.pad_symbol_id = self.symbol_vocab.token_to_id[self.symbol_vocab.PAD_TOKEN]

    def __call__(self, examples):
        tokenized_output = self.tokenizer(
            examples["content"],
            truncation=True,
            max_length=self.model_args.max_position_embeddings,
            padding="max_length",
        )
        batch_symbol_ids = []
        batch_symbol_attention_mask = []
        
        for code in examples["content"]:
            symbols = self.symbol_vocab._extract_symbols_from_code(code)
            symbol_ids = [self.symbol_vocab.token_to_id.get(s, self.unk_symbol_id) for s in symbols]
            symbol_ids = symbol_ids[:self.model_args.max_symbols_per_file]
            
            num_symbols = len(symbol_ids)
            padding_length = self.model_args.max_symbols_per_file - num_symbols
            symbol_attention_mask = [1] * num_symbols + [0] * padding_length
            symbol_ids.extend([self.pad_symbol_id] * padding_length)
            
            batch_symbol_ids.append(symbol_ids)
            batch_symbol_attention_mask.append(symbol_attention_mask)
            
        tokenized_output["symbol_ids"] = batch_symbol_ids
        tokenized_output["symbol_attention_mask"] = batch_symbol_attention_mask
        
        return tokenized_output

@dataclass
class CustomDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, examples):
        input_ids = [e["input_ids"] for e in examples]
        attention_mask = [e["attention_mask"] for e in examples]
        symbol_ids = [e["symbol_ids"] for e in examples]
        symbol_attention_mask = [e["symbol_attention_mask"] for e in examples]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "symbol_ids": torch.tensor(symbol_ids, dtype=torch.long),
            "symbol_attention_mask": torch.tensor(symbol_attention_mask, dtype=torch.long),
        }

        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            print(f"\nEvaluation Perplexity: {perplexity:.4f}")
            metrics["perplexity"] = perplexity

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("--- 1. Loading tokenizer and dataset ---")
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset(data_args.dataset_name)
    if "validation" not in raw_datasets:
        print("Validation set not found. Splitting from train set.")
        split = raw_datasets["train"].train_test_split(test_size=0.01, seed=training_args.seed)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]
    
    print("\n--- 2. Preparing Symbol Vocabulary ---")
    symbol_vocab = SymbolVocabulary(
        vocab_file=data_args.symbol_vocab_file,
        max_vocab_size=model_args.symbol_vocab_size,
        language=data_args.code_language
    )
    if not os.path.exists(data_args.symbol_vocab_file) or training_args.overwrite_output_dir:
        symbol_vocab.build_from_dataset(raw_datasets["train"])
    else:
        symbol_vocab.load()
    
    print("\n--- 3. Preprocessing dataset ---")
    preprocessor = CodePreprocessor(tokenizer, model_args, symbol_vocab)
    
    with training_args.main_process_first(desc="Dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            preprocessor,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=raw_datasets["train"].column_names,
        )

    print("\n--- 4. Initializing model ---")
    # 3. 从 config 创建中移除 use_flash_attention_2
    config = CodeCCATConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_position_embeddings=model_args.max_position_embeddings,
        symbol_vocab_size=len(symbol_vocab.token_to_id),
        max_symbols_per_file=model_args.max_symbols_per_file,
    )
    model = CodeCCATForCausalLM(config)
    print(f"Model created. Number of parameters: {model.num_parameters() / 1e6:.2f}M")

    print("\n--- 5. Setting up Trainer ---")
    data_collator = CustomDataCollator(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[PerplexityCallback()],
    )

    print("\n--- 6. Starting training ---")
    train_result = trainer.train()
    
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    print("\n--- 7. Final evaluation ---")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
