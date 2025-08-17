python pretrain_academic.py \
    --model_name "CodeCCAT-Small" \
    --dataset_name "codeparrot/codeparrot-clean-valid" \
    --output_dir "./results/codeccat_small" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --bf16
