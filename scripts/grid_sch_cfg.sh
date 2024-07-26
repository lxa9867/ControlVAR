# !/bin/bash

# Define the list of values for --cfg
cfg_values=(4 5 6)

# Outer loop for first value
for cfg_val1 in "${cfg_values[@]}"; do
    # Middle loop for second value
    for cfg_val2 in "${cfg_values[@]}"; do
        # Inner loop for third value
        for cfg_val3 in "${cfg_values[@]}"; do
            # Run the command with the current combination of values
            python3 train_mask_var_hpu.py \
                --batch_size 10 \
                --dataset_name imagenetC \
                --data_dir ../ImageNet2012 \
                --gpus 8 \
                --output_dir ImageNetM_lr4e-5_d20 \
                --multi_cond True \
                --config configs/train_mask_var_ImageNetC_d20.yaml \
                --val_only True \
                --resume experiments/ImageNetM_lr8e-5_d20/checkpoint_step_598890.pth \
                --debug True \
                --c_mask True \
                --cfg "$cfg_val1" "$cfg_val2" "$cfg_val3" \
                --save_val True \
                --val_cond 'mask'
        done
    done
done