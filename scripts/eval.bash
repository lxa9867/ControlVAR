cfg_list=("1.5" "2.0" "2.5" "3.0" "3.5" "4.0")
control_list=("mask" "canny" "depth" "normal")

# 外层循环遍历cfg_list
for cfg in "${cfg_list[@]}"
do
    # 内层循环遍历control_list
    for control in "${control_list[@]}"
    do
        echo "cfg: $cfg_$cfg, control: $control"
        echo "../experiments/d30/$control/cond/cfg_$cfg_$cfg_$cfg_$control"
        python3 evaluate.py --gt_dir ../../ImageNet2012/val --img_dir "../experiments/d30/$control/cond/cfg_$cfg_$cfg_$cfg_$control" --inception
        # python create_npz.py --path "/voyager/MaskVAR/d30/$control/cond/cfg_${cfg}_${cfg}_${cfg}_$control"
        python evaluator.py VIRTUAL_imagenet256_labeled.npz "/voyager/MaskVAR/d30/$control/cond/cfg_${cfg}_${cfg}_${cfg}_${control}.npz"
    done
done