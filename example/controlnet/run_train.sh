# python train.py \
#     --cfg_path "./config/base.yaml" \
#     --work_dir "./out/train_out"

# accelerate launch \
#     --config_file ./config/accelerate_config.yaml \
#     train.py \
#     --cfg_path "./config/base-gpu1.yaml" \
#     --work_dir "./out/train_out"


accelerate launch \
    --config_file ./config/accelerate-deepspeed.yaml \
    train.py \
    --cfg_path "./config/base-gpu1.yaml" \
    --work_dir "./out/train_out"
