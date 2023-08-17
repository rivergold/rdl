# python train.py \
#     --cfg_path "./config/base.yaml" \
#     --work_dir "./out/train_out"

# accelerate launch \
#     --config_file ./config/accelerate/config.yaml \
#     train.py \
#     --cfg_path "./config/base-gpu1.yaml" \
#     --work_dir "./out/train_out"

# accelerate launch \
#     --config_file ./config/accelerate/with_deepspeed.yaml \
#     train.py \
#     --cfg_path "./config/base-gpu1.yaml" \
#     --work_dir "./out/train_out"

# accelerate launch \
#     --config_file ./config/accelerate/single_gpu.yaml \
#     train.py \
#     --cfg_path "./config/base.yaml" \
#     --work_dir "./out/train_out"

MODEL_NAME="/mnt/cd-aigc/model/stable-diffusion-v1-5"
accelerate launch \
    --config_file "./accelerate.yaml" \
    --mixed_precision="fp16" \
    train_by_diffusers.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --learning_rate=4e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" --lr_warmup_steps=0 \
    --output_dir="./out/train_out/sd-style" \
    --logging_dir="./out/train_out/sd-style" \
    --validation_prompt "a beautiful picture of a mug in the style of watercolor" \
    --validation_epochs 1 \
    --seed 100

# --max_train_steps=15000 \
