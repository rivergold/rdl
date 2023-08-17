from diffusers import DiffusionPipeline
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
import copy

if __name__ == '__main__':
    model_dir = '/mnt/cd-aigc/model/stable-diffusion-v1-5'
    cur_date_str = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    out_dir = Path(f"./out/test/{cur_date_str}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # new_out_dir = out_dir / 'new'
    # new_out_dir.mkdir(parents=True, exist_ok=True)
    # raw_out_dir = out_dir / 'raw'
    # raw_out_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16,)
    lora_dir = '/home/hejing/r/project/rdl/example/StyleLora/out/train_out/sd-style'
    pipe.unet.load_attn_procs(lora_dir)
    pipe.to('cuda:0')

    # prompt = 'watercolor, a cute dog, centered, white background, 4k, HDR'
    # prompt = 'watercollor, a beautiful picture of a seat'
    prompt = 'a beautiful picture of a baby penguin waring a blue hat, red gloves, green shirt in the style of watercollor'
    neg_prompt = ''
    seed = 14567
    w = 512
    h = 512
    num_images_per_prompt = 4
    num_inference_steps = 30
    # guidance_scale = 
    generator = torch.Generator("cuda").manual_seed(seed)
    res = pipe(
        prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            # guidance_scale=guidance_scale,
            generator=generator,
            # eta=eta,
    )
    new_imgs = copy.deepcopy(res.images)
    # for idx, img in enumerate(imgs):
    #     img.save(f"{new_out_dir}/{idx}.png")
    del pipe


    # Raw
    raw_pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16,)
    lora_dir = '/home/hejing/r/project/rdl/example/StyleLora/out/train_out/sd-style'
    # pipe.unet.load_attn_procs(lora_dir)
    raw_pipe.to('cuda:0')

    # prompt = 'watercolor, a cute dog, centered, white background, 4k, HDR'
    # prompt = 'a beautiful picture of a dog in the style of watercolor'
    neg_prompt = ''
    seed = 14567
    w = 512
    h = 512
    num_images_per_prompt = 4
    num_inference_steps = 30
    # guidance_scale = 
    generator = torch.Generator("cuda").manual_seed(seed)
    res = raw_pipe(
        prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            # guidance_scale=guidance_scale,
            generator=generator,
            # eta=eta,
    )
    raw_imgs = res.images
    # for idx, img in enumerate(imgs):
    #     img.save(f"{raw_out_dir}/{idx}.png")

    for idx, (raw_img, new_img) in enumerate(zip(raw_imgs, new_imgs)):
        out_img = Image.new('RGB', (raw_img.width + new_img.width, min(raw_img.height, new_img.height)))
        out_img.paste(raw_img, (0, 0))
        out_img.paste(new_img, (raw_img.width, 0))
        out_img.save(f"{out_dir}/{idx}.png")