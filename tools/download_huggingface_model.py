from huggingface_hub import snapshot_download, login

if __name__ == '__main__':
    # login()
    # repo_id = 'runwayml/stable-diffusion-v1-5'
    # repo_id = 'openai/clip-vit-large-patch14'
    # repo_id = 'lllyasviel/ControlNet-v1-1'
    # repo_id = 'CompVis/stable-diffusion-v-1-4-original'
    # repo_id = 'stabilityai/stable-diffusion-xl-base-0.9'
    # cache_dir = '/home/hejing/data/opensource/huggingface/hub'
    # local_dir = '/home/hejing/data/opensource/huggingface/model/model--stabilityai--stable-diffusion-xl-base-0.9'

    # repo_id = 'openai/clip-vit-large-patch14'
    # local_dir = '/home/hejing/data/opensource/huggingface/model/model--openai--clip-vit-large-patch14'

    repo_id = 'stabilityai/stable-diffusion-2-1-base'
    local_dir = '/home/hejing/data/opensource/huggingface/model/model--stabilityai--stable-diffusion-2-1-base'

    # snapshot_download(repo_id=repo_id,
    #                   cache_dir=cache_dir,
    #                   resume_download=True,
    #                   ignore_patterns=['tf_model'])

    snapshot_download(repo_id=repo_id,
                      local_dir=local_dir,
                      local_dir_use_symlinks=False,
                      resume_download=True,
                      ignore_patterns=['tf_model', '*.h5'])
