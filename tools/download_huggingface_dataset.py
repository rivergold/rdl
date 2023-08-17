from argparse import ArgumentParser
import datasets

if __name__ == '__main__':
    # dataset = datasets.load_dataset(
    #     'fusing/fill50k',
    #     cache_dir='/home/hejing/data/opensource/huggingface/data')
    # dataset.save_to_disk(
    #     '/home/hejing/data/opensource/huggingface/data/fusing-fill50k')

    # dataset_name = 'huggan/wikiart'
    # cache_dir = '/home/hejing/data/opensource/huggingface/data'
    # disk_dir_path = '/home/hejing/data/opensource/huggingface/disk-data/huggan__wikiart'

    dataset_name = 'Fung804/makoto-shinkai-picture'
    cache_dir = '/home/hejing/data/opensource/huggingface/data'
    disk_dir_path = '/home/hejing/data/opensource/huggingface/disk-data/Fung804__makoto-shinkai-picture'

    dataset = datasets.load_dataset(
        dataset_name,
        cache_dir=cache_dir)
    dataset.save_to_disk(
        disk_dir_path)