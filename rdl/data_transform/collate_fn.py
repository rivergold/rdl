import torch


def common_collate_fn(in_batch, **kwargs):
    # start_time = time.time()
    out_batch = {}
    keys = in_batch[0]['keys']
    out_batch['keys'] = keys

    if isinstance(in_batch[0], list):
        tmp_batch = []
        for each_batch in in_batch:
            tmp_batch.extend(each_batch)
        in_batch = tmp_batch

    # Init out_batch
    for k in keys:
        out_batch[f'batch_{k}'] = []

    # Feed data
    for _, sample in enumerate(in_batch):
        for k, v in sample.items():
            if k not in keys:
                continue
            if v is not None:
                #print(k)
                out_batch[f'batch_{k}'].append(v)
    # Stack data
    #ks = ['rgb_imgs', 'gray_imgs', 'head_pose_gts']
    # ks = ['rgb_imgs', 'gray_imgs', 'head_pose_gts', 'head_pose_cls_gts']
    for k in keys:
        if len(out_batch[f"batch_{k}"]):
            first_element = out_batch[f"batch_{k}"][0]
            if isinstance(first_element, int):
                out_batch[f"batch_{k}"] = torch.tensor(out_batch[f"batch_{k}"])
            elif isinstance(first_element, torch.Tensor):
                if first_element.dim() > 2:  # Image into batch
                    out_batch[f"batch_{k}"] = torch.stack(
                        out_batch[f"batch_{k}"], dim=0)
                elif first_element.dim() == 2:  # Text into batch
                    out_batch[f"batch_{k}"] = torch.cat(
                        out_batch[f"batch_{k}"], dim=0)
                else:
                    out_batch[f"batch_{k}"] = torch.tensor(
                        out_batch[f"batch_{k}"])
        else:
            raise ValueError(f'{out_batch[f"batch_{k}"]} length is 0.')

    # out_batch['face_gts'] = torch.cat(out_batch['face_gts'], dim=0)

    # Generate target
    if kwargs.get('anchorfree_generator'):
        out_batch['target'] = kwargs['anchorfree_generator'].gen_target(
            out_batch['rgb_imgs'].shape[2:4], out_batch['face_gts'])
    # print(f'[Debug] collate_fn time: {time.time() - start_time}')
    return out_batch
