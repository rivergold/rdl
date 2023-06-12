from typing import List, Dict
from pathlib import Path
from collections import OrderedDict
import itertools, copy
from rconfig import Config


def combine_hps(hyperparam_map: OrderedDict) -> List[Dict]:
    hp_names = hyperparam_map.keys()
    hp_maps = []
    for hp_group in itertools.product(*(hyperparam_map.values())):
        hp_map = OrderedDict()
        for hp_name, hp_value in zip(hp_names, hp_group):
            hp_map[hp_name] = hp_value
        hp_maps.append(hp_map)
    return hp_maps


def update_config(base_cfg: Config, hp_maps):
    cfgs = []
    for idx, hp_map in enumerate(hp_maps):
        cfg = copy.deepcopy(base_cfg)
        cfg.exp_name = f"{cfg.exp_name}-{idx}"
        for k, v in hp_map.items():
            if isinstance(v, str):
                str_code = f"cfg.{k} = '{v}'"
            else:
                str_code = f"cfg.{k} = {v}"
            print(str_code)
            exec(str_code)
        cfgs.append(cfg)
    return cfgs


def gen_multi_cfg_files(hyperparams, base_cfg_path, work_dir='./'):
    base_cfg_path = Path(base_cfg_path)
    work_dir = Path(work_dir).resolve()
    tmp_cfg_dir = work_dir / 'cfgs'
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_file(base_cfg_path)
    cfg.work_dir = work_dir.as_posix()
    hp_maps = combine_hps(hyperparams)
    cfgs = update_config(cfg, hp_maps)
    cfg_paths = []
    for idx, cfg in enumerate(cfgs):
        cfg_path = tmp_cfg_dir / f"{base_cfg_path.name.split('.')[0]}-{idx}.json"
        cfg.to_json(cfg_path)
        cfg_paths.append(cfg_path)
    return cfgs, cfg_paths