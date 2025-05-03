import yaml
from yacs.config import CfgNode as CN


def load_cfg(cfg_file):
    with open(cfg_file, "r") as f:
        yaml_content = yaml.safe_load(f)
    config = CN(yaml_content)
    return config