#!/usr/bin/env python3

import itertools
import pathlib
import json
import yaml

# -------------------------
# edit only this block
# -------------------------

# usage:
# python sh/write-grid.py

PARAM_SPACE = {

    # paths
    "paths.dataset": [
        "workflow/clm-input/bsm-2m-AL-step0-gt_0_5.csv",
        "workflow/clm-input/bsm-2m-AL-step0-gt_0_8.csv",
        "workflow/clm-input/bsm-2m-AL-step0-gt_0_99.csv",
        "workflow/clm-input/bsm-2m-AL-step2-gt_0_5.csv",
        "workflow/clm-input/bsm-2m-AL-step2-gt_0_8.csv",
        "workflow/clm-input/bsm-2m-AL-step2-gt_0_99.csv",
        "workflow/clm-input/bsm-2m-AL-step7-gt_0_5.csv",
        "workflow/clm-input/bsm-2m-AL-step7-gt_0_8.csv",
        "workflow/clm-input/bsm-2m-AL-step7-gt_0_99.csv",
    ],

    # model
    # "model_params.rnn_type": ["LSTM", "GRU"],
    # "model_params.hidden_size": [512, 1024],
    # "model_params.patience": [1000, 5000],

}

BASE_CONFIG = "workflow/config/config_bsm.yaml"
OUT_ROOT = "/scratch/tmp/sa7998/clm/AL-v0"

GRID_FILE = "sh/grids/grid_bsm_AL.txt"
CONFIG_DIR = "sh/grids/configs_bsm_AL"

# -------------------------
# tag builder
# -------------------------

def build_tag(params):
    parts = []
    ds_stem = pathlib.Path(params["paths.dataset"]).stem
    parts.append(ds_stem)
    for key, value in params.items():
        if key in ["paths.dataset", "paths.output_dir"]:
            continue
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}={value}")
    return "-".join(parts)

def dict_set_nested(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

# -------------------------
# grid generation
# -------------------------

with open(BASE_CONFIG, "r") as f:
    base_cfg_data = yaml.safe_load(f)

keys = list(PARAM_SPACE.keys())
values = list(PARAM_SPACE.values())

config_dir_path = pathlib.Path(CONFIG_DIR)
config_dir_path.mkdir(parents=True, exist_ok=True)

grid_lines = []

for combo in itertools.product(*values):
    params = dict(zip(keys, combo))
    tag = build_tag(params)
    params["paths.output_dir"] = f"{OUT_ROOT}/{tag}"

    import copy
    current_config = copy.deepcopy(base_cfg_data)

    for k, v in params.items():
        dict_set_nested(current_config, k.split('.'), v)

    json_path = config_dir_path / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(current_config, f, indent=4)
    
    grid_lines.append(str(json_path.absolute()))

# -------------------------
# write file
# -------------------------

grid_path = pathlib.Path(GRID_FILE)
grid_path.parent.mkdir(parents=True, exist_ok=True)

with open(grid_path, "w") as f:
    f.write("\n".join(grid_lines))

print(f"Total {len(grid_lines)} experiments generated with full config.")