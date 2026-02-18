#!/usr/bin/env python3

import numpy as np
import itertools
import pathlib
import json
import yaml
import copy

# usage:
# python sh/write-grid-v0.py


# -------------------------
# edit only this block
# -------------------------

BASE_CONFIG = "sh/grids/config_bsm.yaml"
OUT_ROOT = "/scratch/tmp/sa7998/clm/AL-v0"

GRID_FILE = "sh/grids/clm_v0.txt"
CONFIG_DIR = "sh/grids/clm_v0"

INPUT_DIR = "/scratch/tmp/sa7998/clm-input/"

PARAM_SPACE = {

    # paths
    "paths.dataset": [
        f"{INPUT_DIR}/bsm-2m-AL-step0-gt_0_5.csv",
        f"{INPUT_DIR}/bsm-2m-AL-step0-gt_0_8.csv",
        f"{INPUT_DIR}/bsm-2m-AL-step0-gt_0_99.csv",

        f"{INPUT_DIR}/bsm-2m-AL-step7-gt_0_5.csv",
        f"{INPUT_DIR}/bsm-2m-AL-step7-gt_0_8.csv",
        f"{INPUT_DIR}/bsm-2m-AL-step7-gt_0_99.csv",

        f"{INPUT_DIR}/bsm-2m-gt_0.csv",
        f"{INPUT_DIR}/bsm-2m-gt_0_5.csv",
        f"{INPUT_DIR}/bsm-2m-gt_0_8.csv",
        f"{INPUT_DIR}/bsm-2m-gt_0_9.csv",
        f"{INPUT_DIR}/bsm-2m-gt_0_95.csv",
        f"{INPUT_DIR}/bsm-2m-gt_0_99.csv",

        f"{INPUT_DIR}/bsm-train.csv",

        f"{INPUT_DIR}/hmdb4-chemex.csv",
        f"{INPUT_DIR}/hmdb4-deepmet.csv",
        f"{INPUT_DIR}/hmdb5-full.csv",
    ],

    # model
    # "model_params.rnn_type": ["LSTM", "GRU"],
    # "model_params.hidden_size": [512, 1024],
    # "model_params.patience": [1000, 5000],

}

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

    # -------------------------
    # avoid excessive enum
    # -------------------------
    MAX_SAMPLES = 10_000_000
    MAX_ENUMS = 3

    dataset_path = params["paths.dataset"]
    n_rows = sum(1 for _ in open(dataset_path)) - 1

    allowed_enums = []

    for ef in base_cfg_data["enum_factors"]:
        if ef * n_rows <= MAX_SAMPLES:
            allowed_enums.append(ef)

    # skip config if no enum survives
    if not allowed_enums:
        continue

    # limit enum count to max 3 using alternate selection
    if len(allowed_enums) > MAX_ENUMS:
        idx = np.linspace(
            1,
            len(allowed_enums) - 1,
            MAX_ENUMS,
            dtype=int
        )
        allowed_enums = [allowed_enums[i] for i in idx]

    # -------------------------
    # copy and replace configs
    # -------------------------
    current_config = copy.deepcopy(base_cfg_data)
    current_config["enum_factors"] = allowed_enums

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