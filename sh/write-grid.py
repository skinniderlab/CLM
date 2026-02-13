#!/usr/bin/env python3

import itertools
import pathlib

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

OUT_ROOT = "/scratch/tmp/sa7998/clm/AL-v0"

GRID_FILE = "sh/grids/grid_bsm_AL.txt"

# -------------------------
# tag builder
# -------------------------

def build_tag(params):

    parts = []

    # dataset stem first
    ds_stem = pathlib.Path(params["paths.dataset"]).stem
    parts.append(ds_stem)

    for key, value in params.items():

        if key in ["paths.dataset", "paths.output_dir"]:
            continue

        short_key = key.split(".")[-1]

        parts.append(f"{short_key}={value}")

    return "-".join(parts)

# -------------------------
# grid generation
# -------------------------

keys = list(PARAM_SPACE.keys())
values = list(PARAM_SPACE.values())

grid_lines = []

for combo in itertools.product(*values):

    params = dict(zip(keys, combo))

    # build tag
    tag = build_tag(params)

    # assign output dir
    params["paths.output_dir"] = f"{OUT_ROOT}/{tag}"

    # serialize
    line = " ".join(f"{k}={v}" for k, v in params.items())

    grid_lines.append(line)

# -------------------------
# write file
# -------------------------

grid_path = pathlib.Path(GRID_FILE)
grid_path.parent.mkdir(parents=True, exist_ok=True)

with open(grid_path, "w") as f:
    f.write("\n".join(grid_lines))

print(f"wrote {len(grid_lines)} experiments -> {grid_path}")
