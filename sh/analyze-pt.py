import argparse
from pathlib import Path
import torch
import csv


# python sh/analyze-pt.py /scratch/tmp/sa7998/clm

def count_params(state_dict):
    return sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))


def load_params(file):
    try:
        obj = torch.load(file, map_location="cpu")

        if isinstance(obj, dict):
            state_dict = obj.get("state_dict", obj)
        else:
            return None

        return count_params(state_dict)

    except Exception as e:
        print(f"[ERROR] {file} -> {e}")
        return None


def analyze(file):
    size_mb = file.stat().st_size / (1024 ** 2)
    params = load_params(file)

    print(file)
    print(f"  size       : {size_mb:.2f} MB")
    print(f"  parameters : {params:,}" if params else "  parameters : UNKNOWN")

    return file, size_mb, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--out", default="summary.csv")
    args = parser.parse_args()

    path = Path(args.path)
    files = [path] if path.is_file() else list(path.rglob("*.pt")) + list(path.rglob("*.pth"))

    if not files:
        print("no files found.")
        return

    rows = []

    for f in sorted(files):
        print()
        rows.append(analyze(f))

    with open(args.out, "w", newline="") as w:
        writer = csv.writer(w)
        writer.writerow(["file", "size_mb", "params"])
        writer.writerows(rows)

    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()