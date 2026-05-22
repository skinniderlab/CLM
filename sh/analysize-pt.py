import argparse
from pathlib import Path
import torch

# python sh/analysize-pt.py

def count_parameters(obj):
    if isinstance(obj, dict):
        state_dict = obj["state_dict"] if "state_dict" in obj else obj

        tensor_items = {
            k: v for k, v in state_dict.items() if torch.is_tensor(v)
        }

        if tensor_items:
            return sum(v.numel() for v in tensor_items.values())

    elif hasattr(obj, "parameters"):
        return sum(p.numel() for p in obj.parameters())

    return None


def analyze_file(file):
    size_mb = file.stat().st_size / (1024 ** 2)

    try:
        obj = torch.load(file, map_location="cpu")
        params = count_parameters(obj)

    except Exception as e:
        print(f"{file}")
        print(f"  ERROR: {e}")
        return

    print(f"{file}")
    print(f"  Size       : {size_mb:.2f} MB")

    if params is not None:
        print(f"  Parameters : {params:,}")
    else:
        print(f"  Parameters : UNKNOWN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [path]

    elif path.is_dir():
        files = list(path.rglob("*.pt"))
        files += list(path.rglob("*.pth"))

    else:
        raise ValueError(f"Invalid path: {path}")

    if not files:
        print("No .pt/.pth files found.")
        return

    for file in sorted(files):
        analyze_file(file)
        print()


if __name__ == "__main__":
    main()