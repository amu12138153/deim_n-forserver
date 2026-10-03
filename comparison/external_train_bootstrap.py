"""Launch upstream training with a narrow faster-coco-eval API compatibility fix."""
import argparse
import inspect
import runpy
import sys
from pathlib import Path


STANDARD_RANGES = {
    "small": [0, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e5**2],
}


def patch_standard_coco_ranges():
    """Adapt a mixed faster-coco-eval install without changing COCO's default ranges."""
    from faster_coco_eval import COCOeval_faster
    from faster_coco_eval.utils.pytorch import coco_eval as pytorch_eval

    if "ranges" in inspect.signature(COCOeval_faster).parameters:
        return False

    def compatible_eval(*args, **kwargs):
        ranges = kwargs.pop("ranges", None)
        if ranges is not None and ranges != STANDARD_RANGES:
            raise ValueError("Installed faster-coco-eval cannot honor nonstandard COCO area ranges")
        return COCOeval_faster(*args, **kwargs)

    pytorch_eval.COCOeval_faster = compatible_eval
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-entry", required=True)
    args, upstream_args = parser.parse_known_args()
    train_entry = Path(args.train_entry).resolve()
    if not train_entry.is_file():
        raise FileNotFoundError(train_entry)
    if patch_standard_coco_ranges():
        print("Applied standard COCO ranges compatibility for faster-coco-eval", flush=True)
    sys.path.insert(0, str(train_entry.parent.parent if train_entry.parent.name == "tools" else train_entry.parent))
    sys.argv = [str(train_entry), *upstream_args]
    runpy.run_path(str(train_entry), run_name="__main__")


if __name__ == "__main__":
    main()
