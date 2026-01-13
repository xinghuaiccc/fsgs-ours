#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from typing import List


LLFF8 = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]


def build_cmd(python_bin: str, train_py: str, src: str, out: str, args, extra: List[str]) -> List[str]:
    cmd = [
        python_bin,
        train_py,
        "--source_path",
        src,
        "--model_path",
        out,
        "--eval",
        "--n_views",
        str(args.n_views),
        "--iterations",
        str(args.iterations),
        "--sample_pseudo_interval",
        str(args.sample_pseudo_interval),
        "--mvs_cycle_weight",
        str(args.mvs_cycle_weight),
        "--mvs_occ_epsilon",
        str(args.mvs_occ_epsilon),
    ]
    cmd.extend(extra)
    return cmd


def run_scene(scene: str, args, extra: List[str]) -> None:
    src = os.path.join(args.data_root, scene)
    out = f"{args.model_prefix}_{scene}"
    cmd = build_cmd(sys.executable, args.train_py, src, out, args, extra)
    print(f"=== Running {scene} -> {out} ===")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLFF 8 scenes sequentially with MVS+occ cycle settings."
    )
    parser.add_argument(
        "--data_root",
        default=os.getenv("DATA_ROOT", "/root/all-data/nerf_llff_data"),
        help="Root directory containing LLFF scenes.",
    )
    parser.add_argument(
        "--model_prefix",
        default=os.getenv("MODEL_PREFIX", "output/llff_mvs_occ_cycle"),
        help="Prefix for output model paths.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("ITERS", 5000)),
        help="Training iterations per scene.",
    )
    parser.add_argument(
        "--n_views", type=int, default=3, help="Number of views for training."
    )
    parser.add_argument(
        "--sample_pseudo_interval",
        type=int,
        default=1,
        help="Pseudo sample interval passed to train.py.",
    )
    parser.add_argument(
        "--mvs_cycle_weight",
        type=float,
        default=0.1,
        help="Weight for MVS cycle loss.",
    )
    parser.add_argument(
        "--mvs_occ_epsilon",
        type=float,
        default=0.05,
        help="Relative depth tolerance for occlusion masking.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=LLFF8,
        help="Scenes to run (default: LLFF 8 scenes).",
    )
    parser.add_argument(
        "--train_py",
        default="train.py",
        help="Path to train.py (relative or absolute).",
    )
    args, extra = parser.parse_known_args()

    for scene in args.scenes:
        run_scene(scene, args, extra)

    print("All scenes completed.")


if __name__ == "__main__":
    main()
