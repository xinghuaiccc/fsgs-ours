#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run training over all LLFF scenes.")
    parser.add_argument(
        "--llff_root",
        default="/root/all-data/nerf_llff_data",
        help="Root directory containing LLFF scenes.",
    )
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--sample_pseudo_interval", type=int, default=1)
    parser.add_argument("--train_script", default="train.py")
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated list of scene names to run (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    llff_root = args.llff_root
    if not os.path.isdir(llff_root):
        print(f"LLFF root not found: {llff_root}", file=sys.stderr)
        return 1

    default_scenes = [
        # "fern",
        "flower",
        "fortress",
        "horns",
        "leaves",
        "orchids",
        "room",
        "trex",
    ]
    available = {
        d for d in os.listdir(llff_root)
        if os.path.isdir(os.path.join(llff_root, d))
    }
    if args.scenes:
        requested = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = sorted([s for s in default_scenes if s in requested])
    else:
        scenes = [s for s in default_scenes if s in available]
    if not scenes:
        print(f"No scenes found under: {llff_root}", file=sys.stderr)
        return 1

    for scene_name in scenes:
        scene_path = os.path.join(llff_root, scene_name)
        model_path = os.path.join("output", scene_name)
        print(f"==> Training LLFF scene: {scene_name}")
        cmd = [
            sys.executable,
            args.train_script,
            "--source_path",
            scene_path,
            "--model_path",
            model_path,
            "--eval",
            "--n_views",
            str(args.n_views),
            "--sample_pseudo_interval",
            str(args.sample_pseudo_interval),
        ]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
