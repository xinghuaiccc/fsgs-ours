#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run metrics over all LLFF scenes.")
    parser.add_argument(
        "--llff_root",
        default="/root/all-data/nerf_llff_data",
        help="Root directory containing LLFF scenes.",
    )
    parser.add_argument(
        "--output_root",
        default="output",
        help="Root directory containing trained models.",
    )
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--metrics_script", default="metrics.py")
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
        "fern",
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

    source_paths = [os.path.join(llff_root, s) for s in scenes]
    model_paths = [os.path.join(args.output_root, s) for s in scenes]
    print("==> Running metrics for scenes:", ", ".join(scenes))
    cmd = [
        sys.executable,
        args.metrics_script,
        "--source_paths",
        *source_paths,
        "--model_paths",
        *model_paths,
        "--iteration",
        str(args.iteration),
    ]
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
