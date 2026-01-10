#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run rendering over all LLFF scenes.")
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
    parser.add_argument("--render_script", default="render.py")
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated list of scene names to run (default: all).",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render_depth", action="store_true")
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

    for scene_name in scenes:
        scene_path = os.path.join(llff_root, scene_name)
        model_path = os.path.join(args.output_root, scene_name)
        print(f"==> Rendering LLFF scene: {scene_name}")
        cmd = [
            sys.executable,
            args.render_script,
            "--source_path",
            scene_path,
            "--model_path",
            model_path,
            "--iteration",
            str(args.iteration),
        ]
        if args.skip_train:
            cmd.append("--skip_train")
        if args.skip_test:
            cmd.append("--skip_test")
        if args.quiet:
            cmd.append("--quiet")
        if args.video:
            cmd.append("--video")
            cmd.extend(["--fps", str(args.fps)])
        if args.render_depth:
            cmd.append("--render_depth")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
