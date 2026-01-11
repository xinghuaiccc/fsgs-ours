#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the 8 Blender scenes with the SH renderer."
    )
    parser.add_argument(
        "--blender_root",
        default=os.path.join("..", "small-data", "nerf_synthetic_dngaussian"),
        help="Root directory containing Blender scenes.",
    )
    parser.add_argument(
        "--model_root",
        default=os.path.join("output", "blender"),
        help="Root directory containing trained models.",
    )
    parser.add_argument("--render_script", default="render_sh_dngs.py")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated list of scene names to render (default: all 8).",
    )
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    blender_root = args.blender_root
    if not os.path.isdir(blender_root):
        print(f"Blender root not found: {blender_root}", file=sys.stderr)
        return 1

    default_scenes = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]
    if args.scenes:
        requested = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = sorted([s for s in default_scenes if s in requested])
    else:
        scenes = default_scenes

    if not scenes:
        print("No scenes selected.", file=sys.stderr)
        return 1

    for scene_name in scenes:
        scene_path = os.path.join(blender_root, scene_name)
        model_path = os.path.join(args.model_root, scene_name)
        print(f"==> Rendering Blender scene: {scene_name}")
        cmd = [
            sys.executable,
            args.render_script,
            "--source_path",
            scene_path,
            "--model_path",
            model_path,
            "--resolution",
            str(args.resolution),
        ]
        if args.iteration != -1:
            cmd.extend(["--iteration", str(args.iteration)])
        if args.white_background:
            cmd.append("--white_background")
        if args.eval:
            cmd.append("--eval")
        if args.skip_train:
            cmd.append("--skip_train")
        if args.skip_test:
            cmd.append("--skip_test")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
