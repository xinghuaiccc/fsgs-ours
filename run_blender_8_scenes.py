#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run training over the 8 Blender scenes with DNGS defaults."
    )
    parser.add_argument(
        "--blender_root",
        default=os.path.join("..", "small-data", "nerf_synthetic_dngaussian"),
        help="Root directory containing Blender scenes.",
    )
    parser.add_argument("--iterations", type=int, default=7_000)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--train_script", default="train_blender_ours.py")
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated list of scene names to run (default: all 8).",
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Enable white background for Blender scenes.",
    )
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--prune_threshold", type=float, default=0.005)
    parser.add_argument("--densify_until_iter", type=int, default=5_000)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_max_steps", type=int, default=7_000)
    parser.add_argument("--position_lr_start", type=int, default=0)
    parser.add_argument("--hard_depth_start", type=int, default=99_999)
    parser.add_argument("--error_tolerance", type=float, default=0.2)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--shape_pena", type=float, default=0.0)
    parser.add_argument("--opa_pena", type=float, default=0.0)
    parser.add_argument("--scale_pena", type=float, default=0.0)
    parser.add_argument("--no_sh", action="store_true", default=False)
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

    available = {
        d
        for d in os.listdir(blender_root)
        if os.path.isdir(os.path.join(blender_root, d))
    }
    if args.scenes:
        requested = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = sorted([s for s in default_scenes if s in requested])
    else:
        scenes = [s for s in default_scenes if s in available]

    if not scenes:
        print(f"No scenes found under: {blender_root}", file=sys.stderr)
        return 1

    for scene_name in scenes:
        scene_path = os.path.join(blender_root, scene_name)
        model_path = os.path.join("output", "blender", scene_name)
        print(f"==> Training Blender scene: {scene_name}")
        cmd = [
            sys.executable,
            args.train_script,
            "--source_path",
            scene_path,
            "--model_path",
            model_path,
            "--resolution",
            str(args.resolution),
            "--eval",
            "--rand_pcd",
            "--iterations",
            str(args.iterations),
            "--lambda_dssim",
            str(args.lambda_dssim),
            "--densify_grad_threshold",
            str(args.densify_grad_threshold),
            "--prune_threshold",
            str(args.prune_threshold),
            "--densify_until_iter",
            str(args.densify_until_iter),
            "--percent_dense",
            str(args.percent_dense),
            "--densify_from_iter",
            str(args.densify_from_iter),
            "--position_lr_init",
            str(args.position_lr_init),
            "--position_lr_final",
            str(args.position_lr_final),
            "--position_lr_max_steps",
            str(args.position_lr_max_steps),
            "--position_lr_start",
            str(args.position_lr_start),
            "--test_iterations",
            str(args.iterations),
            "--save_iterations",
            str(args.iterations),
            "--hard_depth_start",
            str(args.hard_depth_start),
            "--error_tolerance",
            str(args.error_tolerance),
            "--scaling_lr",
            str(args.scaling_lr),
            "--shape_pena",
            str(args.shape_pena),
            "--opa_pena",
            str(args.opa_pena),
            "--scale_pena",
            str(args.scale_pena),
        ]
        if args.white_background:
            cmd.append("--white_background")
        if not args.no_sh:
            cmd.append("--use_SH")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
