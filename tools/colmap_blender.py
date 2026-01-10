import argparse
import json
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded


def _resolve_image_path(scene_path, frame_path, extension):
    if os.path.splitext(frame_path)[1]:
        rel_path = frame_path
    else:
        rel_path = frame_path + extension
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    rel_path = os.path.normpath(rel_path)
    if os.path.isabs(rel_path):
        return rel_path
    scene_norm = os.path.normpath(scene_path)
    if rel_path.startswith(scene_norm + os.sep):
        return rel_path
    return os.path.normpath(os.path.join(scene_path, rel_path))


def _select_frames(frames, n_views):
    if n_views <= 0 or len(frames) <= n_views:
        return frames
    idx_sub = [round_python3(i) for i in np.linspace(0, len(frames) - 1, n_views)]
    return [f for idx, f in enumerate(frames) if idx in idx_sub]


def _write_colmap_model(created_dir, frames, image_paths, camera_params):
    os.makedirs(created_dir, exist_ok=True)
    cameras_txt = os.path.join(created_dir, "cameras.txt")
    images_txt = os.path.join(created_dir, "images.txt")
    points3d_txt = os.path.join(created_dir, "points3D.txt")

    cam_id = 1
    width, height, fx, fy, cx, cy = camera_params
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, (frame, img_path) in enumerate(zip(frames, image_paths), start=1):
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            # Blender/OpenGL to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat2qvec(R)
            name = os.path.basename(img_path)
            f.write(f"{idx} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n\n")

    with open(points3d_txt, "w") as f:
        f.write("")


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, shell=True, check=False)
    return result.returncode


def pipeline(scene_path, n_views, extension):
    view_path = os.path.join(scene_path, f"{n_views}_views")

    if os.path.exists(view_path):
        shutil.rmtree(view_path)
    os.makedirs(view_path, exist_ok=True)
    os.makedirs(os.path.join(view_path, "created"), exist_ok=True)
    os.makedirs(os.path.join(view_path, "triangulated"), exist_ok=True)
    os.makedirs(os.path.join(view_path, "images"), exist_ok=True)

    transforms_path = os.path.join(scene_path, "transforms_train.json")
    with open(transforms_path, "r") as f:
        contents = json.load(f)

    frames = contents["frames"]
    train_frames = _select_frames(frames, n_views)

    image_paths = []
    for frame in train_frames:
        img_path = _resolve_image_path(scene_path, frame["file_path"], extension)
        image_paths.append(img_path)
        shutil.copy(img_path, os.path.join(view_path, "images", os.path.basename(img_path)))

    first_image = Image.open(image_paths[0])
    width, height = first_image.size
    fovx = contents["camera_angle_x"]
    fx = 0.5 * width / math.tan(0.5 * fovx)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    _write_colmap_model(os.path.join(view_path, "created"), train_frames, image_paths,
                        (width, height, fx, fy, cx, cy))

    _run(
        f"colmap feature_extractor --database_path database.db --image_path images "
        f"--ImageReader.camera_model PINHOLE "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_params {fx},{fy},{cx},{cy} "
        f"--SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 "
        f"--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1",
        cwd=view_path,
    )
    match_ret = _run(
        "colmap exhaustive_matcher --database_path database.db",
        cwd=view_path,
    )
    if match_ret != 0:
        print(f"Matcher failed in {view_path}, skipping.")
        return
    tri_ret = _run(
        "colmap point_triangulator --database_path database.db --image_path images "
        "--input_path created --output_path triangulated "
        "--Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 "
        "--Mapper.ba_global_max_num_iterations 100",
        cwd=view_path,
    )
    if tri_ret != 0:
        print(f"Triangulation failed in {view_path}, skipping dense steps.")
        return
    _run("colmap model_converter --input_path triangulated --output_path triangulated --output_type TXT",
         cwd=view_path)
    _run("colmap image_undistorter --image_path images --input_path triangulated --output_path dense",
         cwd=view_path)
    _run("colmap patch_match_stereo --workspace_path dense", cwd=view_path)
    _run("colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply", cwd=view_path)


def _resolve_scene_path(base_path, scene):
    direct_path = os.path.join(base_path, scene)
    if scene.endswith("_8views"):
        return direct_path
    alt_path = os.path.join(base_path, f"{scene}_8views")
    if os.path.exists(alt_path) and not os.path.exists(direct_path):
        return alt_path
    return direct_path


def _discover_scenes(base_path):
    if os.path.isfile(os.path.join(base_path, "transforms_train.json")):
        return ["."]
    entries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    views_dirs = [d for d in entries if d.endswith("_8views")]
    if views_dirs:
        return views_dirs
    return ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def main():
    parser = argparse.ArgumentParser(description="Run COLMAP on Blender dataset with sparse training views.")
    parser.add_argument("--base_path", required=True, type=str,
                        help="Path containing Blender scenes, e.g., /data/blender/")
    parser.add_argument("--scenes", nargs="*", default=None, type=str,
                        help="Optional scene list; defaults to standard 8 scenes or detected *_8views dirs.")
    parser.add_argument("--n_views", default=8, type=int,
                        help="Number of training images to use (default: 8).")
    parser.add_argument("--extension", default=".png", type=str,
                        help="Image extension, e.g., .png or .jpg.")
    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)
    scenes = args.scenes if args.scenes else _discover_scenes(base_path)

    for scene in scenes:
        if scene == ".":
            scene_path = base_path
        else:
            scene_path = _resolve_scene_path(base_path, scene)
        pipeline(scene_path, n_views=args.n_views, extension=args.extension)


if __name__ == "__main__":
    main()
