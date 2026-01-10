import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import focal2fov, fov2focal


def store_ply(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    normals = np.zeros_like(xyz, dtype=np.float32)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def align_depth_scale_shift(gt_depth, pred_depth):
    """Fit gt = scale * pred + shift with least squares on valid pixels."""
    mask = (gt_depth > 0) & (pred_depth > 0)
    if mask.sum() < 10:
        return 1.0, 0.0

    gt_vals = gt_depth[mask]
    pred_vals = pred_depth[mask]
    A = np.vstack([pred_vals, np.ones(len(pred_vals))]).T
    scale, shift = np.linalg.lstsq(A, gt_vals, rcond=None)[0]
    return scale, shift


@dataclass
class SimpleCamera:
    R: np.ndarray
    T: np.ndarray
    FovX: float
    FovY: float
    image: np.ndarray
    image_name: str


def project_points(points_world, cam, img):
    """Project world points into camera view and sample colors."""
    R = cam.R
    T = cam.T
    points_cam = (points_world @ R.T) + T

    z = points_cam[:, 2]
    valid_z = z > 0.01

    h, w = img.shape[:2]
    fx = w / (2 * np.tan(cam.FovX / 2))
    fy = h / (2 * np.tan(cam.FovY / 2))
    cx = w / 2.0
    cy = h / 2.0

    u = (points_cam[:, 0] * fx) / z + cx
    v = (points_cam[:, 1] * fy) / z + cy

    in_bound = (u >= 0) & (u < w) & (v >= 0) & (v < h) & valid_z
    u_safe = np.clip(u[in_bound], 0, w - 1).astype(np.int32)
    v_safe = np.clip(v[in_bound], 0, h - 1).astype(np.int32)
    sampled = img[v_safe, u_safe]
    return in_bound, sampled


def load_blender_cameras(source_path, white_background=True):
    transforms_path = os.path.join(source_path, "transforms_train.json")
    if not os.path.exists(transforms_path):
        return []

    with open(transforms_path, "r", encoding="utf-8") as f:
        contents = json.load(f)

    fovx = contents["camera_angle_x"]
    cameras = []
    for frame in contents["frames"]:
        frame_path = frame["file_path"]
        if not os.path.splitext(frame_path)[1]:
            frame_path = frame_path + ".png"
        if frame_path.startswith("./"):
            frame_path = frame_path[2:]
        frame_path = os.path.normpath(frame_path)
        if os.path.isabs(frame_path):
            image_path = frame_path
        else:
            image_path = os.path.join(source_path, frame_path)

        if not os.path.exists(image_path):
            continue

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # Blender (Y up, Z back) -> COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        rgb = (arr * 255.0).astype(np.uint8)

        fovy = focal2fov(fov2focal(fovx, rgb.shape[1]), rgb.shape[0])
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        cameras.append(
            SimpleCamera(
                R=R,
                T=T,
                FovX=fovx,
                FovY=fovy,
                image=rgb,
                image_name=image_name,
            )
        )
    return cameras


def main():
    parser = ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument(
        "--depths_path",
        type=str,
        required=True,
        help="Path to Depth Anything .npy files",
    )
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="Use white background for Blender RGBA images.",
    )
    args = parser.parse_args()

    use_blender = os.path.exists(os.path.join(args.source_path, "transforms_train.json"))

    if use_blender:
        print("Detected Blender dataset. Loading transforms_train.json...")
        cam_infos = load_blender_cameras(args.source_path, white_background=args.white_background)
        if not cam_infos:
            raise RuntimeError("No Blender cameras loaded. Check transforms_train.json paths.")
        kdtree = None
        fused_xyz = []
        fused_rgb = []
    else:
        print("Reading COLMAP info...")
        scene_info = readColmapSceneInfo(args.source_path, "images", False)
        cam_infos = scene_info.train_cameras
        pcd_colmap = scene_info.point_cloud

        xyz_colmap = np.asarray(pcd_colmap.points)
        rgb_colmap = np.asarray(pcd_colmap.colors)

        if xyz_colmap.size == 0:
            raise RuntimeError("COLMAP point cloud is empty. Cannot align depths.")

        print("Building KDTree for COLMAP points...")
        kdtree = cKDTree(xyz_colmap)

        fused_xyz = [xyz_colmap]
        fused_rgb = [rgb_colmap]

    print("Processing views for dense fusion...")
    for idx, cam in enumerate(cam_infos):
        depth_name = cam.image_name + ".npy"
        depth_path = os.path.join(args.depths_path, depth_name)
        if not os.path.exists(depth_path):
            continue

        pred_depth = np.load(depth_path)

        img = np.array(cam.image)
        h, w = img.shape[:2]
        pred_depth = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_NEAREST)

        if use_blender:
            aligned_depth = pred_depth
        else:
            R = cam.R
            T = cam.T
            p_cam_colmap = (xyz_colmap @ R.T) + T
            z_colmap = p_cam_colmap[:, 2]

            fovx = cam.FovX
            fovy = cam.FovY
            fx = w / (2 * np.tan(fovx / 2))
            fy = h / (2 * np.tan(fovy / 2))
            cx = w / 2.0
            cy = h / 2.0

            u_col = ((p_cam_colmap[:, 0] * fx) / z_colmap + cx).astype(np.int32)
            v_col = ((p_cam_colmap[:, 1] * fy) / z_colmap + cy).astype(np.int32)

            valid = (u_col >= 0) & (u_col < w) & (v_col >= 0) & (v_col < h) & (z_colmap > 0)
            gt_depths = np.zeros_like(pred_depth)
            gt_depths[v_col[valid], u_col[valid]] = z_colmap[valid]

            scale, shift = align_depth_scale_shift(gt_depths, pred_depth)
            print(f"View {idx}: Scale={scale:.4f}, Shift={shift:.4f}")
            aligned_depth = pred_depth * scale + shift

        stride = 4
        y_grid, x_grid = np.meshgrid(
            np.arange(0, h, stride),
            np.arange(0, w, stride),
            indexing="ij",
        )
        z_grid = aligned_depth[0:h:stride, 0:w:stride]
        rgb_grid = img[0:h:stride, 0:w:stride] / 255.0

        x_cam = (x_grid - cx) * z_grid / fx
        y_cam = (y_grid - cy) * z_grid / fy

        xyz_cam = np.stack([x_cam, y_cam, z_grid], axis=-1).reshape(-1, 3)
        colors_flat = rgb_grid.reshape(-1, 3)

        valid_depth = xyz_cam[:, 2] > 0.01
        xyz_cam = xyz_cam[valid_depth]
        colors_flat = colors_flat[valid_depth]

        R = cam.R
        T = cam.T
        xyz_world = (xyz_cam - T) @ R

        check_idx = idx + 1 if idx < len(cam_infos) - 1 else idx - 1
        if check_idx < 0 or check_idx >= len(cam_infos):
            continue
        src_cam = cam_infos[check_idx]
        src_img = np.array(src_cam.image) / 255.0

        in_bound, src_sampled_colors = project_points(xyz_world, src_cam, src_img)
        ref_colors = colors_flat[in_bound]
        valid_points = xyz_world[in_bound]

        diff = np.abs(ref_colors - src_sampled_colors).mean(axis=1)
        color_consistent = diff < 0.15

        good_points = valid_points[color_consistent]
        good_colors = ref_colors[color_consistent]

        if len(good_points) > 0:
            if kdtree is not None:
                dists, _ = kdtree.query(good_points, k=1)
                is_new_structure = dists > 0.05
            else:
                is_new_structure = np.ones(len(good_points), dtype=bool)

            final_xyz = good_points[is_new_structure]
            final_rgb = good_colors[is_new_structure]

            if len(final_xyz) > 20000:
                choice = np.random.choice(len(final_xyz), 20000, replace=False)
                final_xyz = final_xyz[choice]
                final_rgb = final_rgb[choice]

            fused_xyz.append(final_xyz)
            fused_rgb.append(final_rgb)

    fused_xyz = np.concatenate(fused_xyz, axis=0)
    fused_rgb = np.concatenate(fused_rgb, axis=0)

    print(f"Final Fused Point Cloud Size: {fused_xyz.shape[0]}")

    save_path = os.path.join(args.source_path, "points3D_fused.ply")
    store_ply(save_path, fused_xyz, (fused_rgb * 255).astype(np.uint8))
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
