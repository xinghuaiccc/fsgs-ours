#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import math
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import BasicPointCloud, geom_transform_points
from utils.sh_utils import eval_sh
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips


def get_gradient(image):
    # 计算图像梯度 (Edge Map)
    # image shape: [1, 3, H, W]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dx, dy


def depth_gradients(depth):
    # depth: [H, W]
    dx = depth[:, 1:] - depth[:, :-1]
    dy = depth[1:, :] - depth[:-1, :]
    return dx, dy


def sh_colors_for_camera(gaussians, camera):
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
    return torch.clamp_min(sh2rgb + 0.5, 0.0)


def sample_image_at(image, coords):
    grid = coords.view(1, -1, 1, 2)
    samples = F.grid_sample(image[None], grid, align_corners=True, mode="bilinear")
    return samples.squeeze(0).squeeze(-1).transpose(0, 1)


def sh_angular_weights(max_sh_degree, device):
    weights = []
    for l in range(1, max_sh_degree + 1):
        weights.extend([float(l)] * (2 * l + 1))
    return torch.tensor(weights, device=device).view(1, 1, -1)


def _maybe_patch_dinov2_cache(hub_dir):
    import re
    repo_dirs = [
        os.path.join(hub_dir, name)
        for name in os.listdir(hub_dir)
        if name.startswith("facebookresearch_dinov2_")
    ]
    if not repo_dirs:
        return False
    patched_any = False
    for repo_dir in repo_dirs:
        for root, _, files in os.walk(repo_dir):
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(root, filename)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        text = handle.read()
                except OSError:
                    continue
                if re.search(r"\|\s*None", text) is None:
                    continue
                if "from __future__ import annotations" in text:
                    continue
                lines = text.splitlines()
                insert_at = 0
                if lines and lines[0].startswith("#!"):
                    insert_at = 1
                if insert_at < len(lines) and "coding" in lines[insert_at]:
                    insert_at += 1
                if insert_at < len(lines) and (lines[insert_at].startswith('"""') or lines[insert_at].startswith("'''")):
                    quote = lines[insert_at][:3]
                    insert_at += 1
                    while insert_at < len(lines) and quote not in lines[insert_at]:
                        insert_at += 1
                    if insert_at < len(lines):
                        insert_at += 1
                lines.insert(insert_at, "from __future__ import annotations")
                try:
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")
                    patched_any = True
                except OSError:
                    continue
    return patched_any


def _load_dinov2_from_cache(model_name, device):
    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    if not os.path.isdir(repo_dir):
        return None
    try:
        model = torch.hub.load(repo_dir, model_name, source="local")
        model.feature_kind = "dinov2"
        model.eval().to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        return model
    except Exception:
        return None


def _ensure_scaled_dot_product_attention():
    if hasattr(F, "scaled_dot_product_attention"):
        return

    def _sdp_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        d_k = query.shape[-1]
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if is_causal:
            q_len = attn.shape[-2]
            k_len = attn.shape[-1]
            causal_mask = torch.triu(torch.ones(q_len, k_len, device=attn.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float("-inf"))
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = _sdp_attention


def load_dino_model(model_name, device):
    if model_name.startswith("torchvision_"):
        model_id = model_name[len("torchvision_"):]
        try:
            from torchvision import models
        except Exception as exc:
            print(f"Warning: torchvision unavailable for '{model_name}': {exc}")
            return None
        if model_id == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)
            model.feature_kind = "torchvision_vit"
        else:
            print(f"Warning: unknown torchvision model '{model_id}'.")
            return None
    else:
        try:
            _ensure_scaled_dot_product_attention()
            model = torch.hub.load("facebookresearch/dinov2", model_name)
            model.feature_kind = "dinov2"
        except Exception as exc:
            if "unsupported operand type(s) for |" in str(exc):
                hub_dir = torch.hub.get_dir()
                if _maybe_patch_dinov2_cache(hub_dir):
                    _ensure_scaled_dot_product_attention()
                    model = _load_dinov2_from_cache(model_name, device)
                    if model is not None:
                        return model
            print(f"Warning: failed to load DINOv2 model '{model_name}': {exc}")
            if model_name != "torchvision_vit_b_16":
                print("Falling back to torchvision_vit_b_16.")
                return load_dino_model("torchvision_vit_b_16", device)
            return None
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def dino_forward(model, image, resize, detach):
    image_b = image.unsqueeze(0)
    if resize and (image_b.shape[2] != resize or image_b.shape[3] != resize):
        image_b = F.interpolate(image_b, size=(resize, resize), mode="bilinear", align_corners=False)
    if detach:
        image_b = image_b.detach()
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_b.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_b.device).view(1, 3, 1, 1)
    image_b = (image_b - mean) / std
    if getattr(model, "feature_kind", None) == "torchvision_vit":
        x = model._process_input(image_b)
        batch_class_token = model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        tokens = x[:, 1:, :]
    else:
        feats = model.forward_features(image_b)
        if isinstance(feats, dict):
            tokens = feats.get("x_norm_patchtokens", None)
            if tokens is None:
                tokens = feats.get("x_norm_clstoken", None)
                if tokens is not None:
                    tokens = tokens.unsqueeze(1)
        else:
            tokens = feats
    return tokens


def make_point_cloud_from_masks(views, N=100_000):
    # 创新点：利用 Mask 生成 Visual Hull 点云，替代失败的 COLMAP
    # 仅当 COLMAP 初始化失败时调用
    print("Generating point cloud from masks (Visual Hull)...")

    min_bound = torch.tensor([-1.5, -1.5, -1.5], device="cuda")
    max_bound = torch.tensor([1.5, 1.5, 1.5], device="cuda")
    xyz = (max_bound - min_bound) * torch.rand((N, 3), device="cuda") + min_bound

    valid_mask = torch.ones(N, dtype=torch.bool, device="cuda")

    for cam in views:
        full_proj_transform = cam.full_proj_transform.to(xyz.device)
        ones = torch.ones((N, 1), device="cuda")
        points_hom = torch.cat([xyz, ones], dim=1)
        p_hom = (full_proj_transform @ points_hom.T).T
        p_w = 1.0 / (p_hom[:, 3] + 1e-7)

        x = p_hom[:, 0] * p_w
        y = p_hom[:, 1] * p_w
        z = p_hom[:, 2]
        in_view = (x > -1) & (x < 1) & (y > -1) & (y < 1) & (z > 0.2)

        gt_image = cam.original_image.to(xyz.device)
        H, W = gt_image.shape[1], gt_image.shape[2]
        u = ((x + 1) * W - 1) * 0.5
        v = ((y + 1) * H - 1) * 0.5
        u = u.long().clamp(0, W - 1)
        v = v.long().clamp(0, H - 1)

        pixel_vals = gt_image[:, v, u]
        is_foreground = pixel_vals.sum(dim=0) > 0.1

        valid_mask = valid_mask & in_view & is_foreground

    final_xyz = xyz[valid_mask]
    if final_xyz.shape[0] < 100:
        print("Warning: Visual Hull produced too few points. Fallback to random cube.")
        final_xyz = (max_bound - min_bound) * torch.rand((2000, 3), device="cuda") + min_bound

    print(f"Generated {final_xyz.shape[0]} points from Visual Hull.")
    final_xyz_np = final_xyz.cpu().numpy()
    final_rgb_np = np.random.random_sample((final_xyz_np.shape[0], 3))
    return BasicPointCloud(points=final_xyz_np, colors=final_rgb_np, normals=np.zeros_like(final_xyz_np))


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    # --- Innovation: Blender Sparse Fix ---
    if dataset.white_background and gaussians.get_xyz.shape[0] < 500:
        print("Detected Blender Sparse Scene with failed COLMAP. Switching to Mask-Guided Init.")
        train_cams = scene.getTrainCameras()
        pcd = make_point_cloud_from_masks(train_cams, N=200_000)
        gaussians.create_from_pcd(pcd, scene.cameras_extent)
    # --------------------------------------
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    best_test_psnr = -1.0
    best_state = None
    best_iter = -1
    freeze_sh = False
    lr_scale = 1.0
    dino_model = None
    gfc_stage1_active = False
    base_param_lrs = None
    current_depth_weight = args.depth_weight
    if opt.dino_weight > 0.0:
        dino_model = load_dino_model(opt.dino_model, "cuda")
        if dino_model is None:
            opt.dino_weight = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if opt.gfc_enable:
            gfc_stage1_active = iteration <= opt.gfc_stage1_end
            if gfc_stage1_active:
                current_depth_weight = max(current_depth_weight, opt.gfc_depth_weight)
                if opt.gfc_freeze_sh:
                    freeze_sh = True
                if base_param_lrs is None:
                    base_param_lrs = {group.get("name"): group["lr"] for group in gaussians.optimizer.param_groups}
                for group in gaussians.optimizer.param_groups:
                    if group.get("name") in ("f_dc", "f_rest"):
                        group["lr"] = base_param_lrs[group.get("name")] * opt.gfc_feature_lr_scale_stage1
            else:
                if base_param_lrs is not None:
                    for group in gaussians.optimizer.param_groups:
                        name = group.get("name")
                        if name in base_param_lrs:
                            group["lr"] = base_param_lrs[name]
                freeze_sh = False
                current_depth_weight = args.depth_weight

        allow_sh = not freeze_sh and not (opt.capacity_freeze_sh_stage1 and iteration < opt.capacity_stage1_end)
        if iteration % 500 == 0 and allow_sh:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        alpha = render_pkg.get("alpha", None)


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

        dino_loss = None
        if opt.dino_weight > 0.0 and dino_model is not None:
            dino_interval = max(1, int(opt.dino_interval))
            if iteration % dino_interval == 0:
                gt_tokens = dino_forward(dino_model, gt_image, opt.dino_resize, detach=True)
                render_tokens = dino_forward(dino_model, image, opt.dino_resize, detach=False)
                if gt_tokens is not None and render_tokens is not None and gt_tokens.shape == render_tokens.shape:
                    gt_tokens = F.normalize(gt_tokens, dim=-1)
                    render_tokens = F.normalize(render_tokens, dim=-1)
                    dino_loss = (1.0 - (render_tokens * gt_tokens).sum(dim=-1)).mean()
                    loss += opt.dino_weight * dino_loss
                else:
                    print("Warning: DINO tokens unavailable or mismatched; skipping DINO loss.")

        depth_shape_loss = None
        if viewpoint_cam.depth_image is not None:
            rendered_depth = render_pkg["depth"][0]
            midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
            if midas_depth.shape != rendered_depth.shape:
                midas_depth = torch.nn.functional.interpolate(
                    midas_depth[None, None, ...],
                    size=rendered_depth.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            rendered_depth = rendered_depth.reshape(-1, 1)
            midas_depth = midas_depth.reshape(-1, 1)

            depth_loss = min(
                            (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                            (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
            )
            loss += current_depth_weight * depth_loss

            if iteration > args.end_sample_pseudo:
                current_depth_weight = 0.001

            # [Innovation] Scale-Invariant Depth Shape Consistency.
            pred_depth = render_pkg["depth"][0]
            gt_depth = torch.tensor(viewpoint_cam.depth_image, device=pred_depth.device)
            if gt_depth.shape != pred_depth.shape:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth[None, None, ...],
                    size=pred_depth.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            pred_norm = (pred_depth - pred_depth.mean()) / (pred_depth.std() + 1e-6)
            gt_norm = (gt_depth - gt_depth.mean()) / (gt_depth.std() + 1e-6)
            pred_dx, pred_dy = depth_gradients(pred_norm)
            gt_dx, gt_dy = depth_gradients(gt_norm)
            depth_shape_loss = (pred_dx - gt_dx).abs().mean() + (pred_dy - gt_dy).abs().mean()
            loss = loss + opt.depth_shape_weight * depth_shape_loss




        # if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
        #     if not pseudo_stack:
        #         pseudo_stack = scene.getPseudoCameras().copy()
        #     pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
        #
        #     render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
        #     rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
        #     midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')
        #
        #     rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
        #     midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
        #     depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()
        #
        #     if torch.isnan(depth_loss_pseudo).sum() == 0:
        #         loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
        #         loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

        # --- Innovation 2: Strong SASR (Structure-Appearance Dual Regularization) ---
        # [A. 结构锐化] Gradient Consistency Loss
        pred_dx, pred_dy = get_gradient(image.unsqueeze(0))
        gt_dx, gt_dy = get_gradient(gt_image.unsqueeze(0))
        grad_loss = torch.abs(pred_dx - gt_dx).mean() + torch.abs(pred_dy - gt_dy).mean()
        loss += opt.grad_loss_weight * grad_loss
        # [B. 外观防过拟合] SH Sparsity / Angular Regularization
        # 这能有效压制 Train/Test 的巨大泛化误差
        if gaussians._features_rest.numel() > 0:
            sh_sparsity_loss = gaussians._features_rest.abs().mean()
            loss += opt.sh_sparsity_weight * sh_sparsity_loss
            if opt.sh_angular_weight > 0.0:
                if not hasattr(gaussians, "_sh_ang_weights") or gaussians._sh_ang_weights.shape[-1] != gaussians._features_rest.shape[-1]:
                    gaussians._sh_ang_weights = sh_angular_weights(gaussians.max_sh_degree, gaussians._features_rest.device)
                sh_energy = (gaussians._features_rest ** 2 * gaussians._sh_ang_weights).mean()
                loss += opt.sh_angular_weight * sh_energy
        else:
            sh_sparsity_loss = torch.tensor(0.0, device=loss.device)

        rpc_weight = 0.0
        if opt.rpc_weight > 0.0 and opt.rpc_interval > 0:
            if iteration >= opt.capacity_stage1_end:
                if iteration >= opt.capacity_stage2_end:
                    rpc_weight = opt.rpc_weight
                else:
                    span = max(1, opt.capacity_stage2_end - opt.capacity_stage1_end)
                    rpc_weight = opt.rpc_weight * (iteration - opt.capacity_stage1_end) / span

        if rpc_weight > 0.0 and iteration % opt.rpc_interval == 0:
            other_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]
            if other_cam.image_name != viewpoint_cam.image_name:
                render_pkg_other = render(other_cam, gaussians, pipe, background)
                image_other = render_pkg_other["render"]
                alpha_other = render_pkg_other.get("alpha", None)
                vis_other = render_pkg_other["visibility_filter"]
                vis = visibility_filter & vis_other
                if vis.any():
                    vis_idx = vis.nonzero(as_tuple=False).squeeze(1)
                    if opt.rpc_sample_size > 0 and vis_idx.numel() > opt.rpc_sample_size:
                        perm = torch.randperm(vis_idx.numel(), device=vis_idx.device)[:opt.rpc_sample_size]
                        vis_idx = vis_idx[perm]
                    points = gaussians.get_xyz[vis_idx]
                    ndc_a = geom_transform_points(points, viewpoint_cam.full_proj_transform)[:, :2]
                    ndc_b = geom_transform_points(points, other_cam.full_proj_transform)[:, :2]
                    in_a = (ndc_a.abs().max(dim=1).values <= 1.0)
                    in_b = (ndc_b.abs().max(dim=1).values <= 1.0)
                    in_bounds = in_a & in_b
                    if in_bounds.any():
                        ndc_a = ndc_a[in_bounds]
                        ndc_b = ndc_b[in_bounds]
                        colors_a = sample_image_at(image, ndc_a)
                        colors_b = sample_image_at(image_other, ndc_b)
                        if alpha is not None and alpha_other is not None:
                            weights_a = sample_image_at(alpha, ndc_a)[:, 0]
                            weights_b = sample_image_at(alpha_other, ndc_b)[:, 0]
                            weights = (weights_a * weights_b).clamp(min=0.0)
                        else:
                            weights = torch.ones(colors_a.shape[0], device=colors_a.device)
                        diff = (colors_a - colors_b).abs().mean(dim=1)
                        denom = weights.sum() + 1e-6
                        rpc_loss = (diff * weights).sum() / denom
                        loss += rpc_weight * rpc_loss

        if opt.loss_print_interval > 0 and iteration % opt.loss_print_interval == 0:
            depth_shape_val = depth_shape_loss.item() if depth_shape_loss is not None else 0.0
            dino_val = dino_loss.item() if dino_loss is not None else 0.0
            print(
                "[ITER {}] loss_terms: l1={:.6f} ssim={:.6f} grad={:.6f} sh_sparse={:.6f} depth_shape={:.6f} dino={:.6f}".format(
                    iteration,
                    Ll1.item(),
                    (1.0 - ssim(image, gt_image)).item(),
                    grad_loss.item(),
                    sh_sparsity_loss.item(),
                    depth_shape_val,
                    dino_val,
                )
            )

        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                        testing_iterations, scene, render, (pipe, background))
            if test_psnr is not None:
                if test_psnr > best_test_psnr:
                    best_test_psnr = test_psnr
                    best_state = gaussians.capture()
                    best_iter = iteration
                elif opt.monotonic_psnr and iteration >= opt.monotonic_start_iter \
                        and test_psnr + opt.monotonic_psnr_tolerance < best_test_psnr:
                    print(
                        "[ITER {}] PSNR drop {:.4f} -> {:.4f}, applying monotonic controller.".format(
                            iteration, test_psnr, best_test_psnr
                        )
                    )
                    if opt.monotonic_restore_best and best_state is not None:
                        gaussians.restore(best_state, opt)
                    lr_scale *= opt.monotonic_lr_decay
                    for group in gaussians.optimizer.param_groups:
                        if group.get("name") != "xyz":
                            group["lr"] *= opt.monotonic_lr_decay
                    gaussians.xyz_lr_scale = lr_scale
                    opt.grad_loss_weight *= opt.monotonic_reg_growth
                    opt.sh_sparsity_weight *= opt.monotonic_reg_growth
                    if opt.monotonic_freeze_densify:
                        opt.densify_until_iter = min(opt.densify_until_iter, iteration)
                    if opt.monotonic_freeze_sh:
                        freeze_sh = True
                    if opt.monotonic_restore_best:
                        print("[ITER {}] Restored best model from iter {}.".format(iteration, best_iter))
                        if opt.monotonic_retest:
                            post_psnr = eval_test_psnr(scene, render, (pipe, background))
                            if post_psnr is not None:
                                print("[ITER {}] Post-restore test PSNR {:.4f}".format(iteration, post_psnr))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            allow_densify = iteration < opt.densify_until_iter
            if opt.capacity_freeze_densify_stage1 and iteration < opt.capacity_stage1_end:
                allow_densify = False
            if allow_densify:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def eval_test_psnr(scene: Scene, renderFunc, renderArgs):
    cameras = scene.getTestCameras()
    if not cameras:
        return None
    psnr_test = 0.0
    for viewpoint in cameras:
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        psnr_test += psnr(image, gt_image, None).mean().double()
    psnr_test /= len(cameras)
    return psnr_test


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
        test_psnr_value = None

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if config['name'] == 'test':
                    test_psnr_value = psnr_test
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        return test_psnr_value
    return None

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        raise SystemExit(
            "Blender dataset detected. Use `python train_blender_dngs.py` for Blender training."
        )

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
