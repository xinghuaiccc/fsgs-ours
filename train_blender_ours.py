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

import os
import sys
import uuid
import torch
import numpy as np
from random import randint
from types import SimpleNamespace

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "dngs", "submodules", "diff-gaussian-rasterization"))
sys.path.insert(0, os.path.join(ROOT_DIR, "dngs", "submodules", "simple-knn"))

from dngs.utils.graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix
from dngs.utils.loss_utils import (
    l1_loss,
    loss_depth_smoothness,
    patch_norm_mse_loss,
    patch_norm_mse_loss_global,
    ssim,
)
from dngs.gaussian_renderer import render, render_for_depth, render_for_opa
from dngs.gaussian_renderer import render_sh, render_for_depth_sh
from dngs.scene import Scene, GaussianModel, GaussianModelSH
from dngs.utils.general_utils import safe_state
from dngs.utils.warping import warp_image_based_on_depth
from dngs.utils.image_utils import psnr
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from dngs.arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    print("Launch TensorBoard")
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def add_virtual_view_loss(
    viewpoint_cam, gaussians, pipe, background, aligned_depth, weight, render_func
):
    angle = (torch.rand(1, device=aligned_depth.device) * 0.2 - 0.1).item()
    axis = torch.randn(3, device=aligned_depth.device)
    axis = axis / (axis.norm() + 1e-8)

    kx, ky, kz = axis
    zero = torch.tensor(0.0, device=aligned_depth.device)
    k_mat = torch.stack(
        [torch.stack([zero, -kz, ky]),
         torch.stack([kz, zero, -kx]),
         torch.stack([-ky, kx, zero])],
        dim=0,
    )
    eye = torch.eye(3, device=aligned_depth.device)
    rot_perturb = eye + torch.sin(torch.tensor(angle, device=aligned_depth.device)) * k_mat + \
        (1.0 - torch.cos(torch.tensor(angle, device=aligned_depth.device))) * (k_mat @ k_mat)

    base_R = torch.as_tensor(viewpoint_cam.R, device=aligned_depth.device, dtype=torch.float32)
    base_T = torch.as_tensor(viewpoint_cam.T, device=aligned_depth.device, dtype=torch.float32)
    delta_t = (torch.rand(3, device=aligned_depth.device) * 0.1 - 0.05)

    new_R = (rot_perturb @ base_R).detach().cpu().numpy()
    new_T = (base_T + delta_t).detach().cpu().numpy()

    trans = getattr(viewpoint_cam, "trans", None)
    if trans is None:
        trans = np.array([0.0, 0.0, 0.0])
    elif isinstance(trans, torch.Tensor):
        trans = trans.detach().cpu().numpy()
    scale = getattr(viewpoint_cam, "scale", 1.0)
    world_view = torch.tensor(
        getWorld2View2(new_R, new_T, trans, scale),
        device=aligned_depth.device,
    ).transpose(0, 1)
    proj = getProjectionMatrix(
        znear=viewpoint_cam.znear,
        zfar=viewpoint_cam.zfar,
        fovX=viewpoint_cam.FoVx,
        fovY=viewpoint_cam.FoVy,
    ).to(aligned_depth.device).transpose(0, 1)
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cam_center = world_view.inverse()[3, :3]

    virtual_cam = SimpleNamespace(
        FoVx=viewpoint_cam.FoVx,
        FoVy=viewpoint_cam.FoVy,
        image_width=viewpoint_cam.image_width,
        image_height=viewpoint_cam.image_height,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
        camera_center=cam_center,
        znear=viewpoint_cam.znear,
        zfar=viewpoint_cam.zfar,
    )

    h, w = aligned_depth.shape
    fx = fov2focal(viewpoint_cam.FoVx, w)
    fy = fov2focal(viewpoint_cam.FoVy, h)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    k_src = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=aligned_depth.device,
        dtype=aligned_depth.dtype,
    )

    aligned_depth_in = aligned_depth.unsqueeze(0).unsqueeze(0)
    warped_image, valid_mask = warp_image_based_on_depth(
        viewpoint_cam.original_image.unsqueeze(0),
        aligned_depth_in,
        k_src,
        viewpoint_cam.world_view_transform,
        virtual_cam.world_view_transform,
    )

    render_virtual = render_func(virtual_cam, gaussians, pipe, background)
    pred_virtual = render_virtual["render"]

    valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.sum() > 0:
        warped_masked = warped_image * valid_mask
        pred_masked = pred_virtual.unsqueeze(0) * valid_mask
        loss_virtual = torch.abs(warped_masked - pred_masked).sum() / (valid_mask.sum() * 3.0)
        return weight * loss_virtual
    return torch.tensor(0.0, device=aligned_depth.device)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    depth_smoothness_start,
    depth_smoothness_weight,
    virtual_view_start,
    virtual_view_weight,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, _) = torch.load(checkpoint)
        gaussians.load_shape(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1

    patch_range = (5, 17)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg_mask = (gt_image.min(0, keepdim=True).values > 254 / 255)

        # -------------------------------------------------- hard --------------------------------------------
        if iteration > opt.hard_depth_start and iteration < opt.densify_until_iter and iteration % 10 == 0:
            render_pkg = render_for_depth(viewpoint_cam, gaussians, pipe, background)
            depth = render_pkg["depth"]

            loss_hard = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono
            depth_mono[bg_mask] = 0

            loss_l2_dpt = patch_norm_mse_loss(
                depth[None, ...],
                depth_mono[None, ...],
                randint(patch_range[0], patch_range[1]),
                opt.error_tolerance,
            )
            loss_hard += 0.1 * loss_l2_dpt

            if iteration > depth_smoothness_start:
                loss_hard += depth_smoothness_weight * loss_depth_smoothness(
                    depth[None, ...], depth_mono[None, ...]
                )

            loss_global = patch_norm_mse_loss_global(
                depth[None, ...],
                depth_mono[None, ...],
                randint(patch_range[0], patch_range[1]),
                opt.error_tolerance,
            )
            loss_hard += 1 * loss_global

            loss_hard.backward()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------- soft --------------------------------------------
        if iteration > opt.soft_depth_start and iteration < opt.densify_until_iter and iteration % 10 == 0:
            render_pkg = render_for_opa(viewpoint_cam, gaussians, pipe, background)
            viewspace_point_tensor, visibility_filter = (
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
            )
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            loss_pnt = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono
            depth_mono[bg_mask] = 0

            loss_l2_dpt = patch_norm_mse_loss(
                depth[None, ...],
                depth_mono[None, ...],
                randint(patch_range[0], patch_range[1]),
                opt.error_tolerance,
            )
            loss_pnt += 0.1 * loss_l2_dpt

            if iteration > depth_smoothness_start:
                loss_pnt += depth_smoothness_weight * loss_depth_smoothness(
                    depth[None, ...], depth_mono[None, ...]
                )

            loss_global = patch_norm_mse_loss_global(
                depth[None, ...],
                depth_mono[None, ...],
                randint(patch_range[0], patch_range[1]),
                opt.error_tolerance,
            )
            loss_pnt += 1 * loss_global

            loss_pnt.backward()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------------- Photometric --------------------------------------------
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        depth, opacity, alpha = render_pkg["depth"], render_pkg["opacity"], render_pkg["alpha"]

        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # [Innovation] Error-Guided Geometry Densification (Blender only).
        if viewpoint_cam.depth_mono is not None:
            render_depth = depth.squeeze()
            d_render_norm = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min() + 1e-6)
            midas_depth = viewpoint_cam.depth_mono.squeeze()
            d_midas_norm = (midas_depth - midas_depth.min()) / (midas_depth.max() - midas_depth.min() + 1e-6)
            depth_error = torch.abs(d_render_norm - d_midas_norm)
            loss += 0.05 * (render_depth * depth_error.detach()).mean()

        if (
            virtual_view_weight > 0.0
            and iteration > virtual_view_start
            and viewpoint_cam.depth_mono is not None
        ):
            aligned_depth = 255.0 - viewpoint_cam.depth_mono
            aligned_depth[bg_mask] = 0
            aligned_depth = aligned_depth.squeeze()
            loss = loss + add_virtual_view_loss(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                aligned_depth,
                virtual_view_weight,
                render,
            )

        # Reg
        loss_reg = torch.tensor(0.0, device=loss.device)
        shape_pena = (
            gaussians.get_scaling.max(dim=1).values
            / gaussians.get_scaling.min(dim=1).values
        ).mean()
        scale_pena = (gaussians.get_scaling.max(dim=1, keepdim=True).values ** 2).mean()
        opa_pena = 1 - (opacity[opacity > 0.2] ** 2).mean() + (
            (1 - opacity[opacity < 0.2]) ** 2
        ).mean()

        loss_reg += opt.shape_pena * shape_pena + opt.scale_pena * scale_pena + opt.opa_pena * opa_pena
        if iteration > opt.densify_until_iter:
            loss_reg *= 0.1

        loss += loss_reg

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if not loss.isnan():
                ema_loss_for_log = 0.4 * (loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            clean_iterations = testing_iterations + [first_iter]
            clean_views(iteration, clean_iterations, scene, gaussians, pipe, background, render)
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, render(viewpoint_cam, gaussians, pipe, background)["color"])

            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = max_dist = None

                    color = render(viewpoint_cam, gaussians, pipe, background)["color"]
                    white_mask = color.min(-1, keepdim=True).values > 253 / 255
                    gaussians.xyz_gradient_accum[white_mask] = 0
                    gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(
                        gaussians.opacity_activation(gaussians._opacity[white_mask]) * 0.1
                    )

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.prune_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        opt.split_opacity_thresh,
                        max_dist,
                    )

                    if "ship" in scene.source_path:
                        gaussians.prune_points(gaussians.get_xyz[:, -1] < -0.5)
                    if "hotdog" in scene.source_path:
                        gaussians.prune_points(gaussians.get_xyz[:, -1] < -0.2)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_latest.pth")


def training_sh(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    depth_smoothness_start,
    depth_smoothness_weight,
    virtual_view_start,
    virtual_view_weight,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModelSH(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, _) = torch.load(checkpoint)
        gaussians.load_shape(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1

    patch_range = (5, 17)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg_mask = (gt_image.min(0, keepdim=True).values > 254 / 255)

        # -------------------------------------------------- DEPTH --------------------------------------------
        if iteration > opt.hard_depth_start and iteration < opt.densify_until_iter and iteration % 10 == 0:
            render_pkg = render_for_depth_sh(viewpoint_cam, gaussians, pipe, background)
            depth = render_pkg["depth"]

            loss_hard = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono
            depth_mono[bg_mask] = 0

            loss_l2_dpt = patch_norm_mse_loss(
                depth[None, ...],
                depth_mono[None, ...],
                randint(patch_range[0], patch_range[1]),
                opt.error_tolerance,
            )
            loss_hard += 0.1 * loss_l2_dpt

            if iteration > depth_smoothness_start:
                loss_hard += depth_smoothness_weight * loss_depth_smoothness(
                    depth[None, ...], depth_mono[None, ...]
                )

            loss_global = patch_norm_mse_loss_global(
                depth[None, ...],
                depth_mono[None, ...],
                randint(patch_range[0], patch_range[1]),
                opt.error_tolerance,
            )
            loss_hard += 1 * loss_global

            loss_hard.backward()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------------- Photometric --------------------------------------------
        render_pkg = render_sh(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        depth, opacity, alpha = render_pkg["depth"], render_pkg["opacity"], render_pkg["alpha"]

        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # [Innovation] Error-Guided Geometry Densification (Blender only).
        if viewpoint_cam.depth_mono is not None:
            render_depth = depth.squeeze()
            d_render_norm = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min() + 1e-6)
            midas_depth = viewpoint_cam.depth_mono.squeeze()
            d_midas_norm = (midas_depth - midas_depth.min()) / (midas_depth.max() - midas_depth.min() + 1e-6)
            depth_error = torch.abs(d_render_norm - d_midas_norm)
            loss += 0.05 * (render_depth * depth_error.detach()).mean()

        if (
            virtual_view_weight > 0.0
            and iteration > virtual_view_start
            and viewpoint_cam.depth_mono is not None
        ):
            aligned_depth = 255.0 - viewpoint_cam.depth_mono
            aligned_depth[bg_mask] = 0
            aligned_depth = aligned_depth.squeeze()
            loss = loss + add_virtual_view_loss(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                aligned_depth,
                virtual_view_weight,
                render_sh,
            )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if not loss.isnan():
                ema_loss_for_log = 0.4 * (loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            clean_iterations = testing_iterations + [first_iter]
            clean_views(iteration, clean_iterations, scene, gaussians, pipe, background, render_sh)
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render_sh,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
                    )

                    if "chair" not in scene.source_path:
                        color = render_sh(viewpoint_cam, gaussians, pipe, background)["color"]
                        white_mask = color.min(-1, keepdim=True).values > 253 / 255
                        gaussians.xyz_gradient_accum[white_mask] = 0
                        gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(
                            gaussians.opacity_activation(gaussians._opacity[white_mask]) * 0.1
                        )

                    if "ship" in scene.source_path:
                        gaussians.prune_points(gaussians.get_xyz[:, -1] < -0.5)
                    if "hotdog" in scene.source_path:
                        gaussians.prune_points(gaussians.get_xyz[:, -1] < -0.2)

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_latest.pth")


def prepare_output_and_logger(args, opt):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), "w") as opt_log_f:
        opt_log_f.write(str(Namespace(**vars(opt))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background, render_func):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render_func(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts)


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    depth_loss=torch.tensor(0),
    reg_loss=torch.tensor(0),
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar("train_loss_patches/depth_kl_loss", depth_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/reg_loss", reg_loss.item(), iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "eval", "cameras": scene.getEvalCameras()},
            {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    alpha = render_results["alpha"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bg_mask = (gt_image.min(0, keepdim=True).values > 254 / 255)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}/depth".format(viewpoint.image_name),
                            depth[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}_alpha/alpha".format(viewpoint.image_name),
                            alpha[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"] + "_view_{}_alpha/mask".format(viewpoint.image_name),
                            bg_mask[None],
                            global_step=iteration,
                        )

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 6000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 6000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--use_SH", action="store_true")
    parser.add_argument("--depth_smoothness_start", type=int, default=3000)
    parser.add_argument("--depth_smoothness_weight", type=float, default=0.1)
    parser.add_argument("--virtual_view_start", type=int, default=2000)
    parser.add_argument("--virtual_view_weight", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.use_SH:
        training_sh(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.depth_smoothness_start,
            args.depth_smoothness_weight,
            args.virtual_view_start,
            args.virtual_view_weight,
        )
    else:
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.depth_smoothness_start,
            args.depth_smoothness_weight,
            args.virtual_view_start,
            args.virtual_view_weight,
        )

    print("\nTraining complete.")
