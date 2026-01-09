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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, device):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(gt_dir)):
        gt_path = gt_dir / fname
        render_path = renders_dir / fname
        if not gt_path.is_file() or not render_path.is_file():
            continue
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device))
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device))
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, device):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            if not test_dir.exists():
                print("  Skipping, missing test directory:", test_dir)
                continue

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                if not gt_dir.exists() or not renders_dir.exists():
                    print("  Skipping, missing gt/renders:", method_dir)
                    continue
                renders, gts, image_names = readImages(renders_dir, gt_dir, device)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                if len(ssims) == 0:
                    print("  Skipping, no valid image pairs")
                    continue
                ssims_t = torch.stack(ssims)
                psnrs_t = torch.stack(psnrs)
                lpipss_t = torch.stack(lpipss)

                print("  SSIM : {:>12.7f}".format(ssims_t.mean().item()))
                print("  PSNR : {:>12.7f}".format(psnrs_t.mean().item()))
                print("  LPIPS: {:>12.7f}".format(lpipss_t.mean().item()))
                print("")

                full_dict[scene_dir][method].update({"SSIM": ssims_t.mean().item(),
                                                        "PSNR": psnrs_t.mean().item(),
                                                        "LPIPS": lpipss_t.mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(ssims_t.tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(psnrs_t.tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(lpipss_t.tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', '-s', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args()
    # if not os.path.exists(os.path.join(args.model_paths[0], 'test', 'ours_'+str(args.iteration), 'renders')):
    # os.system('python render.py -s /mnt/vita-nas/zehao/nerf_llff_data/horns/ --iteration ' + str(args.iteration) + ' -m ' +args.model_paths[0])
    evaluate(args.model_paths, device)
