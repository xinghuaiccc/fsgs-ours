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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 1
        self._source_path = ""
        self._model_path = ""
        self._images = "images_8"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.n_views = 0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.05
        self.densification_interval = 100
        self.opacity_reset_interval = 30_000
        self.densify_from_iter = 500
        self.prune_from_iter = 500
        self.densify_until_iter = 4_000
        self.densify_grad_threshold = 0.0005
        self.unpooling_grad_threshold = 0.0002
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 9500
        self.sample_pseudo_interval = 100
        self.dist_thres = 10.
        self.depth_weight = 0.05
        self.mvs_cycle_weight = 0.0
        self.mvs_occ_epsilon = 0.05
        self.depth_pseudo_weight = 0.5
        self.grad_loss_weight = 0.1
        self.sh_sparsity_weight = 0.2
        self.depth_shape_weight = 0.1
        self.loss_print_interval = 0
        self.monotonic_psnr = False
        self.monotonic_start_iter = 1000
        self.monotonic_psnr_tolerance = 0.0
        self.monotonic_lr_decay = 0.5
        self.monotonic_reg_growth = 1.2
        self.monotonic_freeze_densify = True
        self.monotonic_freeze_sh = True
        self.monotonic_restore_best = True
        self.monotonic_retest = False
        self.capacity_stage1_end = 1000
        self.capacity_stage2_end = 3000
        self.capacity_freeze_sh_stage1 = True
        self.capacity_freeze_densify_stage1 = True
        self.rpc_weight = 0.02
        self.rpc_interval = 50
        self.rpc_sample_size = 4096
        self.sh_angular_weight = 0.0
        self.dino_weight = 0.0
        self.dino_interval = 50
        self.dino_resize = 224
        self.dino_model = "dinov2_vits14"
        self.gfc_enable = False
        self.gfc_stage1_end = 3000
        self.gfc_feature_lr_scale_stage1 = 0.0
        self.gfc_depth_weight = 0.05
        self.gfc_freeze_sh = True
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
