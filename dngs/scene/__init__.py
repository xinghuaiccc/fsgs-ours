# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #
#
# import os
# import random
# import json
# from dngs.utils.system_utils import searchForMaxIteration
# from dngs.scene.dataset_readers import sceneLoadTypeCallbacks
# from dngs.scene.gaussian_model import GaussianModel
# from dngs.scene.gaussian_model_sh import GaussianModelSH
# from dngs.arguments import ModelParams
# from dngs.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos
#
# class Scene:
#
#     gaussians : GaussianModel
#
#     def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         self.model_path = args.model_path
#         self.source_path = args.source_path
#         self.loaded_iter = None
#         self.gaussians = gaussians
#
#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
#             else:
#                 self.loaded_iter = load_iteration
#             print("Loading trained model at iteration {}".format(self.loaded_iter))
#
#         self.train_cameras = {}
#         self.test_cameras = {}
#         self.eval_cameras = {}
#
#         if os.path.exists(os.path.join(args.source_path, "sparse")):
#             scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.dataset, args.eval, args.rand_pcd, args.mvs_pcd, N_sparse = args.n_sparse)
#         elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
#             print("Found transforms_train.json file, assuming Blender data set!")
#             scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.rand_pcd, N_sparse = args.n_sparse)
#         else:
#             assert False, "Could not recognize scene type!"
#
#         if not self.loaded_iter:
#             with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
#                 dest_file.write(src_file.read())
#             json_cams = []
#             camlist = []
#             if scene_info.test_cameras:
#                 camlist.extend(scene_info.test_cameras)
#             if scene_info.train_cameras:
#                 camlist.extend(scene_info.train_cameras)
#             if scene_info.eval_cameras:
#                 camlist.extend(scene_info.eval_cameras)
#             for id, cam in enumerate(camlist):
#                 json_cams.append(camera_to_JSON(id, cam))
#             with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
#                 json.dump(json_cams, file)
#
#         if shuffle:
#             random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(scene_info.eval_cameras)  # Multi-res consistent random shuffling
#
#         self.cameras_extent = scene_info.nerf_normalization["radius"]
#
#         for resolution_scale in resolution_scales:
#             print("Loading Training Cameras", resolution_scales)
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
#             print("Loading Test Cameras", resolution_scales)
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
#             print("Loading Eval Cameras", resolution_scales)
#             self.eval_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.eval_cameras, resolution_scale, args)
#
#         if self.loaded_iter:
#             self.gaussians.load_ply(os.path.join(self.model_path,
#                                                            "point_cloud",
#                                                            "iteration_" + str(self.loaded_iter),
#                                                            "point_cloud.ply"))
#         else:
#             self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
#
#     def save(self, iteration, color=None):
#         point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
#         self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
#         if color is not None:
#             self.gaussians.save_ply_color(os.path.join(point_cloud_path, "point_cloud_color.ply"), color)
#
#     def getTrainCameras(self, scale=1.0):
#         return self.train_cameras[scale]
#
#     def getTestCameras(self, scale=1.0):
#         return self.test_cameras[scale]
#
#     def getEvalCameras(self, scale=1.0):
#         return self.eval_cameras[scale]
#
#
#
#
# class RenderScene:
#
#     gaussians : GaussianModel
#
#     def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, spiral=True, resolution_scales=[1.0]):
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         self.model_path = args.model_path
#         self.loaded_iter = None
#         self.gaussians = gaussians
#
#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
#             else:
#                 self.loaded_iter = load_iteration
#             print("Loading trained model at iteration {}".format(self.loaded_iter))
#
#         self.test_cameras = {}
#
#         if 'scan' in args.source_path:
#             scene_info = sceneLoadTypeCallbacks["SpiralDTU"](args.source_path)
#         else:
#             scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)
#
#         self.cameras_extent = scene_info.nerf_normalization["radius"]
#
#         for resolution_scale in resolution_scales:
#             print("Loading Render Cameras", resolution_scales)
#             self.test_cameras[resolution_scale] = renderCameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
#
#         if self.loaded_iter:
#             self.gaussians.load_ply(os.path.join(self.model_path,
#                                                            "point_cloud",
#                                                            "iteration_" + str(self.loaded_iter),
#                                                            "point_cloud.ply"))
#         else:
#             pass
#
#
#     def getRenderCameras(self, scale=1.0):
#         return self.test_cameras[scale]


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
import random
import json
from dngs.utils.system_utils import searchForMaxIteration
from dngs.scene.dataset_readers import sceneLoadTypeCallbacks
from dngs.scene.gaussian_model import GaussianModel
from dngs.scene.gaussian_model_sh import GaussianModelSH
from dngs.arguments import ModelParams
from dngs.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.eval_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.dataset, args.eval,
                                                          args.rand_pcd, args.mvs_pcd, N_sparse=args.n_sparse)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,
                                                           args.rand_pcd, N_sparse=args.n_sparse)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.eval_cameras:
                camlist.extend(scene_info.eval_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.eval_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", resolution_scales)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
            print("Loading Eval Cameras", resolution_scales)
            self.eval_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.eval_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, color=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if color is not None:
            self.gaussians.save_ply_color(os.path.join(point_cloud_path, "point_cloud_color.ply"), color)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getEvalCameras(self, scale=1.0):
        return self.eval_cameras[scale]


class RenderScene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, spiral=True,
                 resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}

        # --- 核心修改：适配 Tanks 数据集，跳过螺旋轨迹生成 ---
        from types import SimpleNamespace
        if 'scan' in args.source_path:
            scene_info = sceneLoadTypeCallbacks["SpiralDTU"](args.source_path)
        elif 'Tanks' in args.source_path or 'tandt_db' in args.source_path:
            # Tanks 数据集不包含 poses_bounds.npy，无法生成螺旋轨迹，这里创建一个空信息以跳过
            print("Detected Tanks dataset, skipping spiral render path generation due to missing poses_bounds.npy")
            scene_info = SimpleNamespace(test_cameras=[], nerf_normalization={"radius": 1.0})
        else:
            scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)
        # --- 修改结束 ---

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Render Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = renderCameraList_from_camInfos(scene_info.test_cameras,
                                                                                 resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            pass

    def getRenderCameras(self, scale=1.0):
        return self.test_cameras[scale]
