import torch
import torch.nn as nn
import numpy as np
import os
from simple_knn._C import distCUDA2
from lib.config import cfg
from lib.utils.general_utils import quaternion_to_matrix, \
    build_scaling_rotation, \
    strip_symmetric, \
    quaternion_raw_multiply, \
    startswith_any, \
    matrix_to_quaternion, \
    quaternion_invert
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.models.gaussian_model_actor import GaussianModelActor, ActorManager
from lib.models.gaussian_model_sky import GaussinaModelSky
from bidict import bidict
from lib.utils.camera_utils import Camera
from lib.utils.sh_utils import eval_sh
from lib.models.actor_pose import ActorPose
from lib.models.sky_cubemap import SkyCubeMap
from lib.models.color_correction import ColorCorrection
from lib.models.camera_pose import PoseCorrection


def load_ply(path, max_sh_degree):

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    # if 'Lamborghini' in path:
    #     import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    assert len(extra_f_names) == 3*(max_sh_degree + 1) ** 2 - 3

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float,
                        device="cuda").requires_grad_(True))
    _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(
        1, 2).contiguous().requires_grad_(True))
    _features_rest = nn.Parameter(torch.tensor(
        features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    _opacity = nn.Parameter(torch.tensor(
        opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    _scaling = nn.Parameter(torch.tensor(
        scales, dtype=torch.float, device="cuda").requires_grad_(True))
    _rotation = nn.Parameter(torch.tensor(
        rots, dtype=torch.float, device="cuda").requires_grad_(True))

    return _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation


def load_ply_temp_test_new_car(path, max_sh_degree):

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3

    features_extra = np.zeros((xyz.shape[0], 3*(max_sh_degree + 1) ** 2 - 3))
    for idx in range(3*(max_sh_degree + 1) ** 2 - 3):
        features_extra[:, idx] = np.zeros(xyz.shape[0], dtype=np.float32)
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float,
                        device="cuda").requires_grad_(True))
    _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(
        1, 2).contiguous().requires_grad_(True))
    _features_rest = nn.Parameter(torch.tensor(
        features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    _opacity = nn.Parameter(torch.tensor(
        opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    _scaling = nn.Parameter(torch.tensor(
        scales, dtype=torch.float, device="cuda").requires_grad_(True))
    _rotation = nn.Parameter(torch.tensor(
        rots, dtype=torch.float, device="cuda").requires_grad_(True))

    return _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation


def detect_and_replace_outliers(_xyz, threshold=4):
    if _xyz.is_cuda:
        _xyz = _xyz.cpu()

    processed_xyz = _xyz.clone()

    mean = _xyz.mean(dim=0)
    std = _xyz.std(dim=0)

    z_scores = (_xyz - mean) / std

    outliers = torch.abs(z_scores) > threshold
    outlier_values = _xyz[outliers]

    num_outliers = torch.sum(outliers)
    for dim in range(_xyz.shape[1]):
        processed_xyz[outliers[:, dim], dim] = mean[dim]

    return processed_xyz.cuda()


def create_rotation_matrix():
    rotation_matrix = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=torch.float32).cuda()
    return rotation_matrix


def create_rotation_matrix_x_90():
    """创建绕X轴旋转90度的旋转矩阵（向y轴正方向）"""
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],   # cos(90°)=0, sin(90°)=1
        [0, -1, 0]   # -sin(90°)=-1, cos(90°)=0
    ], dtype=torch.float32).cuda()
    return rotation_matrix


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(1)
    w2, x2, y2, z2 = q2.unbind(1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


# 转为四元数
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


class StreetGaussianModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

        self.max_sh_degree = cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        # background + moving objects
        self.include_background = cfg.model.nsg.get('include_bkgd', True)
        self.include_obj = cfg.model.nsg.get('include_obj', True)

        # sky (modeling sky with gaussians, if set to false represent the sky with cube map)
        self.include_sky = cfg.model.nsg.get('include_sky', False)
        if self.include_sky:
            assert cfg.data.white_background is False

        # fourier sh dimensions
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)

        # layer color correction
        self.use_color_correction = cfg.model.use_color_correction

        # camera pose optimizations (not test)
        self.use_pose_correction = cfg.model.use_pose_correction

        self.model_paths = {}

        for root, dirs, files in os.walk("/mnt/xuhr/street-gs/data/relight_car"):
            for file in files:
                key = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                self.model_paths[key] = full_path

        # 获取所有的 key 列表
        self.model_keys = list(self.model_paths.keys())

        self.model_size = {}

        self.init_render_setup(metadata)

        self.setup_functions()

    def set_visibility(self, include_list):
        self.include_list = include_list  # prefix

    def get_visibility(self, model_name):
        if model_name == 'background':
            if model_name in self.include_list and self.include_background:
                return True
            else:
                return False
        elif model_name == 'sky':
            if model_name in self.include_list and self.include_sky:
                return True
            else:
                return False
        elif model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.create_from_pcd(pcd, spatial_lr_scale)
            else:
                model.create_from_pcd(spatial_lr_scale)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModel = getattr(self, model_name)
            plydata = model.make_ply()
            plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            plydata_list.append(plydata)

        PlyData(plydata_list).write(path)

    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:]  # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: GaussianModel = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements

        self.active_sh_degree = self.max_sh_degree

    def load_state_dict(self, state_dict, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])

        if self.actor_pose is not None:
            self.actor_pose.load_state_dict(state_dict['actor_pose'])

        if self.sky_cubemap is not None:
            self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])

        if self.color_correction is not None:
            self.color_correction.load_state_dict(
                state_dict['color_correction'])

        if self.pose_correction is not None:
            self.pose_correction.load_state_dict(state_dict['pose_correction'])

    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)

        if self.actor_pose is not None:
            state_dict['actor_pose'] = self.actor_pose.save_state_dict(
                is_final)

        if self.sky_cubemap is not None:
            state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(
                is_final)

        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(
                is_final)

        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(
                is_final)

        return state_dict

    def setup_functions(self):
        obj_tracklets = self.metadata['obj_tracklets']
        obj_info = self.metadata['obj_meta']
        all_obj_tracklets = self.metadata['all_obj_tracklets']
        all_obj_info = self.metadata['all_obj_info']
        tracklet_timestamps = self.metadata['tracklet_timestamps']
        camera_timestamps = self.metadata['camera_timestamps']

        self.model_name_id = bidict()
        self.obj_list = []
        # self.all_obj_list = []
        self.models_num = 0
        self.obj_info = obj_info

        # Build background model
        if self.include_background:
            self.background = GaussianModelBkgd(
                model_name='background',
                scene_center=self.metadata['scene_center'],
                scene_radius=self.metadata['scene_radius'],
                sphere_center=self.metadata['sphere_center'],
                sphere_radius=self.metadata['sphere_radius'],
            )

            self.model_name_id['background'] = 0
            self.models_num += 1

        # Build object model
        if self.include_obj:
            for track_id, obj_meta in self.obj_info.items():
                model_name = f'obj_{track_id:03d}'
                setattr(self, model_name, GaussianModelActor(
                    model_name=model_name, obj_meta=obj_meta))
                self.model_name_id[model_name] = self.models_num
                self.obj_list.append(model_name)
                self.models_num += 1

        # Build sky model
        if self.include_sky:
            self.sky_cubemap = SkyCubeMap()
        else:
            self.sky_cubemap = None

        # Build actor model
        if self.include_obj:
            self.actor_pose = ActorPose(
                obj_tracklets, tracklet_timestamps, camera_timestamps, obj_info)
            self.all_actor_pose = ActorPose(
                all_obj_tracklets, tracklet_timestamps, camera_timestamps, all_obj_info)
        else:
            self.actor_pose = None
            self.all_actor_pose = None

        # Build color correction
        if self.use_color_correction:
            self.color_correction = ColorCorrection(self.metadata)
        else:
            self.color_correction = None

        # Build pose correction
        if self.use_pose_correction:
            self.pose_correction = PoseCorrection(self.metadata)
        else:
            self.pose_correction = None

    def parse_camera(self, camera: Camera):
        # set camera
        self.viewpoint_camera = camera

        # set background mask
        self.background.set_background_mask(camera)

        self.frame = camera.meta['frame']
        self.frame_idx = camera.meta['frame_idx']
        self.frame_is_val = camera.meta['is_val']
        self.num_gaussians = 0

        # background
        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.num_gaussians += num_gaussians_bkgd

        # object (build scene graph)
        self.graph_obj_list = []
        self.graph_all_obj_list = []
        # import pdb; pdb.set_trace()
        if self.include_obj:
            timestamp = camera.meta['timestamp']
            for i, obj_name in enumerate(self.obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                start_timestamp, end_timestamp = obj_model.start_timestamp, obj_model.end_timestamp
                if timestamp >= start_timestamp and timestamp <= end_timestamp and self.get_visibility(obj_name):
                    self.graph_obj_list.append(obj_name)
                    num_gaussians_obj = getattr(
                        self, obj_name).get_xyz.shape[0]
                    self.num_gaussians += num_gaussians_obj

            timestamp = camera.meta['timestamp']
            for i, obj_name in enumerate(self.all_obj_list):
                obj_model: GaussianModelActor = self.all_actor_manager.get_actor(
                    obj_name)
                start_timestamp, end_timestamp = obj_model.start_timestamp, obj_model.end_timestamp
                # 这里原本需要判断是否是 visable但是那个基于的是原本的 include list
                if timestamp >= start_timestamp and timestamp <= end_timestamp:
                    self.graph_all_obj_list.append(obj_name)
                    # num_gaussians_obj = getattr(self, obj_name).get_xyz.shape[0]
                    # self.num_gaussians += num_gaussians_obj

        # set index range
        self.graph_gaussian_range = dict()
        idx = 0

        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.graph_gaussian_range['background'] = [
                idx, idx+num_gaussians_bkgd-1]
            idx += num_gaussians_bkgd

        for obj_name in self.graph_obj_list:
            num_gaussians_obj = getattr(self, obj_name).get_xyz.shape[0]
            self.graph_gaussian_range[obj_name] = [
                idx, idx+num_gaussians_obj-1]
            idx += num_gaussians_obj

    @property
    def get_scaling(self):
        scalings = []

        if self.get_visibility('background'):
            scaling_bkgd = self.background.get_scaling
            scalings.append(scaling_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            scaling = obj_model.get_scaling

            scalings.append(scaling)

        scalings = torch.cat(scalings, dim=0)
        return scalings

    @property
    def get_rotation(self):
        rotations = []

        if self.get_visibility('background'):
            rotations_bkgd = self.background.get_rotation

            if self.use_pose_correction:
                rotations_bkgd = self.pose_correction.correct_gaussian_rotation(
                    self.viewpoint_camera, rotations_bkgd)

            rotations.append(rotations_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id
            rotations_local = obj_model.get_rotation
            rotations_local = obj_model.flip_rotation(rotations_local)

            obj_rot = self.actor_pose.get_tracking_rotation(
                track_id, self.viewpoint_camera)
            if cfg.render.coord == 'world':
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(
                    ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(
                    ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)

            rotations_global = quaternion_raw_multiply(
                obj_rot[None], rotations_local)  # [N, 4]
            rotations_global = torch.nn.functional.normalize(rotations_global)
            rotations.append(rotations_global)

        rotations = torch.cat(rotations, dim=0)
        return rotations

    @property
    def get_xyz(self):
        xyzs = []
        if self.get_visibility('background'):
            xyzs_bkgd = self.background.get_xyz

            if self.use_pose_correction:
                xyzs_bkgd = self.pose_correction.correct_gaussian_xyz(
                    self.viewpoint_camera, xyzs_bkgd)

            xyzs.append(xyzs_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id

            xyzs_local = obj_model.get_xyz
            xyzs_local = obj_model.flip_xyz(xyzs_local)

            obj_rot = self.actor_pose.get_tracking_rotation(
                track_id, self.viewpoint_camera)
            obj_trans = self.actor_pose.get_tracking_translation(
                track_id, self.viewpoint_camera)
            if cfg.render.coord == 'world':
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(
                    ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(
                    ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]

            obj_rot = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
            xyzs_global = xyzs_local @ obj_rot.transpose(0, 1) + obj_trans

            xyzs.append(xyzs_global)

        xyzs = torch.cat(xyzs, dim=0)

        return xyzs

    @property
    def get_features(self):
        features = []

        if self.get_visibility('background'):
            features_bkgd = self.background.get_features
            features.append(features_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            # feature_obj = obj_model.get_features_fourier(self.frame)
            feature_obj = obj_model.get_features_fourier(0)
            features.append(feature_obj)

        features = torch.cat(features, dim=0)

        return features

    def get_colors(self, camera_center):
        colors = []

        model_names = []
        if self.get_visibility('background'):
            model_names.append('background')

        model_names.extend(self.graph_obj_list)

        for model_name in model_names:
            if model_name == 'background':
                model: GaussianModel = getattr(self, model_name)
            else:
                model: GaussianModelActor = getattr(self, model_name)

            max_sh_degree = model.max_sh_degree
            sh_dim = (max_sh_degree + 1) ** 2

            if model_name == 'background':
                shs = model.get_features.transpose(1, 2).view(-1, 3, sh_dim)
            elif hasattr(model, 'is_static_replaced') and model.is_static_replaced:

                # 全部用 frame 0 的特征
                features = model.get_features_fourier(0)
                shs = features.transpose(1, 2).view(-1, 3, sh_dim)
            else:

                features = model.get_features_fourier(self.frame)
                shs = features.transpose(1, 2).view(-1, 3, sh_dim)

            directions = model.get_xyz - camera_center
            directions = directions / \
                torch.norm(directions, dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs, directions)
            color = torch.clamp_min(sh2rgb + 0.5, 0.)
            colors.append(color)

        colors = torch.cat(colors, dim=0)
        return colors

    @property
    def get_semantic(self):
        semantics = []
        if self.get_visibility('background'):
            semantic_bkgd = self.background.get_semantic
            semantics.append(semantic_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            semantic = obj_model.get_semantic

            semantics.append(semantic)

        semantics = torch.cat(semantics, dim=0)
        return semantics

    @property
    def get_opacity(self):
        opacities = []

        if self.get_visibility('background'):
            opacity_bkgd = self.background.get_opacity
            opacities.append(opacity_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            opacity = obj_model.get_opacity

            opacities.append(opacity)

        opacities = torch.cat(opacities, dim=0)
        return opacities

    def get_covariance(self, scaling_modifier=1):
        scaling = self.get_scoaling  # [N, 1]
        rotation = self.get_rotation  # [N, 4]
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def get_normals(self, camera: Camera):
        normals = []

        if self.get_visibility('background'):
            normals_bkgd = self.background.get_normals(camera)
            normals.append(normals_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id

            normals_obj_local = obj_model.get_normals(camera)  # [N, 3]

            obj_rot = self.actor_pose.get_tracking_rotation(
                track_id, self.viewpoint_camera)
            obj_rot = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)

            normals_obj_global = normals_obj_local @ obj_rot.T
            normals_obj_global = torch.nn.functinal.normalize(
                normals_obj_global)
            normals.append(normals_obj_global)

        normals = torch.cat(normals, dim=0)
        return normals

    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            model: GaussianModel = getattr(self, model_name)
            model.oneupSHdegree()

        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.training_setup()

        if self.actor_pose is not None:
            self.actor_pose.training_setup()

        if self.sky_cubemap is not None:
            self.sky_cubemap.training_setup()

        if self.color_correction is not None:
            self.color_correction.training_setup()

        if self.pose_correction is not None:
            self.pose_correction.training_setup()

    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_learning_rate(iteration)

        if self.actor_pose is not None:
            self.actor_pose.update_learning_rate(iteration)

        if self.sky_cubemap is not None:
            self.sky_cubemap.update_learning_rate(iteration)

        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)

        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)

    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_optimizer()

        if self.actor_pose is not None:
            self.actor_pose.update_optimizer()

        if self.sky_cubemap is not None:
            self.sky_cubemap.update_optimizer()

        if self.color_correction is not None:
            self.color_correction.update_optimizer()

        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def set_max_radii2D(self, radii, visibility_filter):
        radii = radii.float()

        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            end += 1
            visibility_model = visibility_filter[start:end]
            max_radii2D_model = radii[start:end]
            model.max_radii2D[visibility_model] = torch.max(
                model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])

    def add_densification_stats(self, viewspace_point_tensor, visibility_filter):
        viewspace_point_tensor_grad = viewspace_point_tensor.grad

        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            end += 1
            visibility_model = visibility_filter[start:end]
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end]
            model.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(
                viewspace_point_tensor_grad_model[visibility_model, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(
                viewspace_point_tensor_grad_model[visibility_model, 2:], dim=-1, keepdim=True)
            model.denom[visibility_model] += 1

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points, exclude_list=[]):
        scalars = None
        tensors = None
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)

            if model.get_xyz.shape[0] < 10:
                continue

            scalars_, tensors_ = model.densify_and_prune(
                max_grad, min_opacity, prune_big_points)
            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_

        return scalars, tensors

    def get_box_reg_loss(self):
        box_reg_loss = 0.

        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss

    def set_flip(self, flip=None):
        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            obj_model.set_flip(flip)

    def reset_opacity(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()

    def swap_objects_position(self, obj1_name, obj2_name):
        obj1_model: GaussianModelActor = getattr(self, obj1_name)
        obj2_model: GaussianModelActor = getattr(self, obj2_name)

        obj1_xyz = obj1_model.get_xyz
        obj2_xyz = obj2_model.get_xyz
        obj1_model._xyz = obj2_xyz
        obj2_model._xyz = obj1_xyz

        obj1_rotation = obj1_model.get_original_rotation
        obj2_rotation = obj2_model.get_original_rotation
        obj1_model._rotation = obj2_rotation
        obj2_model._rotation = obj1_rotation

        obj1_scaling = obj1_model.get_original_scaling
        obj2_scaling = obj2_model.get_original_scaling
        obj1_model._scaling = obj2_scaling
        obj2_model._scaling = obj1_scaling

    @property
    def get_xyz_edit(self):
        xyzs = []

        swap_pairs = [
            # ('obj_2762', 'obj_2763'),
            # ('obj_2805', 'obj_2765'),
            # ('obj_2798', 'obj_2788'),
            # ('obj_2836', 'obj_2839'),
            # ('obj_2764', 'obj_2851'),
            # ('obj_2823', 'obj_2777')

        ]

        swap_dict = {obj1: obj2 for obj1, obj2 in swap_pairs}
        swap_dict.update({obj2: obj1 for obj1, obj2 in swap_pairs})

        if self.get_visibility('background'):
            xyzs_bkgd = self.background.get_xyz
            if self.use_pose_correction:
                xyzs_bkgd = self.pose_correction.correct_gaussian_xyz(
                    self.viewpoint_camera, xyzs_bkgd)
            xyzs.append(xyzs_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id

            xyzs_local = obj_model.get_xyz
            xyzs_local = obj_model.flip_xyz(xyzs_local)

            if obj_name in swap_dict:
                swap_with_name = swap_dict[obj_name]
                swap_with_model: GaussianModelActor = getattr(
                    self, swap_with_name)
                obj_rot = self.actor_pose.get_tracking_rotation(
                    swap_with_model.track_id, self.viewpoint_camera)
                obj_trans = self.actor_pose.get_tracking_translation(
                    swap_with_model.track_id, self.viewpoint_camera)
            else:
                obj_rot = self.actor_pose.get_tracking_rotation(
                    track_id, self.viewpoint_camera)
                obj_trans = self.actor_pose.get_tracking_translation(
                    track_id, self.viewpoint_camera)

            if cfg.render.coord == 'world':
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(
                    ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(
                    ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]

            obj_rot = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
            xyzs_global = xyzs_local @ obj_rot.transpose(0, 1) + obj_trans

            xyzs.append(xyzs_global)

        xyzs = torch.cat(xyzs, dim=0)
        return xyzs

    def load_ply(path, max_sh_degree):

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3*(max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        _xyz = nn.Parameter(torch.tensor(
            xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(
            1, 2).contiguous().requires_grad_(True))
        _features_rest = nn.Parameter(torch.tensor(
            features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        _opacity = nn.Parameter(torch.tensor(
            opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = nn.Parameter(torch.tensor(
            scales, dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = nn.Parameter(torch.tensor(
            rots, dtype=torch.float, device="cuda").requires_grad_(True))

        return _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation

    def replace_gaussian_with_custom_actor(self, model_name, target_actor_name):

        self.model_size = {'wrangler-unlimited-smoky-mountain-jk-2017': 4.5,
                           'peugeot-boxer-window-van-l1h1-2006-2014': 5,
                           'mercedes-benz-s-560-lang-amg-line-v222-2018.fbx': 4.5,
                           'another_normal_car_relight': 4.5,
                           'renault-master-l4h2-van-2010.fbx': 4.5,
                           'iveco-daily-l1h1-2017': 4.5,
                           'white_big_car': 4.5,
                           'van_relight': 4.5,
                           'jeep_relight': 4.5,
                           'lamborghini-aventador-s-roadster-2018': 4.5,
                           'opel-combo-cargo-ru-spec-l1-2021': 4.5,
                           'opel-combo-cargo-ru-spec-l2-2021': 4.5,
                           'jeep_relight_1': 4.5,
                           'volkswagen-golf-7-tdi-5d-2016': 4.5,
                           'iveco-daily-tourus-2017': 4.5,
                           'nissan-nv-300-van-lwb-2021': 4.5,
                           'opel-combo-tour-lwb-d-2015': 4.5,
                           'pickup_relight': 4.5,
                           'vw-beetle-turbo-2017': 4.5,


                           }

        if model_name not in self.model_paths:
            raise ValueError(
                f"Model name '{model_name}' not found in model paths dictionary.")
        custom_model_path = self.model_paths[model_name]

        # Step 1: Load custom Gaussian model point cloud
        max_sh_degree = cfg.model.gaussian.sh_degree
        _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation = load_ply(
            custom_model_path, max_sh_degree)
        _xyz = detect_and_replace_outliers(_xyz)

        # Step 2: Get existing GaussianModelActor
        actor_model: GaussianModelActor = getattr(self, target_actor_name)

        actor_model.edited = 'replaced'

        print(
            f"Replaced {target_actor_name} with custom Gaussian model from {custom_model_path}")

        y_min = torch.min(_xyz[:, 1])
        y_max = torch.max(_xyz[:, 1])

        # print(y_min)
        # print(y_max)

        target_car_length = self.model_size[model_name]

        source_car_length = y_max - y_min

        scale = source_car_length / target_car_length

        # print('---------------------')

        # print(source_car_length)
        # print(target_car_length)
        # print(scale)

        _xyz = _xyz / scale
        _scaling = _scaling - torch.log(scale)
        x_min = torch.min(_xyz[:, 0])
        x_max = torch.max(_xyz[:, 0])
        y_min = torch.min(_xyz[:, 1])
        y_max = torch.max(_xyz[:, 1])
        z_min = torch.min(_xyz[:, 2])

        center_lowest_point = torch.tensor(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=torch.float32).cuda()

        rotation_matrix_to_x_axis = create_rotation_matrix()
        _xyz = _xyz.float() @ rotation_matrix_to_x_axis.T
        _rotation = quaternion_multiply(torch.tensor(rotmat2qvec(rotation_matrix_to_x_axis.cpu(
        ).numpy()), dtype=torch.float32).unsqueeze(0).cuda(), _rotation.float())

        target_x_min, target_x_max = torch.min(
            actor_model._xyz[:, 0]), torch.max(actor_model._xyz[:, 0])
        target_y_min, target_y_max = torch.min(
            actor_model._xyz[:, 1]), torch.max(actor_model._xyz[:, 1])
        target_z_min = torch.min(actor_model._xyz[:, 2])

        target_x = (target_x_min + target_x_max) / 2
        target_y = (target_y_min + target_y_max) / 2
        target_z = target_z_min

        target_z += 0.2

        target_position = torch.tensor(
            [target_x, target_y, target_z], dtype=torch.float32).cuda()

        rotated_center_lowest_point = center_lowest_point @ rotation_matrix_to_x_axis.T

        total_translation = target_position - rotated_center_lowest_point

        _xyz += total_translation

        actor_model._xyz = nn.Parameter(_xyz, requires_grad=False)
        actor_model._features_dc = nn.Parameter(
            _features_dc, requires_grad=False)
        actor_model._features_rest = nn.Parameter(
            _features_rest, requires_grad=False)
        actor_model._opacity = nn.Parameter(_opacity, requires_grad=False)
        actor_model._scaling = nn.Parameter(_scaling, requires_grad=False)
        actor_model._rotation = nn.Parameter(_rotation, requires_grad=False)

        actor_model.is_static_replaced = True

        print(
            f"Replaced {target_actor_name} with custom Gaussian model from {custom_model_path}")

    def replace_gaussian_with_custom_actor_new(self, model_name, target_actor_name):

        self.model_size = {'wrangler-unlimited-smoky-mountain-jk-2017': 4.5,
                           'peugeot-boxer-window-van-l1h1-2006-2014': 5,
                           'mercedes-benz-s-560-lang-amg-line-v222-2018.fbx': 4.5,
                           'another_normal_car_relight': 4.5,
                           'renault-master-l4h2-van-2010.fbx': 4.5,
                           'iveco-daily-l1h1-2017': 4.5,
                           'white_big_car': 4.5,
                           'van_relight': 4.5,
                           'jeep_relight': 4.5,
                           'lamborghini-aventador-s-roadster-2018': 4.5,
                           'opel-combo-cargo-ru-spec-l1-2021': 4.5,
                           'opel-combo-cargo-ru-spec-l2-2021': 4.5,
                           'jeep_relight_1': 4.5,
                           'volkswagen-golf-7-tdi-5d-2016': 4.5,
                           'iveco-daily-tourus-2017': 4.5,
                           'nissan-nv-300-van-lwb-2021': 4.5,
                           'opel-combo-tour-lwb-d-2015': 4.5,
                           'pickup_relight': 4.5,
                           'vw-beetle-turbo-2017': 4.5,
                           'Lamborghini': 4.5
                           }

        print(
            f"Replacing {target_actor_name} with custom Gaussian model from {model_name}")

        if model_name not in self.model_paths:
            raise ValueError(
                f"Model name '{model_name}' not found in model paths dictionary.")
        custom_model_path = self.model_paths[model_name]

        # Step 1: Load custom Gaussian model point cloud
        max_sh_degree = cfg.model.gaussian.sh_degree

        try:
            _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation = load_ply(
                custom_model_path, max_sh_degree)
        except Exception as e:
            _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation = load_ply_temp_test_new_car(
                custom_model_path, max_sh_degree)
            # 添加绕X轴旋转90度
            rotation_matrix_x = create_rotation_matrix_x_90()
            _xyz = _xyz.float() @ rotation_matrix_x.T
            _rotation = quaternion_multiply(torch.tensor(rotmat2qvec(rotation_matrix_x.cpu(
            ).numpy()), dtype=torch.float32).unsqueeze(0).cuda(), _rotation.float())

        _xyz = detect_and_replace_outliers(_xyz)

        # Step 2: Get existing GaussianModelActor
        actor_model: GaussianModelActor = getattr(self, target_actor_name)

        actor_model.edited = 'replaced'

        y_min = torch.min(_xyz[:, 1])
        y_max = torch.max(_xyz[:, 1])

        target_car_length = self.model_size.get(model_name, 4.5)

        source_car_length = y_max - y_min

        scale = source_car_length / target_car_length

        _xyz = _xyz / scale
        _scaling = _scaling - torch.log(scale)
        x_min = torch.min(_xyz[:, 0])
        x_max = torch.max(_xyz[:, 0])
        y_min = torch.min(_xyz[:, 1])
        y_max = torch.max(_xyz[:, 1])
        z_min = torch.min(_xyz[:, 2])

        # center_lowest_point = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=torch.float32).cuda()

        rotation_matrix_to_x_axis = create_rotation_matrix()
        _xyz = _xyz.float() @ rotation_matrix_to_x_axis.T
        _rotation = quaternion_multiply(torch.tensor(rotmat2qvec(rotation_matrix_to_x_axis.cpu(
        ).numpy()), dtype=torch.float32).unsqueeze(0).cuda(), _rotation.float())

        total_translation = torch.tensor([0.0,  0.0, 0.3], device='cuda:0')

        _xyz += total_translation

        actor_model._xyz = nn.Parameter(_xyz, requires_grad=False)
        actor_model._features_dc = nn.Parameter(
            _features_dc, requires_grad=False)
        actor_model._features_rest = nn.Parameter(
            _features_rest, requires_grad=False)
        actor_model._opacity = nn.Parameter(_opacity, requires_grad=False)
        actor_model._scaling = nn.Parameter(_scaling, requires_grad=False)
        actor_model._rotation = nn.Parameter(_rotation, requires_grad=False)

        x_min, x_max = torch.min(_xyz[:, 0]), torch.max(_xyz[:, 0])
        y_min, y_max = torch.min(_xyz[:, 1]), torch.max(_xyz[:, 1])
        z_min, z_max = torch.min(_xyz[:, 2]), torch.max(_xyz[:, 2])

        actor_model.bbox = torch.tensor([
            x_max - x_min,  # length
            y_max - y_min,  # width
            z_max - z_min   # height
        ], device=_xyz.device)

        actor_model.bbox_center = torch.tensor([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        ], device=_xyz.device)

        track_id = actor_model.track_id
        self.actor_pose.update_center_offset(track_id, actor_model.bbox_center)
        # import pdb; pdb.set_trace()

        corners = torch.tensor([
            [-actor_model.bbox[0]/2, -actor_model.bbox[1]/2, -actor_model.bbox[2]/2],
            [actor_model.bbox[0]/2, -actor_model.bbox[1]/2, -actor_model.bbox[2]/2],
            [-actor_model.bbox[0]/2, actor_model.bbox[1]/2, -actor_model.bbox[2]/2],
            [actor_model.bbox[0]/2, actor_model.bbox[1]/2, -actor_model.bbox[2]/2],
            [-actor_model.bbox[0]/2, -actor_model.bbox[1]/2, actor_model.bbox[2]/2],
            [actor_model.bbox[0]/2, -actor_model.bbox[1]/2, actor_model.bbox[2]/2],
            [-actor_model.bbox[0]/2, actor_model.bbox[1]/2, actor_model.bbox[2]/2],
            [actor_model.bbox[0]/2, actor_model.bbox[1]/2, actor_model.bbox[2]/2],
        ], device=_xyz.device)

        actor_model.bbox_corners_local = corners

        # print(f"Replaced {target_actor_name} with custom Gaussian model from {custom_model_path}")
        print(f"New bbox dimensions: {actor_model.bbox.tolist()}")

    def init_render_setup(self, metadata):
        """初始化渲染相关的内容，这个函数可以在需要时调用来重新初始化渲染所需的组件"""

        # 初始化新的actor管理器
        self.all_actor_manager = ActorManager()

        # 获取所需的metadata
        all_obj_tracklets = metadata['all_obj_tracklets']
        all_obj_info = metadata['all_obj_info']
        tracklet_timestamps = metadata['tracklet_timestamps']
        camera_timestamps = metadata['camera_timestamps']

        self.all_obj_info = all_obj_info
        self.all_obj_tracklets = all_obj_tracklets

        # 重新初始化all_obj_list
        self.all_obj_list = []

        # 使用ActorManager管理新的actor
        for track_id, obj_meta in all_obj_info.items():
            model_name = f'obj_{track_id:03d}'
            actor = GaussianModelActor(
                model_name=model_name, obj_meta=obj_meta)
            self.all_actor_manager.add_actor(model_name, actor)
            self.all_obj_list.append(model_name)

        # 初始化all_actor_pose
        self.all_actor_pose = ActorPose(
            all_obj_tracklets,
            tracklet_timestamps,
            camera_timestamps,
            all_obj_info
        )

        print(
            f"Render setup initialized with {len(self.all_obj_list)} additional actors")

    def remove_actor(self, model_name):
        """从StreetGaussianModel中移除指定的actor"""
        if hasattr(self, model_name):
            delattr(self, model_name)  # 删除actor对象

            # 同时清理相关的记录
            if model_name in self.model_name_id:
                self.model_name_id.pop(model_name)
            if model_name in self.obj_list:
                self.obj_list.remove(model_name)

            print(f"Successfully removed actor: {model_name}")
        else:
            print(f"Actor {model_name} not found")

    def export_modified_actors_info(self):
        """导出所有被修改过的actor的关键信息"""
        modified_actors = []

        for obj_name in self.all_obj_list:
            try:
                obj_model = getattr(self, obj_name)
                actor_pose = self.actor_pose
            except:

                obj_model = self.all_actor_manager.get_actor(obj_name)
                actor_pose = self.all_actor_pose

            if hasattr(obj_model, 'edited') and obj_model.edited == 'replaced':
                track_id = obj_model.track_id

                actor_info = {
                    'track_id': track_id,
                    'model_name': obj_name,

                    'bbox': {
                        'length': obj_model.bbox[0].item(),
                        'width': obj_model.bbox[1].item(),
                        'height': obj_model.bbox[2].item()
                    },

                    'center_offset': actor_pose.center_offsets[track_id].tolist() if track_id in actor_pose.center_offsets else None
                }

                modified_actors.append(actor_info)

        return modified_actors

    def save_modified_actors_info(self, save_path):
        """将修改过的actor信息保存到文件"""
        import json

        modified_info = self.export_modified_actors_info()

        # 保存为JSON格式
        with open(save_path, 'w') as f:
            json.dump(modified_info, f, indent=4)
