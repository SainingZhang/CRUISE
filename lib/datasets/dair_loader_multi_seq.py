from lib.utils.dair_utils_multi_seq import generate_dataparser_outputs
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, get_PCA_Norm, get_Sphere_Norm
from lib.config import cfg
from lib.datasets.base_readers import storePly, get_Sphere_Norm
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import cv2
import sys
import copy
import shutil

sys.path.append(os.getcwd())

def readSingleDairSequence(path, images='images', split_train=-1, split_test=-1, specified_sequence_id=None, **kwargs):
    """读取单个序列的数据
    Args:
        path: 序列路径
        images: 图像文件夹名称
        split_train: 训练集划分
        split_test: 测试集划分
    Returns:
        scene_info: 场景信息
    """
    selected_frames_dict = cfg.data.get('selected_frames', None)
    selected_frames = selected_frames_dict[specified_sequence_id]
    print(f'selected_frames: {selected_frames}')
    if cfg.debug:
        selected_frames = [0, 0]

    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)
        
        # colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        # save_dir = os.path.join(cfg.model_path, 'colmap')
        # os.system(f'rm -rf {save_dir}')
        # shutil.copytree(colmap_dir, save_dir)
        
    bkgd_ply_path = os.path.join(cfg.model_path, f'input_ply/points3D_bkgd_{specified_sequence_id}.ply')
    build_pointcloud = (cfg.mode == 'train') and (not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False))
    
    # TODO 有很多东西都没法用
    # dynamic mask 
    # dynamic_mask_dir = os.path.join(path, 'dynamic_mask') 
    load_dynamic_mask = True

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    if not os.path.exists(sky_mask_dir):
        cmd = f'python script/waymo/generate_sky_mask.py --datadir {path}'
        print('Generating sky mask')
        os.system(cmd)
    load_sky_mask = (cfg.mode == 'train')
    
    # lidar depth
    
    # load_lidar_depth = False
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    if not os.path.exists(lidar_depth_dir):
        cmd = f'python script/waymo/generate_lidar_depth.py --datadir {path}'
        print('Generating lidar depth')
        os.system(cmd)
    load_lidar_depth = (cfg.mode == 'train')
    
    
    # Optional: monocular normal cue
    # load_mono_normal = False
    mono_normal_dir = os.path.join(path, 'mono_normal')
    load_mono_normal = cfg.data.use_mono_normal and (cfg.mode == 'train') and os.path.exists(mono_normal_dir)
        
    # Optional: monocular depth cue
    # load_mono_depth = False
    mono_depth_dir = os.path.join(path, 'mono_depth')
    load_mono_depth = cfg.data.use_mono_depth and (cfg.mode == 'train') and os.path.exists(mono_depth_dir)
    
    load_ego_mask = True
    ego_mask_dir = os.path.join(path, 'ego_mask')
    
    load_car_mask = True
    car_mask_dir = os.path.join(path, 'car_mask')

    output = generate_dataparser_outputs(
        datadir=path, 
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cfg.data.get('cameras', [0, 1]),
        specified_sequence_id=specified_sequence_id
    )

    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(ixts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames
    
    camera_timestamps = dict()
    for cam in cfg.data.get('cameras', [0, 1]):
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []      

    ########################################################################################################################
    cam_infos = []
    for i in tqdm(range(len(ixts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height) # TODO 有可能是这里的fov
        FovX = focal2fov(fy, width)    
        
        # if cfg.render.coord == 'world':
        #     RT = np.linalg.inv(c2w)        # render in world space
        # else:
        #     RT = np.linalg.inv(ext)        # render in vehicle space
        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()
        
        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose 
        metadata['extrinsic'] = ext 
        metadata['timestamp'] = cams_timestamps[i]

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])
        
        # load dynamic mask
        if load_dynamic_mask:
            # dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
            # obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.
            # obj_bound = Image.fromarray(obj_bound)
            metadata['obj_bound'] = Image.fromarray(obj_bounds[i])
                    
        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(path, 'lidar_depth', f'{image_name}.npy')
            
            depth = np.load(depth_path, allow_pickle=True)
            if isinstance(depth, np.ndarray):
                depth = dict(depth.item())
                mask = depth['mask']
                value = depth['value']
                depth = np.zeros_like(mask).astype(np.float32)
                depth[mask] = value
            
            metadata['lidar_depth'] = depth
            
        # load sky mask
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.jpg')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            sky_mask = Image.fromarray(sky_mask)
            metadata['sky_mask'] = sky_mask
            
        # load ego mask
        if load_ego_mask:
            if cams[i] == 0:
                ego_mask_path = os.path.join(ego_mask_dir, '000000_0.jpg')
            elif cams[i] == 1:
                ego_mask_path = os.path.join(ego_mask_dir, 'no_mask.jpg')
            ego_mask = (cv2.imread(ego_mask_path)[..., 0]) > 0.
            ego_mask = Image.fromarray(ego_mask)
            metadata['ego_mask'] = ego_mask
        
        # load car mask
        if load_car_mask:
            car_mask_path = os.path.join(car_mask_dir, f'{image_name}.jpg')
            car_mask = (cv2.imread(car_mask_path)[..., 0]) > 0.
            car_mask = Image.fromarray(car_mask)
            metadata['car_mask'] = car_mask
        
        # Optional: load monocular normal
        if load_mono_normal:
            mono_normal_path = os.path.join(mono_normal_dir, f'{image_name}.npy')
            mono_normal = np.load(mono_normal_path)
            metadata['mono_normal'] = mono_normal

        # Optional load midas depth
        if load_mono_depth:
            mono_depth_path = os.path.join(mono_depth_dir, f'{image_name}.npy')
            mono_depth = np.load(mono_depth_path)
            metadata['mono_depth'] = mono_depth

        mask = None        
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height, 
            mask=mask,
            metadata=metadata)
        cam_infos.append(cam_info) # TODO 之后对比一下camera的信息应该是只有fov不太一样其他的应该还好
        
        # sys.stdout.write('\n')
    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]
    
    for cam in cfg.data.get('cameras', [0, 1]):
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps
        
    novel_view_cam_infos = []
    
    #######################################################################################################################
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    
    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    
    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata    
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    # Get sphere center
    lidar_ply_path = os.path.join(cfg.model_path, f'input_ply/points3D_lidar_{specified_sequence_id}.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd: BasicPointCloud = fetchPly(lidar_ply_path)
    else:
        sphere_pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    
    sphere_normalization = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']
    print(f'Sphere extent: {sphere_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    if cfg.mode == 'train':
        point_cloud = pcd
    else:
        point_cloud = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )
    
    return scene_info

def readDairFullInfo_multi_seq(paths, images='images', split_train=-1, split_test=-1, **kwargs):
    """读取多个序列的数据
    Args:
        paths: 序列路径列表
        images: 图像文件夹名称
        split_train: 训练集划分
        split_test: 测试集划分
    Returns:
        merged_scene_info: 合并后的场景信息
    """
    all_train_cameras = []
    all_test_cameras = []
    all_novel_view_cameras = []
    all_point_clouds = []
    merged_metadata = {
        'num_sequences': len(paths),
        'sequence_lengths': [],
        'camera_timestamps': {},
        'obj_tracklets': [],
        'tracklet_timestamps': [],
        'obj_meta': []
    }
    
    # 初始化camera_timestamps
    for cam in cfg.data.get('cameras', [0, 1]):
        merged_metadata['camera_timestamps'][cam] = {
            'train_timestamps': [],
            'test_timestamps': []
        }
    
    total_images = 0
    for seq_idx, path in enumerate(paths):
        print(f"Loading sequence {seq_idx + 1}/{len(paths)}: {path}")
        
        # 读取单个序列
        scene_info = readSingleDairSequence(
            path, images, split_train, split_test, specified_sequence_id=cfg.specified_sequence_id[seq_idx], **kwargs
        )
        
        # 更新camera的uid以避免重复
        for cam in scene_info.train_cameras:
            cam.uid += total_images
        for cam in scene_info.test_cameras:
            cam.uid += total_images
            
        # 合并数据
        all_train_cameras.extend(scene_info.train_cameras)
        all_test_cameras.extend(scene_info.test_cameras)
        all_novel_view_cameras.extend(scene_info.novel_view_cameras)
        if scene_info.point_cloud is not None:
            all_point_clouds.append(scene_info.point_cloud)
            
        # 更新metadata
        seq_metadata = scene_info.metadata
        merged_metadata['sequence_lengths'].append(seq_metadata['num_images'])
        merged_metadata['obj_tracklets'].extend(seq_metadata['obj_tracklets'])
        merged_metadata['tracklet_timestamps'].extend(seq_metadata['tracklet_timestamps'])
        merged_metadata['obj_meta'].extend(seq_metadata['obj_meta'])
        
        # 更新timestamps
        for cam in cfg.data.get('cameras', [0, 1]):
            merged_metadata['camera_timestamps'][cam]['train_timestamps'].extend(
                seq_metadata['camera_timestamps'][cam]['train_timestamps']
            )
            merged_metadata['camera_timestamps'][cam]['test_timestamps'].extend(
                seq_metadata['camera_timestamps'][cam]['test_timestamps']
            )
            
        total_images += seq_metadata['num_images']
    
    # 合并点云
    if len(all_point_clouds) > 0:
        merged_point_cloud = all_point_clouds[0]
        for pc in all_point_clouds[1:]:
            merged_point_cloud.points = np.concatenate([merged_point_cloud.points, pc.points], axis=0)
            merged_point_cloud.colors = np.concatenate([merged_point_cloud.colors, pc.colors], axis=0)
            if hasattr(merged_point_cloud, 'normals') and merged_point_cloud.normals is not None:
                merged_point_cloud.normals = np.concatenate([merged_point_cloud.normals, pc.normals], axis=0)
        
        # 保存合并后的点云
        merged_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_merged.ply')
        if not os.path.exists(os.path.dirname(merged_ply_path)):
            os.makedirs(os.path.dirname(merged_ply_path))
        
        # 保存合并后的点云文件
        storePly(
            merged_ply_path,
            merged_point_cloud.points,
            merged_point_cloud.colors
        )
    else:
        merged_point_cloud = None
        merged_ply_path = None
    
    # 计算合并后的场景范围
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(all_novel_view_cameras)
    else:
        nerf_normalization = getNerfppNorm(all_train_cameras)
    
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    cfg.data.extent = float(nerf_normalization['radius'])
    
    # 计算合并后点云的sphere参数
    if merged_point_cloud is not None:
        sphere_normalization = get_Sphere_Norm(merged_point_cloud.points)
        merged_metadata['sphere_center'] = sphere_normalization['center']
        merged_metadata['sphere_radius'] = sphere_normalization['radius']
        print(f'Merged sphere extent: {sphere_normalization["radius"]}')
    else:
        # 如果没有点云，使用相机位置计算sphere参数
        all_cameras = all_train_cameras + all_test_cameras
        camera_positions = np.array([cam.T for cam in all_cameras])
        sphere_normalization = get_Sphere_Norm(camera_positions)
        merged_metadata['sphere_center'] = sphere_normalization['center']
        merged_metadata['sphere_radius'] = sphere_normalization['radius']
        print(f'Merged sphere extent (from cameras): {sphere_normalization["radius"]}')
    
    merged_metadata['scene_center'] = nerf_normalization['center']
    merged_metadata['scene_radius'] = nerf_normalization['radius']
    merged_metadata['num_images'] = total_images
    
    # 创建合并后的场景信息
    merged_scene_info = SceneInfo(
        point_cloud=merged_point_cloud,
        train_cameras=all_train_cameras,
        test_cameras=all_test_cameras,
        nerf_normalization=nerf_normalization,
        ply_path=merged_ply_path,  # 使用合并后的点云文件路径
        metadata=merged_metadata,
        novel_view_cameras=all_novel_view_cameras,
    )
    
    return merged_scene_info
    
    
    
