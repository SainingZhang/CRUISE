import os
import numpy as np
import cv2
import torch
import json
import pandas as pd
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm 
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import get_rays, sphere_intersection
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm

dair_track2label = {"Car": 0, "Truck": 1, "Van": 2, "Bus": 3, "Pedestrian": 4, "Cyclist": 5, "Tricyclist": 6, "Motorcyclist": 7 ,"Barrowlist": 8, "misc": -1}

image_heights = [1080, 1080]
image_widths = [1920, 1920]
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

def get_extrinsics_lidar_to_novatel(json_data):
    rotation = np.array(json_data['transform']['rotation'])
    translation = np.array(json_data['transform']['translation']).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation, translation))
    return extrinsic_matrix

def get_extrinsics(json_data):
    rotation = np.array(json_data['rotation'])
    translation = np.array(json_data['translation']).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation, translation))
    return extrinsic_matrix

def get_extrinsics_compute_lidar_to_world(lidar_to_novatel_data, novatel_to_world_data):
    lidar_to_novatel = get_extrinsics_lidar_to_novatel(lidar_to_novatel_data)
    novatel_to_world = get_extrinsics(novatel_to_world_data)
    
    lidar_to_world = np.dot(novatel_to_world, np.vstack((lidar_to_novatel, [0, 0, 0, 1])))
    
    return lidar_to_world

def get_intrinsics(json_data):
    cam_K = np.array(json_data['cam_K']).reshape(3, 3)
    return cam_K

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])
    
    return obj_pose_vehicle, obj_pose_world



def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras=[0, 1]):
    tracklets_ls = []    
    objects_info = {}

    if cfg.data.get('use_tracker', False): # TODO 这是个啥
        tracklet_path = os.path.join(datadir, 'track/track_info_castrack.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis_castrack.json')
    else:
        tracklet_path = os.path.join(datadir, 'track/track_info.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis.json')

    print(f'Loading from : {tracklet_path}')
    f = open(tracklet_path, 'r')
    tracklets_str = f.read().splitlines()
    tracklets_str = tracklets_str[1:]

    f = open(tracklet_camera_vis_path, 'r')
    tracklet_camera_vis = json.load(f)

    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, 'images')
    n_cameras = 2
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    n_obj_in_frame = np.zeros(n_frames)


    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        track_id_str = tracklet[1]
        object_class = tracklet[2]
        
        if object_class in ['Pedestrian', "Cyclist","Tricyclist","Motorcyclist","Barrowlist" ,'misc']:
            continue
        
        cameras_vis_list = tracklet_camera_vis[track_id_str][str(frame_id)]
        join_cameras_list = list(set(cameras) & set(cameras_vis_list))
        if len(join_cameras_list) == 0:
            continue
                
        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]['track_id'] = track_id
            objects_info[track_id]['class'] = object_class
            objects_info[track_id]['class_label'] = dair_track2label[object_class]
            objects_info[track_id]['height'] = float(tracklet[4])
            objects_info[track_id]['width'] = float(tracklet[5])
            objects_info[track_id]['length'] = float(tracklet[6])
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))
            
        tr_array = np.concatenate(
            [np.array(tracklet[:2]).astype(np.float64), np.array([type]), np.array(tracklet[4:]).astype(np.float64)]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1
        

    tracklets_array = np.array(tracklets_ls)
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0


    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:            
            ego_pose = ego_poses[frame_id]
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world


    # Remove static objects
    print("Removing static objects")
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)
            
    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0    
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = cfg.data.get('box_scale', 1.0)
    print('box scale: ', box_scale)


    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info   
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'Pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False
        
        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale
        
        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)
        
        objects_info[key] = obj


    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )

    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )

    return objects_tracklets_world, objects_tracklets_vehicle, objects_info


def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]
    
    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)
    
    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)
        
    return tracklets, frame_timestamps
    
def generate_dataparser_outputs(
        datadir, 
        selected_frames=None, 
        build_pointcloud=True, 
        cameras=[0, 1],
        specified_sequence_id=None
    ):

    source_path = '/mnt/xuhr'

    with open(f'{source_path}/V2X-Seq-SPD/cooperative/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    cooperative_data_info_df = pd.DataFrame(data_info_data)
    cooperative_filtered_df = cooperative_data_info_df[cooperative_data_info_df['infrastructure_sequence'] == specified_sequence_id]


    with open(f'{source_path}/V2X-Seq-SPD/infrastructure-side/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    infrastructure_data_info_df = pd.DataFrame(data_info_data)

    infrastructure_filtered_df = infrastructure_data_info_df[infrastructure_data_info_df['sequence_id'] == specified_sequence_id]
    infrastructure_filtered_df = infrastructure_filtered_df.loc[infrastructure_filtered_df['frame_id'].isin(cooperative_filtered_df['infrastructure_frame'])]


    with open(f'{source_path}/V2X-Seq-SPD/vehicle-side/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    vehicle_data_info_df = pd.DataFrame(data_info_data)
    vehicle_filtered_df = vehicle_data_info_df[vehicle_data_info_df['sequence_id'] == specified_sequence_id]
    vehicle_filtered_df = vehicle_filtered_df.loc[vehicle_filtered_df['frame_id'].isin(cooperative_filtered_df['vehicle_frame'])]


    infrastructure_calib_camera_intrinsic_paths = infrastructure_filtered_df['calib_camera_intrinsic_path'].tolist()
    vehicle_calib_camera_intrinsic_paths = vehicle_filtered_df['calib_camera_intrinsic_path'].tolist()

    car_list_infrastructure = [os.path.splitext(os.path.basename(path))[0] for path in infrastructure_calib_camera_intrinsic_paths]
    car_list_vehicle = [os.path.splitext(os.path.basename(path))[0] for path in vehicle_calib_camera_intrinsic_paths]

    vehicle_filtered_df.reset_index(drop=True, inplace=True)
    infrastructure_filtered_df.reset_index(drop=True, inplace=True)
    
    image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.jpg')))
    num_frames_all = len(image_filenames_all) // 2
    num_cameras = len(cameras)
    
    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1
    
    source_path = "/mnt/xuhr/V2X-Seq-SPD"
    car_list = car_list_vehicle

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        vehicle_lidar_to_novatel_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_novatel', car_list_vehicle[idx] +'.json'))
        infrastructure_camera_intrinsics_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'camera_intrinsic',car_list_infrastructure[idx]+'.json'))
        vehicle_camera_intrinsics_json = read_json_file(os.path.join(source_path,'vehicle-side', 'calib', 'camera_intrinsic', car_list_vehicle[idx] +'.json'))
        infrastructure_lidar_to_camera_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_camera',car_list_infrastructure[idx]+'.json'))
        vehicle_lidar_to_camera_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_camera', car_list_vehicle[idx] +'.json'))
        break
        
    lidar2cam = get_extrinsics(vehicle_lidar_to_camera_json)
    lidar2cam_padded = pad_poses(lidar2cam)
    cam2lidar = np.linalg.inv(lidar2cam_padded)
    
    intrinsics = []
    extrinsics = []
    
    intrinsics.append(np.array(vehicle_camera_intrinsics_json["cam_K"]).reshape(3,3))
    intrinsics.append(np.array(infrastructure_camera_intrinsics_json["cam_K"]).reshape(3,3))
    
    extrinsics.append(cam2lidar)
    extrinsics.append(None)
    
    
    ego_frame_poses = []
    ego_cam_poses = []
    cam_road2lidar_cars = []
    c2ws_temp = [[] for i in range(2)]

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        # Read infrastructure cam info
        infrastructure_camera_intrinsics_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'camera_intrinsic',car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_camera_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_camera',car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_world_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_world',car_list_infrastructure[idx]+'.json'))

        # Read vehicle cam info
        vehicle_camera_intrinsics_json = read_json_file(os.path.join(source_path,'vehicle-side', 'calib', 'camera_intrinsic', car_list_vehicle[idx] +'.json'))
        vehicle_lidar_to_camera_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_camera', car_list_vehicle[idx] +'.json'))
        vehicle_lidar_to_novatel_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_novatel', car_list_vehicle[idx] +'.json')) # 车端LiDAR到定位系统的外参文件
        vehicle_novatel_to_world_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'novatel_to_world', car_list_vehicle[idx] +'.json')) # 车端LiDAR到定位系统的外参文件

        Ks = np.array([get_intrinsics(vehicle_camera_intrinsics_json), get_intrinsics(infrastructure_camera_intrinsics_json), ]) 
        lidar2cam = np.array([get_extrinsics(vehicle_lidar_to_camera_json), get_extrinsics(infrastructure_lidar_to_camera_json)])
        lidar2world = np.array([get_extrinsics_compute_lidar_to_world(vehicle_lidar_to_novatel_json, vehicle_novatel_to_world_json), get_extrinsics(infrastructure_lidar_to_world_json)])
        
        novatel_to_world = get_extrinsics(vehicle_novatel_to_world_json)
        
        novatel_to_world_padded = pad_poses(novatel_to_world)

        lidar2cam_padded = pad_poses(lidar2cam)
        lidar2world_padded = pad_poses(lidar2world)
        cam2lidar = np.linalg.inv(lidar2cam_padded)

        c2w = lidar2world_padded @ cam2lidar
        w2c = np.linalg.inv(c2w)
        
        cam_road2cam_car = w2c[0] @ c2w[1]
        cam_road2lidar_car = cam2lidar[0] @ cam_road2cam_car
        
        
        ego_frame_poses.append(lidar2world_padded[0])
        ego_cam_poses.append(lidar2world_padded[0])
        cam_road2lidar_cars.append(cam_road2lidar_car)
        c2ws_temp[0].append(c2w[0])
        c2ws_temp[1].append(c2w[1])
        
        
    ego_frame_poses = np.array(ego_frame_poses)
    ego_cam_poses = np.array(ego_cam_poses)  
    cam_road2lidar_cars = np.array(cam_road2lidar_cars)  
            
    # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []
    
    ixts = []
    exts = []
    poses = []
    c2ws = []
    
    frames_timestamps = []
    cams_timestamps = []
        
    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )
    
   
    
    for frame in range(start_frame, end_frame+1):
        frames_timestamps.append(int(vehicle_filtered_df.loc[frame,'image_timestamp']))

    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            if cam==0:
                pose = ego_cam_poses[frame]
            elif cam==1:
                pose = ego_cam_poses[frame]
                ext = cam_road2lidar_cars[frame]
            c2w = c2ws_temp[cam] [frame]

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)
            
            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)         
          
            if cam == 0:
                cams_timestamps.append(int(vehicle_filtered_df.loc[frame,'image_timestamp']))
            elif cam == 1:
                cams_timestamps.append(int(infrastructure_filtered_df.loc[frame,'image_timestamp']))
                
    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = min(cams_timestamps.min(), frames_timestamps.min()), max(cams_timestamps.max(), frames_timestamps.max())
 
    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        datadir, 
        selected_frames, 
        ego_frame_poses,
        cameras,
    )
    
    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        object_start_timestamp = int(vehicle_filtered_df.loc[object_start_frame,'image_timestamp']) - timestamp_offset - 0.1
        object_end_timestamp = int(vehicle_filtered_df.loc[object_end_frame,'image_timestamp']) - timestamp_offset + 0.1
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)
        
    result = dict()
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info 
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps

    # get object bounding mask
    obj_bounds = []
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights[cam], image_widths[cam]
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0:
                obj_pose_vehicle = np.eye(4)    
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height], 
                                    [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_vehicle = corners_local @ obj_pose_vehicle.T # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(   
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext), 
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)
        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds         

    
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)
        
        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []

        print('no initialize from sfm pointcloud')

        print('initialize from lidar pointcloud')
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()

        for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
            idxs = list(range(i * num_cameras, (i+1) * num_cameras))
            cams_frame = [cams[idx] for idx in idxs]
            image_filenames_frame = [image_filenames[idx] for idx in idxs]
            
            raw_3d = pts3d_dict[frame]
            raw_2d = pts2d_dict[frame]
            
            # use the first projection camera
            points_camera_all = raw_2d[..., 0]
            points_projw_all = raw_2d[..., 1]
            points_projh_all = raw_2d[..., 2]

            # each point should be observed by at least one camera in camera lists
            mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)
            
            # get filtered LiDAR pointcloud position and color        
            points_xyz_vehicle = raw_3d[mask]

            # transfrom LiDAR pointcloud from vehicle frame to world frame
            ego_pose = ego_frame_poses[frame]
            points_xyz_vehicle = np.concatenate(
                [points_xyz_vehicle, 
                np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
            )
            points_xyz_world = points_xyz_vehicle @ ego_pose.T
            
            points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
            points_camera = points_camera_all[mask]
            points_projw = points_projw_all[mask]
            points_projh = points_projh_all[mask]

            for cam, image_filename in zip(cams_frame, image_filenames_frame):
                mask_cam = (points_camera == cam)
                image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.

                mask_projw = points_projw[mask_cam]
                mask_projh = points_projh[mask_cam]
                mask_rgb = image[mask_projh, mask_projw]
                points_rgb[mask_cam] = mask_rgb
        
            # filer points in tracking bbox
            points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

            for tracklet in object_tracklets_vehicle[i]:
                track_id = int(tracklet[0])
                if track_id >= 0:
                    obj_pose_vehicle = np.eye(4)                    
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                    obj_pose_vehicle[:3, 3] = tracklet[1:4]
                    vehicle2local = np.linalg.inv(obj_pose_vehicle)
                    
                    points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                    points_xyz_obj = points_xyz_obj[..., :3]
                    
                    length = object_info[track_id]['length']
                    width = object_info[track_id]['width']
                    height = object_info[track_id]['height']
                    bbox = [[-length/2, -width/2, -height/2], [length/2, width/2, height/2]]
                    obj_corners_3d_local = bbox_to_corner3d(bbox)
                    
                    points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                    points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                    points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                    points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])
        
            points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
            
            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)
            
        initial_num_obj = 20000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    # downsample lidar pointcloud with voxels
                    points_lidar = o3d.geometry.PointCloud()
                    points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                    points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)                                
                elif k.startswith('obj'):
                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]
                        
                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb
                
                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

      
        print('No colmap pointcloud')
        points_bkgd_xyz = points_lidar_xyz
        points_bkgd_rgb = points_lidar_rgb
        
        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb
            
        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict


        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict

        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}_{specified_sequence_id}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue
    return result