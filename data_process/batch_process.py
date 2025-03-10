import json
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import shutil
from glob import glob
import glob
from tqdm import tqdm

sequence_list = ['0000',
 '0001',
 '0002',
 '0003',
 '0004',
 '0005',
 '0007',
 '0008',
 '0010',
 '0014',
 '0015',
 '0016',
 '0017',
 '0018',
 '0020',
 '0021',
 '0022',
 '0023',
 '0025',
 '0029',
 '0030',
 '0032',
 '0033',
 '0034',
 '0035',
 '0036',
 '0037',
 '0040',
 '0041',
 '0042',
 '0047',
 '0048',
 '0049',
 '0050',
 '0052',
 '0054',
 '0055',
 '0056',
 '0057',
 '0058',
 '0059',
 '0060',
 '0061',
 '0062',
 '0063',
 '0066',
 '0068',
 '0070',
 '0071',
 '0072',
 '0073',
 '0075',
 '0077',
 '0078',
 '0079',
 '0080',
 '0081',
 '0082',
 '0084',
 '0085',
 '0086',
 '0087',
 '0088',
 '0089',
 '0092',
 '0093',
 '0094']

source_path = "/mnt/zhangsn/data/V2X-Seq-SPD"  
des_path = "/mnt/zhangsn/data/V2X-Seq-SPD-Processed"

def process(specified_sequence_id):
    
    import json
    import numpy as np
    import os
    from tqdm import tqdm
    import pandas as pd
    import shutil
    from glob import glob
    import glob
    
    def get_padded_number(number, width=6):
        return str(number).zfill(width)

    def read_json_file(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

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

    with open(f'{source_path}/cooperative/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    cooperative_data_info_df = pd.DataFrame(data_info_data)
    cooperative_filtered_df = cooperative_data_info_df[cooperative_data_info_df['infrastructure_sequence'] == specified_sequence_id]


    with open(f'{source_path}/infrastructure-side/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    infrastructure_data_info_df = pd.DataFrame(data_info_data)

    infrastructure_filtered_df = infrastructure_data_info_df[infrastructure_data_info_df['sequence_id'] == specified_sequence_id]
    infrastructure_filtered_df = infrastructure_filtered_df.loc[infrastructure_filtered_df['frame_id'].isin(cooperative_filtered_df['infrastructure_frame'])]


    with open(f'{source_path}/vehicle-side/data_info.json', 'r') as file:
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

    vehicle_source_path = f'{source_path}/vehicle-side'
    road_source_path = f'{source_path}/infrastructure-side'

    move_file_list = ['images']
    image_file_list = ['VEHICLE', 'ROAD']

    destination_folder = f'{des_path}/{specified_sequence_id}_0_original'

    for file_class in move_file_list:
        destination_image_folder = os.path.join(destination_folder, file_class)
        os.makedirs(destination_image_folder,exist_ok=True)
        if file_class == 'images':
            for image_class in image_file_list:
                if image_class =='VEHICLE':
                    for index, row in vehicle_filtered_df.iterrows():
                        _source_path = os.path.join(vehicle_source_path,row['image_path'])
                        destination_path = os.path.join(destination_image_folder, f'{get_padded_number(index)}_0.jpg')
                        shutil.copy(_source_path, destination_path)

                    print("Car image file copy completed.")
                elif image_class =='ROAD':
                    for index, row in infrastructure_filtered_df.iterrows():
                        _source_path = os.path.join(road_source_path,row['image_path'])
                        destination_path = os.path.join(destination_image_folder, f'{get_padded_number(index)}_1.jpg')
                        shutil.copy(_source_path, destination_path)
                    print("Road image file copy completed.")
                    
    # ## Single
    # image_dir = f"data/dair-v2x/exp/{specified_sequence_id}_0_original/images"
    # image_filenames_all = sorted(glob(os.path.join(image_dir, '*.jpg')))
    # image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
    # for image_filename in image_filenames_all:
    #     image_basename = os.path.basename(image_filename)
    #     cam = image_filename_to_cam(image_basename)
    #     os.makedirs(f'data/dair-v2x/exp/{specified_sequence_id}_1_single/images', exist_ok=True)
    #     destination_path = f'data/dair-v2x/exp/{specified_sequence_id}_1_single/images/{image_basename}'
    #     if cam ==0:
    #         shutil.copy(image_filename, destination_path)
    # image_dir = f"data/dair-v2x/exp/{specified_sequence_id}_0_original/sky_mask"
    # image_filenames_all = sorted(glob(os.path.join(image_dir, '*.jpg')))
    # image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
    # for image_filename in image_filenames_all:
    #     image_basename = os.path.basename(image_filename)
    #     cam = image_filename_to_cam(image_basename)
    #     os.makedirs(f'data/dair-v2x/exp/{specified_sequence_id}_1_single/sky_mask', exist_ok=True)
    #     destination_path = f'data/dair-v2x/exp/{specified_sequence_id}_1_single/sky_mask/{image_basename}'
    #     if cam ==0:
    #         shutil.copy(image_filename, destination_path)
            
            
    # Annotation file generation
    ##  Cooperative-view annotation
    car_list = car_list_vehicle

    def convert_annotations(output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        track_info_path = os.path.join(output_dir, 'track_info.txt')
        track_camera_vis_path = os.path.join(output_dir, 'track_camera_vis.json')

        track_info_lines = ['frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed']
        track_camera_vis = {}
        
        for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
            
            annotations = read_json_file(os.path.join(source_path,'cooperative', 'label', car_list_vehicle[idx] +'.json'))
            frame_id = idx

            for ann in annotations:
                track_id = ann['track_id']
                object_class = ann['type']
                alpha = ann['alpha']
                box_height = ann['3d_dimensions']['h']
                box_width = ann['3d_dimensions']['w']
                box_length = ann['3d_dimensions']['l']
                box_center_x = ann['3d_location']['x']
                box_center_y = ann['3d_location']['y']
                box_center_z = ann['3d_location']['z']
                box_heading = ann['rotation']
                speed = 0  

                track_info_lines.append(f'{frame_id} {track_id} {object_class} {alpha} {box_height} {box_width} {box_length} {box_center_x} {box_center_y} {box_center_z} {box_heading} {speed}')

                if track_id not in track_camera_vis:
                    track_camera_vis[track_id] = {}
                    
                if ann["from_side"] =='coop':
                    if ann['occluded_state'] == 1 or  ann['occluded_state'] == 2:
                        track_camera_vis[track_id][frame_id] = [1]
                    else:
                        track_camera_vis[track_id][frame_id] = list(range(2))
                elif ann["from_side"]=="veh":
                    if ['occluded_state'] == 2 :
                        track_camera_vis[track_id][frame_id] = []
                    else:    
                        track_camera_vis[track_id][frame_id] = [0]
                elif ann["from_side"]=='inf':
                    if ['occluded_state'] == 2 :
                        track_camera_vis[track_id][frame_id] = []
                    else:
                        track_camera_vis[track_id][frame_id] = [1]

        with open(track_info_path, 'w') as f:
            f.write('\n'.join(track_info_lines))

        with open(track_camera_vis_path, 'w') as f:
            json.dump(track_camera_vis, f, indent=2)

    output_dir = f'{des_path}/{specified_sequence_id}_0_original/track' 

    convert_annotations(output_dir)

    # def convert_annotations(output_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     track_info_path = os.path.join(output_dir, 'track_info.txt')
    #     track_camera_vis_path = os.path.join(output_dir, 'track_camera_vis.json')

    #     track_info_lines = ['frame_id track_id object_class alpha box_height box_width box_length box_center_x box_center_y box_center_z box_heading speed']
    #     track_camera_vis = {}
        
    #     for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
            
    #         annotations = read_json_file(os.path.join(source_path,'cooperative', 'label', car_list_vehicle[idx] +'.json'))
    #         frame_id = idx

    #         for ann in annotations:
    #             if ann["veh_track_id"] !="-1":
                    
    #                 track_id = ann['track_id']
    #                 object_class = ann['type']
    #                 alpha = ann['alpha']
    #                 box_height = ann['3d_dimensions']['h']
    #                 box_width = ann['3d_dimensions']['w']
    #                 box_length = ann['3d_dimensions']['l']
    #                 box_center_x = ann['3d_location']['x']
    #                 box_center_y = ann['3d_location']['y']
    #                 box_center_z = ann['3d_location']['z']
    #                 box_heading = ann['rotation']
    #                 speed = 0  

    #                 track_info_lines.append(f'{frame_id} {track_id} {object_class} {alpha} {box_height} {box_width} {box_length} {box_center_x} {box_center_y} {box_center_z} {box_heading} {speed}')

    #                 if track_id not in track_camera_vis:
    #                     track_camera_vis[track_id] = {}
                        
    #                 track_camera_vis[track_id][frame_id] = [0]
                    

    #     with open(track_info_path, 'w') as f:
    #         f.write('\n'.join(track_info_lines))

    #     with open(track_camera_vis_path, 'w') as f:
    #         json.dump(track_camera_vis, f, indent=2)


    # output_dir = f'data/dair-v2x/exp/{specified_sequence_id}_1_single/track' 
    # convert_annotations(output_dir)

    ## Need to remove the ego vehicle's label at the roadside.
    # Because retaining the annotation box of the ego vehicle during the training process can lead to data processing errors, the annotation box of the ego vehicle needs to be deleted.
    # input_file = f'{des_path}/{specified_sequence_id}_0_original/track/track_info.txt'
    # output_file = f'{des_path}/{specified_sequence_id}_0_original/track/track_info_temp.txt'

    # with open(input_file, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()

    # filtered_lines = []
    # for line in lines:
    #     elements = line.split()
    #     if len(elements) < 2 or elements[1] != '005686':
    #         '''
    #         0000:002834
    #         0015:007570
    #         0022:003821
    #         0066:005686
    #         '''
    #         filtered_lines.append(line)

    # with open(output_file, 'w', encoding='utf-8') as file:
    #     file.writelines(filtered_lines)

    # print(
    #     f"Processing completed, deleted all rows with the second position as xxxx, and the result saved to '{output_file}'。")
    # Point cloud file generation
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

    car_list = car_list_vehicle
    camera_w2cs = dict()
    camera_lidar2ws = dict()
    camera_lidar2camera = dict()
    world2lidars = []

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        infrastructure_camera_intrinsics_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'camera_intrinsic',car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_camera_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_camera',car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_world_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_world',car_list_infrastructure[idx]+'.json'))

        vehicle_camera_intrinsics_json = read_json_file(os.path.join(source_path,'vehicle-side', 'calib', 'camera_intrinsic', car_list_vehicle[idx] +'.json'))
        vehicle_lidar_to_camera_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_camera', car_list_vehicle[idx] +'.json'))
        vehicle_lidar_to_novatel_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_novatel', car_list_vehicle[idx] +'.json')) 
        vehicle_novatel_to_world_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'novatel_to_world', car_list_vehicle[idx] +'.json')) 

        Ks = np.array([get_intrinsics(infrastructure_camera_intrinsics_json), get_intrinsics(vehicle_camera_intrinsics_json)]) 
        lidar2cam = np.array([get_extrinsics(infrastructure_lidar_to_camera_json),get_extrinsics(vehicle_lidar_to_camera_json)])
        lidar2world = np.array([get_extrinsics(infrastructure_lidar_to_world_json), get_extrinsics_compute_lidar_to_world(vehicle_lidar_to_novatel_json, vehicle_novatel_to_world_json)])
        
        novatel_to_world = get_extrinsics(vehicle_novatel_to_world_json)
        
        novatel_to_world_padded = pad_poses(novatel_to_world)

        lidar2cam_padded = pad_poses(lidar2cam)
        lidar2world_padded = pad_poses(lidar2world)
        
        world2lidar = np.linalg.inv(lidar2world_padded)
        world2lidars.append(world2lidar)
        cam2lidar = np.linalg.inv(lidar2cam_padded)
        c2w = lidar2world_padded @ cam2lidar
        w2c = np.linalg.inv(c2w)
        
        cam_road2cam_car = w2c[1] @ c2w[0]
        cam_road2lidar_car = cam2lidar[1] @ cam_road2cam_car
        lidar_car2cam_road = np.linalg.inv(cam_road2lidar_car)
        
        if idx not in camera_lidar2camera:
            camera_lidar2camera[idx] = {} 
        if idx not in camera_w2cs:
            camera_w2cs[idx] = {} 
        
        camera_lidar2camera[idx]['cam0'] = lidar2cam_padded[1]
        camera_lidar2camera[idx]['cam1'] = lidar_car2cam_road

    camera_intrinsics = {
        "cam0": Ks[1],
        "cam1": Ks[0],
    }
    ## Cooperative pointcloud 
    import open3d as o3d

    point_xyz_vehicle_combineds= []

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        pcd_infrastructure = o3d.io.read_point_cloud(os.path.join(source_path,'infrastructure-side', "velodyne", car_list_infrastructure[idx] + ".pcd"))
        point_infrastructure = np.asarray(pcd_infrastructure.points) 
        intensities_infrastructure = np.zeros((point_infrastructure.shape[0], 1))
        elongation_infrastructure = np.zeros((point_infrastructure.shape[0], 1))
        timestamp_pts_infrastructure = np.zeros((point_infrastructure.shape[0], 1))
        point_data_infrastructure = np.hstack((point_infrastructure, intensities_infrastructure, elongation_infrastructure, timestamp_pts_infrastructure))
        point_xyz_infrastructure, intensities_infrastructure, elongation_infrastructure, timestamp_pts_infrastructure = np.split(point_data_infrastructure, [3, 4, 5], axis=1)
        point_xyz_world_infrastructure = (np.pad(point_xyz_infrastructure, ((0, 0), (0, 1)), constant_values=1) @ lidar2world_padded[0].T)[:, :3]

        pcd_vehicle = o3d.io.read_point_cloud(os.path.join(source_path,'vehicle-side' , "velodyne", car_list_vehicle[idx] + ".pcd"))
        point_vehicle = np.asarray(pcd_vehicle.points) 
        intensities_vehicle = np.zeros((point_vehicle.shape[0], 1))
        elongation_vehicle = np.zeros((point_vehicle.shape[0], 1))
        timestamp_pts_vehicle = np.zeros((point_vehicle.shape[0], 1))
        point_data_vehicle = np.hstack((point_vehicle, intensities_vehicle, elongation_vehicle, timestamp_pts_vehicle))
        point_xyz_vehicle, intensities_vehicle, elongation_vehicle, timestamp_pts_vehicle = np.split(point_data_vehicle, [3, 4, 5], axis=1)
        point_xyz_world_vehicle = (np.pad(point_xyz_vehicle, ((0, 0), (0, 1)), constant_values=1) @ lidar2world_padded[1].T)[:, :3]
        
        point_xyz_world_combined = np.vstack((point_xyz_world_infrastructure, point_xyz_world_vehicle))
        
        world2lidar = world2lidars[idx]
        point_xyz_vehicle_combined = (np.pad(point_xyz_world_combined, ((0, 0), (0, 1)), constant_values=1) @ world2lidar[1].T)[:, :3]
        point_xyz_vehicle_combineds.append(point_xyz_vehicle_combined)

    def load_pcd(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)

    def project_to_camera(pts_3d_camera, intrinsic, camera_id):
        img_width = 1920
        img_height  = 1080
        z = pts_3d_camera[:, 2]
        valid_mask = z > 1e-6 
        
        pts_3d_camera = pts_3d_camera[:,:3]
        
        pts_2d_camera = np.dot(pts_3d_camera[valid_mask], intrinsic.T)
        
        pts_2d_camera[:, 0] /= pts_2d_camera[:, 2]
        pts_2d_camera[:, 1] /= pts_2d_camera[:, 2]
        
        camera_projection = np.zeros((pts_2d_camera.shape[0], 6), dtype=np.int16)
        camera_projection[:, 0] = camera_id 
        camera_projection[:, 1] = np.clip(np.round(pts_2d_camera[:, 0]), 0, img_width-1).astype(np.int16) 
        camera_projection[:, 2] = np.clip(np.round(pts_2d_camera[:, 1]), 0, img_height-1).astype(np.int16) 
        
        return camera_projection, valid_mask

    def process_lidar_data( camera_intrinsics, camera_lidar2camera, output_dir):
        pts_3d_all = dict()
        pts_2d_all = dict()
        
        for frame_idx, frame_id in enumerate(car_list_vehicle):
        
            pts_3d_lidar = point_xyz_vehicle_combineds[frame_idx]
            
            points_homogeneous = np.hstack((pts_3d_lidar, np.ones((pts_3d_lidar.shape[0], 1))))

            
            pts_3d_frame = []
            pts_2d_frame = []
            
            for cam_idx, (cam_name, intrinsic) in enumerate(camera_intrinsics.items()):
                pts_3d_camera = points_homogeneous @ camera_lidar2camera[frame_idx][cam_name].T   
                camera_projection, valid_mask = project_to_camera(pts_3d_camera, intrinsic, cam_idx)
                
                pts_3d_frame.append(pts_3d_lidar[valid_mask]) 
                pts_2d_frame.append(camera_projection)
            
            if pts_3d_frame and pts_2d_frame:
                pts_3d_all[frame_idx] = np.concatenate(pts_3d_frame, axis=0)
                pts_2d_all[frame_idx] = np.concatenate(pts_2d_frame, axis=0)
        
        np.savez_compressed(os.path.join(output_dir, 'pointcloud_cooperative.npz'), 
                            pointcloud=pts_3d_all, 
                            camera_projection=pts_2d_all)
        print("Processing LiDAR data done...")

    output_dir = f"{des_path}/{specified_sequence_id}_0_original"

    process_lidar_data( camera_intrinsics, camera_lidar2camera, output_dir)

    ## Single-side pointcloud
    import os
    import numpy as np
    import open3d as o3d

    def load_pcd(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)

    def project_to_camera(pts_3d_camera, intrinsic, camera_id):
        img_width = 1920
        img_height  = 1080
        z = pts_3d_camera[:, 2]
        valid_mask = z > 1e-6 
        
        pts_3d_camera = pts_3d_camera[:,:3]
        
        pts_2d_camera = np.dot(pts_3d_camera[valid_mask], intrinsic.T)
        
        pts_2d_camera[:, 0] /= pts_2d_camera[:, 2]
        pts_2d_camera[:, 1] /= pts_2d_camera[:, 2]
        
        camera_projection = np.zeros((pts_2d_camera.shape[0], 6), dtype=np.int16)
        camera_projection[:, 0] = camera_id  
        camera_projection[:, 1] = np.clip(np.round(pts_2d_camera[:, 0]), 0, img_width-1).astype(np.int16)  
        camera_projection[:, 2] = np.clip(np.round(pts_2d_camera[:, 1]), 0, img_height-1).astype(np.int16) 
        
        return camera_projection, valid_mask

    def process_lidar_data(seq_dir, camera_intrinsics, camera_lidar2camera, output_dir):
        
        pts_3d_all = dict()
        pts_2d_all = dict()
        
        for frame_idx, frame_id in enumerate(car_list_vehicle):
            
            pcd_file = os.path.join(seq_dir, frame_id) + '.pcd'
            pts_3d_lidar = load_pcd(pcd_file)
            
            pts_3d_lidar = pts_3d_lidar[:, :3]
            
            points_homogeneous = np.hstack((pts_3d_lidar, np.ones((pts_3d_lidar.shape[0], 1))))

            pts_3d_frame = []
            pts_2d_frame = []
            
            for cam_idx, (cam_name, intrinsic) in enumerate(camera_intrinsics.items()):
                pts_3d_camera = points_homogeneous @ camera_lidar2camera[frame_idx][cam_name].T   
                camera_projection, valid_mask = project_to_camera(pts_3d_camera, intrinsic, cam_idx)
                
                pts_3d_frame.append(pts_3d_lidar[valid_mask]) 
                pts_2d_frame.append(camera_projection)
            
            if pts_3d_frame and pts_2d_frame: 
                pts_3d_all[frame_idx] = np.concatenate(pts_3d_frame, axis=0)
                pts_2d_all[frame_idx] = np.concatenate(pts_2d_frame, axis=0)
        
        np.savez_compressed(os.path.join(output_dir, 'pointcloud.npz'), 
                            pointcloud=pts_3d_all, 
                            camera_projection=pts_2d_all)
        print("Processing LiDAR data done...")


    seq_dir = f"{source_path}/vehicle-side/velodyne"
    output_dir = f"{des_path}/{specified_sequence_id}_0_original"

    process_lidar_data(seq_dir, camera_intrinsics, camera_lidar2camera, output_dir)

    # generate lidar depth
    import os
    os.chdir('/mnt/xuhr/CRUISE')
    import sys
    import os
    sys.path.append(os.getcwd())
    import argparse
    import numpy as np
    import cv2
    from glob import glob
    from tqdm import tqdm
    from lib.utils.img_utils import visualize_depth_numpy

    import json
    import pandas as pd

    image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
    image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

    def read_json_file(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data


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

    def pad_poses(p):
        bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
        return np.concatenate([p[..., :3, :4], bottom], axis=-2)

    def get_intrinsics(json_data):
        cam_K = np.array(json_data['cam_K']).reshape(3, 3)
        return cam_K

    # single frame sparse lidar depth
    def generate_lidar_depth_seperate(datadir, camera_lidar2camera):
        save_dir = os.path.join(datadir, 'lidar_depth')
        os.makedirs(save_dir, exist_ok=True)
        
        image_dir = os.path.join(datadir, 'images')
        image_files = glob(image_dir + "/*.jpg") 
        image_files += glob(image_dir + "/*.png")
        image_files = sorted(image_files)
        
        
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  
        
        pointcloud_path_cooperative = os.path.join(datadir, 'pointcloud_cooperative.npz')
        pts3d_dict_coop = np.load(pointcloud_path_cooperative, allow_pickle=True)['pointcloud'].item()
        pts2d_dict_coop = np.load(pointcloud_path_cooperative, allow_pickle=True)['camera_projection'].item()  

        for image_filename in tqdm(image_files):
            image = cv2.imread(image_filename)
            h, w = image.shape[:2]
            
            image_basename = os.path.basename(image_filename)
            frame = image_filename_to_frame(image_basename)
            cam = image_filename_to_cam(image_basename)
            
            if cam == 0:
                depth_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.npy')
                depth_vis_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.png')
                
                raw_3d = pts3d_dict[frame]
                raw_2d = pts2d_dict[frame]
                    
                num_pts = raw_3d.shape[0]
                pts_idx = np.arange(num_pts)
                pts_idx = np.tile(pts_idx[..., None], (1, 2)).reshape(-1) # (num_pts * 2)
                raw_2d = raw_2d.reshape(-1, 3) # (num_pts * 2, 3)
                mask = (raw_2d[:, 0] == cam)
                
                points_xyz = raw_3d[pts_idx[mask]]
                points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
                
                lidar2cam = camera_lidar2camera[cam][frame]
                
                points_xyz_cam = points_xyz @ lidar2cam.T
                points_depth = points_xyz_cam[..., 2]

                valid_mask = points_depth > 0.
                
                points_xyz_pixel = raw_2d[mask][:, 1:3]
                points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
                points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
                points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)
                
                depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
                u, v = points_coord[:, 0], points_coord[:, 1]
                indices = v * w + u
                np.minimum.at(depth, indices, points_depth[valid_mask])
                depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
                valid_depth_pixel = (depth != 0)
                valid_depth_value = depth[valid_depth_pixel]
                valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
                            
                depth_file = dict()
                depth_file['mask'] = valid_depth_pixel
                depth_file['value'] = valid_depth_value
                np.save(depth_path, depth_file)
                
            elif cam == 1:
                depth_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.npy')
                depth_vis_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.png')
                
                raw_3d = pts3d_dict_coop[frame]
                raw_2d = pts2d_dict_coop[frame]
                    
                num_pts = raw_3d.shape[0]
                pts_idx = np.arange(num_pts)
                pts_idx = np.tile(pts_idx[..., None], (1, 2)).reshape(-1) # (num_pts * 2)
                raw_2d = raw_2d.reshape(-1, 3) # (num_pts * 2, 3)
                mask = (raw_2d[:, 0] == cam)
                
                points_xyz = raw_3d[pts_idx[mask]]
                points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)
                
                lidar2cam = camera_lidar2camera[cam][frame]
                
                points_xyz_cam = points_xyz @ lidar2cam.T
                points_depth = points_xyz_cam[..., 2]

                valid_mask = points_depth > 0.
                
                points_xyz_pixel = raw_2d[mask][:, 1:3]
                points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
                points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
                points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)
                
                depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
                u, v = points_coord[:, 0], points_coord[:, 1]
                indices = v * w + u
                np.minimum.at(depth, indices, points_depth[valid_mask])
                depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
                valid_depth_pixel = (depth != 0)
                valid_depth_value = depth[valid_depth_pixel]
                valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
                            
                depth_file = dict()
                depth_file['mask'] = valid_depth_pixel
                depth_file['value'] = valid_depth_value
                np.save(depth_path, depth_file)

            try:
                depth = depth.reshape(h, w).astype(np.float32)
                depth_vis, _ = visualize_depth_numpy(depth)
                depth_on_img = image[..., [2, 1, 0]]
                depth_on_img[depth > 0] = depth_vis[depth > 0]
                cv2.imwrite(depth_vis_path, depth_on_img)      
            except:
                print(f'error in visualize depth of {image_filename}, depth range: {depth.min()} - {depth.max()}')
        

    with open(f'{source_path}/cooperative/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    cooperative_data_info_df = pd.DataFrame(data_info_data)
    cooperative_filtered_df = cooperative_data_info_df[cooperative_data_info_df['infrastructure_sequence'] == specified_sequence_id]


    with open(f'{source_path}/infrastructure-side/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    infrastructure_data_info_df = pd.DataFrame(data_info_data)

    infrastructure_filtered_df = infrastructure_data_info_df[infrastructure_data_info_df['sequence_id'] == specified_sequence_id]
    infrastructure_filtered_df = infrastructure_filtered_df.loc[infrastructure_filtered_df['frame_id'].isin(cooperative_filtered_df['infrastructure_frame'])]


    with open(f'{source_path}/vehicle-side/data_info.json', 'r') as file:
        data_info_data = json.load(file)
    vehicle_data_info_df = pd.DataFrame(data_info_data)
    vehicle_filtered_df = vehicle_data_info_df[vehicle_data_info_df['sequence_id'] == specified_sequence_id]
    vehicle_filtered_df = vehicle_filtered_df.loc[vehicle_filtered_df['frame_id'].isin(cooperative_filtered_df['vehicle_frame'])]
    infrastructure_calib_camera_intrinsic_paths = infrastructure_filtered_df['calib_camera_intrinsic_path'].tolist()
    vehicle_calib_camera_intrinsic_paths = vehicle_filtered_df['calib_camera_intrinsic_path'].tolist()

    car_list_infrastructure = [os.path.splitext(os.path.basename(path))[0] for path in infrastructure_calib_camera_intrinsic_paths]
    car_list_vehicle = [os.path.splitext(os.path.basename(path))[0] for path in vehicle_calib_camera_intrinsic_paths]

    camera_lidar2camera = [[], []]
    car_list = car_list_vehicle

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        # Read infrastructure cam info
        infrastructure_camera_intrinsics_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'camera_intrinsic',car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_camera_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_camera',car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_world_json = read_json_file(os.path.join(source_path,'infrastructure-side', 'calib', 'virtuallidar_to_world',car_list_infrastructure[idx]+'.json'))

        # Read vehicle cam info
        vehicle_camera_intrinsics_json = read_json_file(os.path.join(source_path,'vehicle-side', 'calib', 'camera_intrinsic', car_list_vehicle[idx] +'.json'))
        vehicle_lidar_to_camera_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_camera', car_list_vehicle[idx] +'.json'))
        vehicle_lidar_to_novatel_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'lidar_to_novatel', car_list_vehicle[idx] +'.json')) 
        vehicle_novatel_to_world_json = read_json_file(os.path.join(source_path, 'vehicle-side','calib', 'novatel_to_world', car_list_vehicle[idx] +'.json')) 

        Ks = np.array([get_intrinsics(infrastructure_camera_intrinsics_json), get_intrinsics(vehicle_camera_intrinsics_json)]) 
        lidar2cam = np.array([get_extrinsics(infrastructure_lidar_to_camera_json),get_extrinsics(vehicle_lidar_to_camera_json)])
        lidar2world = np.array([get_extrinsics(infrastructure_lidar_to_world_json), get_extrinsics_compute_lidar_to_world(vehicle_lidar_to_novatel_json, vehicle_novatel_to_world_json)])
        
        novatel_to_world = get_extrinsics(vehicle_novatel_to_world_json)
        
        novatel_to_world_padded = pad_poses(novatel_to_world)

        lidar2cam_padded = pad_poses(lidar2cam)
        lidar2world_padded = pad_poses(lidar2world)
        cam2lidar = np.linalg.inv(lidar2cam_padded)

        c2w = lidar2world_padded @ cam2lidar
        w2c = np.linalg.inv(c2w)

        cam_road2cam_car = w2c[0] @ c2w[1]
        cam_road2lidar_car = cam2lidar[0] @ cam_road2cam_car
        lidar_car2cam_road = np.linalg.inv(cam_road2lidar_car)
        
        camera_lidar2camera[0].append(lidar2cam_padded[1])
        camera_lidar2camera[1].append(lidar_car2cam_road)

    destination_folder = f"{des_path}/{specified_sequence_id}_0_original"
    generate_lidar_depth_seperate(destination_folder, camera_lidar2camera)




if __name__ == "__main__":
    sequence_list = sequence_list[3:]
    for i in tqdm(sequence_list):
        process(i)