
def main():
    import json
    import numpy as np
    import os
    from tqdm import tqdm
    import pandas as pd

    sequence_list = [
    '0000',
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
    '0018',
    '0020',
    '0021',
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
    # '0088',
    '0089',
    '0092',
    '0093',
    '0094']
    
    sequence_list = sequence_list[60:70]

    for sequence_id in sequence_list:

        specified_sequence_id = sequence_id  # Set the sequence_id you want to process.
        source_path = "/mnt/zhangsn/data/V2X-Seq-SPD"  # Original address of the dataset
        des_path = "/mnt/zhangsn/data/V2X-Seq-SPD-Processed"

        # output_dir = "/mnt/zhangsn/data/V2X-Seq-SPD-Processed/0017_0_original_all_cooperative_with_cooperative_pointcloud"
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
            
        with open(f"{source_path}/cooperative/data_info.json", 'r') as file:
            data_info_data = json.load(file)
            
        cooperative_data_info_df = pd.DataFrame(data_info_data)
        cooperative_filtered_df = cooperative_data_info_df[cooperative_data_info_df['infrastructure_sequence'] == specified_sequence_id]
        
        with open(f"{source_path}/infrastructure-side/data_info.json", 'r') as file:
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
        lidar2worlds = []

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
            lidar2worlds.append(lidar2world_padded)
            
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

        def delete_file(file_path):
            try:
                # 检查文件是否存在
                if os.path.exists(file_path):
                    # 删除文件
                    os.remove(file_path)
                    print(f"文件 '{file_path}' 已成功删除")
                else:
                    print(f"文件 '{file_path}' 不存在")
            except Exception as e:
                print(f"删除文件时发生错误: {str(e)}")
                
                
        import shutil

        def delete_directory(dir_path):
            try:
                if os.path.exists(dir_path):
                    # 删除目录及其所有内容
                    shutil.rmtree(dir_path)
                    print(f"目录 '{dir_path}' 已成功删除")
                else:
                    print(f"目录 '{dir_path}' 不存在")
            except Exception as e:
                print(f"删除目录时发生错误: {str(e)}")

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
            point_xyz_world_infrastructure = (np.pad(point_xyz_infrastructure, ((0, 0), (0, 1)), constant_values=1) @ lidar2worlds[idx][0].T)[:, :3]
            # point_xyz_world_infrastructure = (np.pad(point_xyz_infrastructure, ((0, 0), (0, 1)), constant_values=1) @ lidar2world_padded[0].T)[:, :3]

            pcd_vehicle = o3d.io.read_point_cloud(os.path.join(source_path,'vehicle-side' , "velodyne", car_list_vehicle[idx] + ".pcd"))
            point_vehicle = np.asarray(pcd_vehicle.points) 
            intensities_vehicle = np.zeros((point_vehicle.shape[0], 1))
            elongation_vehicle = np.zeros((point_vehicle.shape[0], 1))
            timestamp_pts_vehicle = np.zeros((point_vehicle.shape[0], 1))
            point_data_vehicle = np.hstack((point_vehicle, intensities_vehicle, elongation_vehicle, timestamp_pts_vehicle))
            point_xyz_vehicle, intensities_vehicle, elongation_vehicle, timestamp_pts_vehicle = np.split(point_data_vehicle, [3, 4, 5], axis=1)
            point_xyz_world_vehicle = (np.pad(point_xyz_vehicle, ((0, 0), (0, 1)), constant_values=1) @ lidar2worlds[idx][1].T)[:, :3]
            # point_xyz_world_vehicle = (np.pad(point_xyz_vehicle, ((0, 0), (0, 1)), constant_values=1) @ lidar2world_padded[1].T)[:, :3]
            
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
                    
                    pts_3d_homo = points_homogeneous @ camera_lidar2camera[frame_idx][cam_name].T
                    pts_3d_camera = pts_3d_homo[:, :3] / pts_3d_homo[:, 3].reshape(-1, 1)  # 归一化 
                    
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
        # output_dir = output_dir
        print(output_dir)

        delete_file(output_dir + '/pointcloud_cooperative.npz')
        process_lidar_data( camera_intrinsics, camera_lidar2camera, output_dir)

        ## Single-side pointcloud
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
            point_xyz_world_infrastructure = (np.pad(point_xyz_infrastructure, ((0, 0), (0, 1)), constant_values=1) @ lidar2worlds[idx][0].T)[:, :3]

            pcd_vehicle = o3d.io.read_point_cloud(os.path.join(source_path,'vehicle-side' , "velodyne", car_list_vehicle[idx] + ".pcd"))
            point_vehicle = np.asarray(pcd_vehicle.points) 
            intensities_vehicle = np.zeros((point_vehicle.shape[0], 1))
            elongation_vehicle = np.zeros((point_vehicle.shape[0], 1))
            timestamp_pts_vehicle = np.zeros((point_vehicle.shape[0], 1))
            point_data_vehicle = np.hstack((point_vehicle, intensities_vehicle, elongation_vehicle, timestamp_pts_vehicle))
            point_xyz_vehicle, intensities_vehicle, elongation_vehicle, timestamp_pts_vehicle = np.split(point_data_vehicle, [3, 4, 5], axis=1)
            point_xyz_world_vehicle = (np.pad(point_xyz_vehicle, ((0, 0), (0, 1)), constant_values=1) @ lidar2worlds[idx][1].T)[:, :3]
            
            point_xyz_world_combined = np.vstack((point_xyz_world_infrastructure))
            
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
                    # import pdb; pdb.set_trace()
                    pts_3d_homo = points_homogeneous @ camera_lidar2camera[frame_idx][cam_name].T
                    pts_3d_camera = pts_3d_homo[:, :3] / pts_3d_homo[:, 3].reshape(-1, 1)  # 归一化 
                    camera_projection, valid_mask = project_to_camera(pts_3d_camera, intrinsic, cam_idx)
                    
                    pts_3d_frame.append(pts_3d_lidar[valid_mask]) 
                    pts_2d_frame.append(camera_projection)
                
                if pts_3d_frame and pts_2d_frame:
                    pts_3d_all[frame_idx] = np.concatenate(pts_3d_frame, axis=0)
                    pts_2d_all[frame_idx] = np.concatenate(pts_2d_frame, axis=0)
            
            np.savez_compressed(os.path.join(output_dir, 'pointcloud_inf.npz'), 
                                pointcloud=pts_3d_all, 
                                camera_projection=pts_2d_all)
            print("Processing LiDAR data done...")

        delete_file(output_dir + '/pointcloud.npz')
        # output_dir = f"/mnt/zhangsn/data/V2X-Seq-SPD-Processed/0022_0_original_all_multi_lidar"

        process_lidar_data( camera_intrinsics, camera_lidar2camera, output_dir)

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
                    pts_3d_homo = points_homogeneous @ camera_lidar2camera[frame_idx][cam_name].T
                    pts_3d_camera = pts_3d_homo[:, :3] / pts_3d_homo[:, 3].reshape(-1, 1)  # 归一化 
                    camera_projection, valid_mask = project_to_camera(pts_3d_camera, intrinsic, cam_idx)
                    
                    pts_3d_frame.append(pts_3d_lidar[valid_mask]) 
                    pts_2d_frame.append(camera_projection)
                
                if pts_3d_frame and pts_2d_frame: 
                    pts_3d_all[frame_idx] = np.concatenate(pts_3d_frame, axis=0)
                    pts_2d_all[frame_idx] = np.concatenate(pts_2d_frame, axis=0)
            
            np.savez_compressed(os.path.join(output_dir, 'pointcloud_veh.npz'), 
                                pointcloud=pts_3d_all, 
                                camera_projection=pts_2d_all)
            print("Processing LiDAR data done...")


        seq_dir = f"{source_path}/vehicle-side/velodyne"
        # output_dir = f"{des_path}/{specified_sequence_id}_0_original"
        # output_dir = output_dir

        process_lidar_data(seq_dir, camera_intrinsics, camera_lidar2camera, output_dir)

        # generate lidar depth
        delete_directory(output_dir + '/lidar_depth')



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
            save_dir = os.path.join(datadir, 'lidar_depth_all_seperate')
            os.makedirs(save_dir, exist_ok=True)
            
            image_dir = os.path.join(datadir, 'images')
            image_files = glob(image_dir + "/*.jpg") 
            image_files += glob(image_dir + "/*.png")
            image_files = sorted(image_files)
            

            pointcloud_path = os.path.join(datadir, 'pointcloud_veh.npz')
            pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
            pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  
            
            pointcloud_path_cooperative = os.path.join(datadir, 'pointcloud_cooperative.npz')
            pts3d_dict_coop = np.load(pointcloud_path_cooperative, allow_pickle=True)['pointcloud'].item()
            pts2d_dict_coop = np.load(pointcloud_path_cooperative, allow_pickle=True)['camera_projection'].item()  
            
            pointcloud_path_inf = os.path.join(datadir, 'pointcloud_inf.npz')
            pts3d_dict_inf = np.load(pointcloud_path_inf, allow_pickle=True)['pointcloud'].item()
            pts2d_dict_inf = np.load(pointcloud_path_inf, allow_pickle=True)['camera_projection'].item()  

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
                    
                    raw_3d = pts3d_dict_inf[frame]
                    raw_2d = pts2d_dict_inf[frame]
                        
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


        generate_lidar_depth_seperate(output_dir, camera_lidar2camera)

        import os

        def rename_file_or_directory(old_path, new_path):
            try:
                # 检查原始路径是否存在
                if os.path.exists(old_path):
                    # 执行重命名操作
                    os.rename(old_path, new_path)
                    print(f"已成功将 '{old_path}' 重命名为 '{new_path}'")
                else:
                    print(f"路径 '{old_path}' 不存在")
            except FileExistsError:
                print(f"错误：'{new_path}' 已经存在")
            except PermissionError:
                print("错误：没有权限执行此操作")
            except Exception as e:
                print(f"重命名时发生错误: {str(e)}")


        old_file = output_dir + "/lidar_depth_all_seperate"
        new_file = output_dir + "/lidar_depth"
        rename_file_or_directory(old_file, new_file)

        old_file = output_dir + "/pointcloud_cooperative.npz"
        new_file = output_dir + "/pointcloud.npz"
        rename_file_or_directory(old_file, new_file)


if __name__ == "__main__":
    main()