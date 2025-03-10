import json
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
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
specified_sequence_id = '0017'  # Set the sequence_id you want to process.
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

