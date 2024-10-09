from tqdm import tqdm
import numpy as np
import cv2
import shutil
import torch
import os
import pandas as pd
import json
specified_sequence_id = '0022'

sequence_name = f"{specified_sequence_id}_gen_0"
v2x_seq_spd_path = "path/to/your/V2X-Seq-SPD"

generate_path = f"path/to/your/generate/dataset/path/{specified_sequence_id}/{sequence_name}"

os.makedirs(generate_path, exist_ok=True)
save_dir = "path/to/your/render/output"

with torch.no_grad():            

        with open(f'{v2x_seq_spd_path}/cooperative/data_info.json', 'r') as file:
                data_info_data = json.load(file)
                cooperative_data_info_df = pd.DataFrame(data_info_data)
                cooperative_filtered_df = cooperative_data_info_df[cooperative_data_info_df['infrastructure_sequence'] == specified_sequence_id]


        with open(f'{v2x_seq_spd_path}/infrastructure-side/data_info.json', 'r') as file:
                data_info_data = json.load(file)
                infrastructure_data_info_df = pd.DataFrame(data_info_data)

                infrastructure_filtered_df = infrastructure_data_info_df[infrastructure_data_info_df['sequence_id'] == specified_sequence_id]
                infrastructure_filtered_df = infrastructure_filtered_df.loc[infrastructure_filtered_df['frame_id'].isin(cooperative_filtered_df['infrastructure_frame'])]


        with open(f'{v2x_seq_spd_path}/vehicle-side/data_info.json', 'r') as file:
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
        
        
generate_new_dataset = 1

if generate_new_dataset:
    target_dir_1 = os.path.join(generate_path, 'vehicle-side', 'image')
    target_dir_2 = os.path.join(generate_path, 'infrastructure-side', 'image')

    os.makedirs(target_dir_1, exist_ok=True)
    os.makedirs(target_dir_2, exist_ok=True)

    frame_list = []

    for filename in os.listdir(save_dir):
        if 'rgb.png' in filename:
            parts = filename.split('_')
            frame_id = parts[0] 

            if frame_id not in frame_list:
                frame_list.append(frame_id)

    frame_list.sort()

    for filename in os.listdir(save_dir):
        if 'rgb.png' in filename:
            parts = filename.split('_')
            frame_id = parts[0]  
            view_id = parts[1]  

            index = frame_list.index(frame_id)
            new_filename = f'{index:06}.png'  

            original_path = os.path.join(save_dir, filename)

            if view_id == '0':
                target_path = os.path.join(target_dir_1, new_filename)
                shutil.copy2(original_path, target_path)
            elif view_id == '1':
                target_path = os.path.join(target_dir_2, new_filename)
                shutil.copy2(original_path, target_path)

            print(f"Copied {filename} as {new_filename} to the target directory.")

    print("All files processed.")


target_size = (1920, 1080)

def process_images(image_dir):
    """
    Process the image; if the size is 1600x900, scale it to 1920x1080, convert it to jpg format for saving, and delete the original png image.
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is not None:
            h, w = image.shape[:2]
            if (w, h) == (1600, 900):  
                resized_image = cv2.resize(
                    image, target_size, interpolation=cv2.INTER_LINEAR)
                new_image_path = os.path.splitext(image_path)[0] + ".jpg"
                cv2.imwrite(new_image_path, resized_image, [
                            int(cv2.IMWRITE_JPEG_QUALITY), 95])
                os.remove(image_path)
            else:
                print(f"Skipping image {image_file}, size {w}x{h} not 1600x900")
        else:
            print(f"Failed to read image {image_file}")


image_dirs = [
    target_dir_1,
    target_dir_2
]

for image_dir in image_dirs:
    process_images(image_dir)


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

source_path = f"{v2x_seq_spd_path}"
car_list = car_list_vehicle

rename_idx = 0

exclude_list = []
exclude_list_formatted = []
exclude_list_formatted_car = []
exclude_list_formatted_road = []
for obj in exclude_list:
    number = obj.split('_')[1]
    formatted_number = str(int(number)).zfill(6)
    exclude_list_formatted.append(formatted_number)


os.makedirs(os.path.join(generate_path, 'cooperative', 'label'), exist_ok=True)
os.makedirs(os.path.join(generate_path, 'vehicle-side',
            'label', 'lidar'), exist_ok=True)
os.makedirs(os.path.join(generate_path, 'vehicle-side',
            'label', 'camera'), exist_ok=True)
os.makedirs(os.path.join(generate_path, 'infrastructure-side',
            'label', 'camera'), exist_ok=True)
os.makedirs(os.path.join(generate_path, 'infrastructure-side',
            'label', 'virtuallidar'), exist_ok=True)

for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):

    if str(int(idx)).zfill(6) not in frame_list:
        continue
    else:
        coop_annotations = read_json_file(os.path.join(
            source_path, 'cooperative', 'label', car_list_vehicle[idx] + '.json'))
        car_annotations_cam = read_json_file(os.path.join(
            source_path, 'vehicle-side', 'label', 'camera', car_list_vehicle[idx] + '.json'))
        print(os.path.join(source_path, 'infrastructure-side', 'label',
              'camera', car_list_infrastructure[idx] + '.json'))
        road_annotations_cam = read_json_file(os.path.join(
            source_path, 'infrastructure-side', 'label', 'camera', car_list_infrastructure[idx] + '.json'))

        car_annotations_lidar = read_json_file(os.path.join(
            source_path, 'vehicle-side', 'label', 'lidar', car_list_vehicle[idx] + '.json'))
        road_annotations_lidar = read_json_file(os.path.join(
            source_path, 'infrastructure-side', 'label', 'virtuallidar', car_list_infrastructure[idx] + '.json'))

        new_annotations_car_cam = []
        new_annotations_road_cam = []
        new_annotations_car_lidar = []
        new_annotations_road_lidar = []
        new_annotations_coop = []

        for ann in coop_annotations:

            if ann['track_id'] not in exclude_list_formatted:
                ann['veh_frame_id'] = str(int(rename_idx)).zfill(6)
                ann['inf_frame_id'] = str(int(rename_idx)).zfill(6)

                new_annotations_coop.append(ann)

            elif ann['track_id'] in exclude_list_formatted:
                exclude_list_formatted_car.append(ann["veh_track_id"])
                exclude_list_formatted_road.append(ann["inf_track_id"])

        for ann in car_annotations_cam:
            if ann['track_id'] not in exclude_list_formatted_car:
                new_annotations_car_cam.append(ann)

        for ann in road_annotations_cam:
            if ann['track_id'] not in exclude_list_formatted_road:
                new_annotations_road_cam.append(ann)

        for ann in car_annotations_lidar:
            if ann['track_id'] not in exclude_list_formatted_car:
                new_annotations_car_lidar.append(ann)

        for ann in road_annotations_lidar:
            if ann['track_id'] not in exclude_list_formatted_road:
                new_annotations_road_lidar.append(ann)

        coop_annotation_path = os.path.join(
            generate_path, 'cooperative', 'label',  str(int(rename_idx)).zfill(6) + '.json')

        vehicle_annotation_path_cam = os.path.join(
            generate_path, 'vehicle-side', 'label', 'camera',  str(int(rename_idx)).zfill(6) + '.json')

        infrastructure_annotation_path_cam = os.path.join(
            generate_path, 'infrastructure-side', 'label', 'camera', str(int(rename_idx)).zfill(6) + '.json')

        vehicle_annotation_path_lidar = os.path.join(
            generate_path, 'vehicle-side', 'label', 'lidar',  str(int(rename_idx)).zfill(6) + '.json')

        infrastructure_annotation_path_lidar = os.path.join(
            generate_path, 'infrastructure-side', 'label', 'virtuallidar', str(int(rename_idx)).zfill(6) + '.json')

        with open(coop_annotation_path, 'w') as f:
            json.dump(new_annotations_coop, f, indent=4)
        with open(vehicle_annotation_path_cam, 'w') as f:
            json.dump(new_annotations_car_cam, f, indent=4)
        with open(infrastructure_annotation_path_cam, 'w') as f:
            json.dump(new_annotations_road_cam, f, indent=4)
        with open(vehicle_annotation_path_lidar, 'w') as f:
            json.dump(new_annotations_car_lidar, f, indent=4)
        with open(infrastructure_annotation_path_lidar, 'w') as f:
            json.dump(new_annotations_road_lidar, f, indent=4)

        rename_idx += 1


rename_idx = 0

target_calib_path_car = generate_path + '/vehicle-side/calib'
target_calib_path_road = generate_path + '/infrastructure-side/calib'

os.makedirs(os.path.join(target_calib_path_car,
            "camera_intrinsic"), exist_ok=True)
os.makedirs(os.path.join(target_calib_path_car,
            "lidar_to_camera"), exist_ok=True)
os.makedirs(os.path.join(target_calib_path_car,
            "lidar_to_novatel"), exist_ok=True)
os.makedirs(os.path.join(target_calib_path_car,
            "novatel_to_world"), exist_ok=True)

os.makedirs(os.path.join(target_calib_path_road,
            "camera_intrinsic"), exist_ok=True)
os.makedirs(os.path.join(target_calib_path_road,
            "virtuallidar_to_camera"), exist_ok=True)
os.makedirs(os.path.join(target_calib_path_road,
            "virtuallidar_to_world"), exist_ok=True)


for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
    if str(int(idx)).zfill(6) not in frame_list:
        continue
    else:
        # Read infrastructure and vehicle calibration info
        infrastructure_camera_intrinsics_json = read_json_file(os.path.join(
            source_path, 'infrastructure-side', 'calib', 'camera_intrinsic', car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_camera_json = read_json_file(os.path.join(
            source_path, 'infrastructure-side', 'calib', 'virtuallidar_to_camera', car_list_infrastructure[idx]+'.json'))
        infrastructure_lidar_to_world_json = read_json_file(os.path.join(
            source_path, 'infrastructure-side', 'calib', 'virtuallidar_to_world', car_list_infrastructure[idx]+'.json'))

        # Read vehicle cam info
        vehicle_camera_intrinsics_json = read_json_file(os.path.join(
            source_path, 'vehicle-side', 'calib', 'camera_intrinsic', car_list_vehicle[idx] + '.json'))
        vehicle_lidar_to_camera_json = read_json_file(os.path.join(
            source_path, 'vehicle-side', 'calib', 'lidar_to_camera', car_list_vehicle[idx] + '.json'))
        vehicle_lidar_to_novatel_json = read_json_file(os.path.join(
            source_path, 'vehicle-side', 'calib', 'lidar_to_novatel', car_list_vehicle[idx] + '.json'))  
        vehicle_novatel_to_world_json = read_json_file(os.path.join(
            source_path, 'vehicle-side', 'calib', 'novatel_to_world', car_list_vehicle[idx] + '.json'))  

        image_timestamp = vehicle_filtered_df.loc[idx, 'image_timestamp']

        json_file_car = f"{str(int(rename_idx)).zfill(6)}.json"
        json_file_road = f"{str(int(rename_idx)).zfill(6)}.json"

        target_calib_path_car = generate_path + '/vehicle-side/calib'
        target_calib_path_road = generate_path + '/infrastructure-side/calib'

        with open(os.path.join(target_calib_path_car, "camera_intrinsic", json_file_car), 'w') as f:
            json.dump(vehicle_camera_intrinsics_json, f)
        with open(os.path.join(target_calib_path_car, 'lidar_to_camera', json_file_car), 'w') as f:
            json.dump(vehicle_lidar_to_camera_json, f)
        with open(os.path.join(target_calib_path_car, 'lidar_to_novatel', json_file_car), 'w') as f:
            json.dump(vehicle_lidar_to_novatel_json, f)
        with open(os.path.join(target_calib_path_car, 'novatel_to_world', json_file_car), 'w') as f:
            json.dump(vehicle_novatel_to_world_json, f)

        with open(os.path.join(target_calib_path_road, "camera_intrinsic", json_file_road), 'w') as f:
            json.dump(infrastructure_camera_intrinsics_json, f)
        with open(os.path.join(target_calib_path_road, 'virtuallidar_to_camera', json_file_road), 'w') as f:
            json.dump(infrastructure_lidar_to_camera_json, f)
        with open(os.path.join(target_calib_path_road, 'virtuallidar_to_world', json_file_road), 'w') as f:
            json.dump(infrastructure_lidar_to_world_json, f)

        rename_idx += 1


def create_data_info(frame_list, output_path_car, output_path_road):
    data_info_car = []
    data_info_road = []
    for idx, car_id in tqdm(enumerate(frame_list), desc="Creating data info"):
        frame_id = f"{str(0 + idx).zfill(6)}"
        entry_car = {
            "image_path": f"image/{frame_id}.jpg",
            "pointcloud_path": f"velodyne/{frame_id}.pcd",
            "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{frame_id}.json",
            "calib_lidar_to_camera_path": f"calib/lidar_to_camera/{frame_id}.json",
            "calib_lidar_to_novatel_path": f"calib/lidar_to_novatel/{frame_id}.json",
            "calib_novatel_to_world_path": f"calib/novatel_to_world/{frame_id}.json",
            "label_camera_std_path": f"label/camera/{frame_id}.json",
            "label_lidar_std_path": f"label/lidar/{frame_id}.json",
            "intersection_loc": "yizhuang10",
            "image_timestamp": str(1626167047000000 + idx * 100000),
            "pointcloud_timestamp": str(1626167046900000 + idx * 100000),
            "frame_id": frame_id,
            "start_frame_id": f"{str(0 + idx).zfill(6)}",
            "end_frame_id": f"{str(0 + len(frame_list) - 1).zfill(6)}",
            "num_frames": len(frame_list),
            "sequence_id": f"{sequence_name}"
        }
        entry_road = {
            "image_path": f"image/{frame_id}.jpg",
            "pointcloud_path": f"velodyne/{frame_id}.pcd",
            "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{frame_id}.json",
            "calib_virtuallidar_to_camera_path": f"ccalib/virtuallidar_to_camera/{frame_id}.json",
            "calib_virtuallidar_to_world_path": f"calib/virtuallidar_to_world/{frame_id}.json",
            "label_camera_std_path": f"label/camera/{frame_id}.json",
            "label_lidar_std_path": f"label/lidar/{frame_id}.json",
            "intersection_loc": "yizhuang10",
            "image_timestamp": str(1626167047000000 + idx * 100000),
            "pointcloud_timestamp": str(1626167046900000 + idx * 100000),
            "frame_id": frame_id,
            "start_frame_id": f"{str(0 + idx).zfill(6)}",
            "end_frame_id": f"{str(0 + len(frame_list) - 1).zfill(6)}",
            "num_frames": len(frame_list),
            "sequence_id":  f"{sequence_name}"
        }
        data_info_car.append(entry_car)
        data_info_road.append(entry_road)

    with open(output_path_car, 'w') as f:
        json.dump(data_info_car, f, indent=4)
    with open(output_path_road, 'w') as f:
        json.dump(data_info_car, f, indent=4)


output_path_car = os.path.join(generate_path, 'vehicle-side', 'data_info.json')
output_path_road = os.path.join(
    generate_path, 'infrastructure-side', 'data_info.json')
create_data_info(frame_list, output_path_car, output_path_road)


rename_idx = 0


target_dir_1 = os.path.join(generate_path, 'vehicle-side', 'velodyne')
target_dir_2 = os.path.join(generate_path, 'infrastructure-side', 'velodyne')


os.makedirs(target_dir_1, exist_ok=True)
os.makedirs(target_dir_2, exist_ok=True)


for idx, car_id in tqdm(enumerate(car_list), desc="copying origingal lidar"):
    if str(int(idx)).zfill(6) not in frame_list:
        continue
    else:
        # Read infrastructure and vehicle calibration info
        infrastructure_camera_intrinsics_json = os.path.join(
            source_path, 'infrastructure-side', 'velodyne', car_list_infrastructure[idx]+'.pcd')

        # Read vehicle cam info
        vehicle_camera_intrinsics_json = os.path.join(
            source_path, 'vehicle-side', 'velodyne',  car_list_vehicle[idx] + '.pcd')

        new_filename_car = f'{str(int(rename_idx)).zfill(6)}.pcd'

        target_path_car = os.path.join(target_dir_1, new_filename_car)

        shutil.copy2(infrastructure_camera_intrinsics_json, target_path_car)

        new_filename_road = f'{str(int(rename_idx)).zfill(6)}.pcd'

        target_path_road = os.path.join(target_dir_2, new_filename_road)

        shutil.copy2(vehicle_camera_intrinsics_json, target_path_road)

        rename_idx += 1

print("All files processed.")


def modify_and_copy_data_info(source_path, output_path, start_vehicle_frame=0, start_infrastructure_frame=0, car_list_vehicle=[]):
    """
    
    """
    with open(source_path, 'r') as f:
        data_info = json.load(f)

    vehicle_frame_counter = start_vehicle_frame
    infrastructure_frame_counter = start_infrastructure_frame

    modified_data_info = []

    for entry in tqdm(data_info, desc="Modifying data info"):
        if entry['vehicle_frame'] in car_list_vehicle[int(frame_list[0]):int(frame_list[-1])+1]:
            new_entry = entry.copy()

            
            new_entry['vehicle_frame'] = f"{vehicle_frame_counter:06d}"
            new_entry['infrastructure_frame'] = f"{infrastructure_frame_counter:06d}"
            
            new_entry['vehicle_sequence'] = f"{sequence_name}"
            
            new_entry['infrastructure_sequence'] = f"{sequence_name}"

            
            vehicle_frame_counter += 1
            infrastructure_frame_counter += 1

            modified_data_info.append(new_entry)

    
    with open(output_path, 'w') as f:
        json.dump(modified_data_info, f, indent=4)

    print(f"Modified data_info.json has been appended to {output_path}")



source_data_info_path = os.path.join(
    source_path, 'cooperative', 'data_info.json')  
output_data_info_path = os.path.join(
    generate_path, 'cooperative', 'data_info.json') 
os.makedirs(os.path.dirname(output_data_info_path), exist_ok=True)

modify_and_copy_data_info(source_data_info_path, output_data_info_path, start_vehicle_frame=0,
                          start_infrastructure_frame=0, car_list_vehicle=car_list_vehicle)
