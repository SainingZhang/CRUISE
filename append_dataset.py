import os
from tqdm import tqdm
import json
import shutil

def get_max_frame_id(directory, file_extension):
    max_frame_id = 0
    file_list = [f for f in os.listdir(
        directory) if f.endswith(file_extension)]

    # Extract the numeric part from the filename and update the maximum frame_id.
    for filename in file_list:
        frame_id = int(os.path.splitext(filename)[0])  
        if frame_id > max_frame_id:
            max_frame_id = frame_id

    return max_frame_id


source_path = "paht/to/V2X-Seq-SPD"
generate_path = "path/to/V2X-Seq-SPD-aug"

# Assume that the synthesized data numbering starts from 500000.
# start_frame_id = 500000  #! The first time append manually specify an id that won't be confused, after that just use the automatic method below.
max_frame_id = get_max_frame_id(os.path.join(
    generate_path, 'cooperative', 'label'), ".json")
# max_frame_id = get_max_frame_id(os.path.join(generate_path, 'vehicle-side','label','camera'), ".json")
start_frame_id = max_frame_id + 1


# Image path (this needs to be modified)
new_dataset_path = "path/to/result/of/your/generate_dataset.py"

# New sequence ID (can be modified as needed)
new_sequence_id = os.path.basename(new_dataset_path)
start_frame_id
new_sequence_id

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def copy_and_rename_files(source_dir, target_dir, file_extension, start_id):
    os.makedirs(target_dir, exist_ok=True)
    file_list = sorted([f for f in os.listdir(source_dir)
                       if f.endswith(file_extension)])
    for idx, filename in enumerate(tqdm(file_list, desc=f"Processing {file_extension} files")):
        new_filename = f"{str(start_id + idx).zfill(6)}{file_extension}"
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, new_filename)
        shutil.copy2(source_file, target_file)
    return len(file_list)


def update_data_info_car(data_info_file, new_sequence_id, num_images, start_frame_id):
    data_info = read_json_file(data_info_file)
    for idx in range(num_images):
        frame_id = str(start_frame_id + idx).zfill(6)
        entry = {
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
            "start_frame_id": frame_id,
            "end_frame_id": str(start_frame_id + num_images - 1).zfill(6),
            "num_frames": num_images,
            "sequence_id": new_sequence_id
        }
        data_info.append(entry)

    with open(data_info_file, 'w') as f:
        json.dump(data_info, f, indent=4)


def update_data_info_road(data_info_file, new_sequence_id, num_images, start_frame_id):
    data_info = read_json_file(data_info_file)
    for idx in range(num_images):
        frame_id = str(start_frame_id + idx).zfill(6)
        entry = {
            "image_path": f"image/{frame_id}.jpg",
            "pointcloud_path": f"velodyne/{frame_id}.pcd",
            "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{frame_id}.json",
            "calib_virtuallidar_to_camera_path": f"calib/virtuallidar_to_camera/{frame_id}.json",
            "calib_virtuallidar_to_world_path": f"calib/virtuallidar_to_world/{frame_id}.json",
            "label_camera_std_path": f"label/camera/{frame_id}.json",
            "label_lidar_std_path": f"label/virtuallidar/{frame_id}.json",
            "intersection_loc": "yizhuang10",
            "image_timestamp": str(1626167047000000 + idx * 100000),
            "pointcloud_timestamp": str(1626167046900000 + idx * 100000),
            "frame_id": frame_id,
            "start_frame_id": frame_id,
            "end_frame_id": str(start_frame_id + num_images - 1).zfill(6),
            "num_frames": num_images,
            "sequence_id": new_sequence_id
        }
        data_info.append(entry)

    with open(data_info_file, 'w') as f:
        json.dump(data_info, f, indent=4)


# Obtain the original data_info.json file.
vehicle_data_info_file = os.path.join(
    generate_path, 'vehicle-side', 'data_info.json')
infrastructure_data_info_file = os.path.join(
    generate_path, 'infrastructure-side', 'data_info.json')

vehicle_image_dir = os.path.join(new_dataset_path, 'vehicle-side', 'image')
infrastructure_image_dir = os.path.join(
    new_dataset_path, 'infrastructure-side', 'image')

vehicle_image_target = os.path.join(generate_path, 'vehicle-side', 'image')
infrastructure_image_target = os.path.join(
    generate_path, 'infrastructure-side', 'image')

# Point cloud path
vehicle_lidar_dir = os.path.join(new_dataset_path, 'vehicle-side', 'velodyne')
infrastructure_lidar_dir = os.path.join(
    new_dataset_path, 'infrastructure-side', 'velodyne')

vehicle_lidar_target = os.path.join(generate_path, 'vehicle-side', 'velodyne')
infrastructure_lidar_target = os.path.join(
    generate_path, 'infrastructure-side', 'velodyne')

# Annotation file path
vehicle_label_cam_dir = os.path.join(
    new_dataset_path, 'vehicle-side', 'label', 'camera')
infrastructure_label_cam_dir = os.path.join(
    new_dataset_path, 'infrastructure-side', 'label', 'camera')

vehicle_label_lidar_dir = os.path.join(
    new_dataset_path, 'vehicle-side', 'label', 'lidar')
infrastructure_label_lidar_dir = os.path.join(
    new_dataset_path, 'infrastructure-side', 'label', 'virtuallidar')

# Copy the image and renumber it.
num_images = copy_and_rename_files(
    vehicle_image_dir, vehicle_image_target, '.jpg', start_frame_id)
copy_and_rename_files(infrastructure_image_dir,
                      infrastructure_image_target, '.jpg', start_frame_id)

# Copy point cloud and renumber.
copy_and_rename_files(
    vehicle_lidar_dir, vehicle_lidar_target, '.pcd', start_frame_id)
copy_and_rename_files(infrastructure_lidar_dir,
                      infrastructure_lidar_target, '.pcd', start_frame_id)

# Copy the annotation file and renumber it.
copy_and_rename_files(vehicle_label_cam_dir, os.path.join(
    generate_path, 'vehicle-side', 'label', 'camera'), '.json', start_frame_id)
copy_and_rename_files(infrastructure_label_cam_dir, os.path.join(
    generate_path, 'infrastructure-side', 'label', 'camera'), '.json', start_frame_id)
copy_and_rename_files(vehicle_label_lidar_dir, os.path.join(
    generate_path, 'vehicle-side', 'label', 'lidar'), '.json', start_frame_id)
copy_and_rename_files(infrastructure_label_lidar_dir, os.path.join(
    generate_path, 'infrastructure-side', 'label', 'virtuallidar'), '.json', start_frame_id)

# Read infrastructure and vehicle calibration info
infrastructure_camera_intrinsics_json = os.path.join(
    new_dataset_path, 'infrastructure-side', 'calib', 'camera_intrinsic')
infrastructure_lidar_to_camera_json = os.path.join(
    new_dataset_path, 'infrastructure-side', 'calib', 'virtuallidar_to_camera')
infrastructure_lidar_to_world_json = os.path.join(
    new_dataset_path, 'infrastructure-side', 'calib', 'virtuallidar_to_world')

# Read vehicle cam info
vehicle_camera_intrinsics_json = os.path.join(
    new_dataset_path, 'vehicle-side', 'calib', 'camera_intrinsic')
vehicle_lidar_to_camera_json = os.path.join(
    new_dataset_path, 'vehicle-side', 'calib', 'lidar_to_camera')
vehicle_lidar_to_novatel_json = os.path.join(
    new_dataset_path, 'vehicle-side', 'calib', 'lidar_to_novatel')
vehicle_novatel_to_world_json = os.path.join(
    new_dataset_path, 'vehicle-side', 'calib', 'novatel_to_world')

# Copy the annotation file and renumber it.
copy_and_rename_files(infrastructure_camera_intrinsics_json, os.path.join(
    generate_path, 'infrastructure-side', 'calib', 'camera_intrinsic'), '.json', start_frame_id)
copy_and_rename_files(infrastructure_lidar_to_camera_json, os.path.join(
    generate_path, 'infrastructure-side', 'calib', 'virtuallidar_to_camera'), '.json', start_frame_id)
copy_and_rename_files(infrastructure_lidar_to_world_json, os.path.join(
    generate_path, 'infrastructure-side', 'calib', 'virtuallidar_to_world'), '.json', start_frame_id)

copy_and_rename_files(vehicle_camera_intrinsics_json, os.path.join(
    generate_path, 'vehicle-side', 'calib', 'camera_intrinsic'), '.json', start_frame_id)
copy_and_rename_files(vehicle_lidar_to_camera_json, os.path.join(
    generate_path, 'vehicle-side', 'calib', 'lidar_to_camera'), '.json', start_frame_id)
copy_and_rename_files(vehicle_lidar_to_novatel_json, os.path.join(
    generate_path, 'vehicle-side', 'calib', 'lidar_to_novatel'), '.json', start_frame_id)
copy_and_rename_files(vehicle_novatel_to_world_json, os.path.join(
    generate_path, 'vehicle-side', 'calib', 'novatel_to_world'), '.json', start_frame_id)


def modify_and_copy_data_info(source_path, output_path, start_vehicle_frame=500000, start_infrastructure_frame=500000, new_sequence_id="generate_tmp"):
    """
    Copy the original cooperative data_info.json file, modify the frame_id, and append it to the target file.
    """
    
    with open(source_path, 'r') as f:
        data_info = json.load(f)

    vehicle_frame_counter = start_vehicle_frame
    infrastructure_frame_counter = start_infrastructure_frame

    modified_data_info = []

    for entry in tqdm(data_info, desc="Modifying data info"):
        new_entry = entry.copy()

        new_entry['vehicle_frame'] = f"{vehicle_frame_counter:06d}"
        new_entry['infrastructure_frame'] = f"{infrastructure_frame_counter:06d}"
        new_entry['vehicle_sequence'] = new_sequence_id
        new_entry['infrastructure_sequence'] = new_sequence_id

        vehicle_frame_counter += 1
        infrastructure_frame_counter += 1

        modified_data_info.append(new_entry)

    with open(output_path, 'r') as f:
        existing_data = json.load(f)
        print(existing_data)
    
    existing_data.extend(modified_data_info)

    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Modified data_info.json has been appended to {output_path}")


source_data_info_path = os.path.join(
    new_dataset_path, 'cooperative', 'data_info.json')  
output_data_info_path = os.path.join(
    generate_path, 'cooperative', 'data_info.json')  
os.makedirs(os.path.dirname(output_data_info_path), exist_ok=True)

modify_and_copy_data_info(source_data_info_path, output_data_info_path, start_vehicle_frame=start_frame_id,
                          start_infrastructure_frame=start_frame_id, new_sequence_id=new_sequence_id)

cooperative_label_json = os.path.join(new_dataset_path, 'cooperative', 'label')


def copy_and_rename_files_cooperative(source_dir, target_dir, file_extension, start_id):
    os.makedirs(target_dir, exist_ok=True)
    file_list = sorted([f for f in os.listdir(source_dir)
                       if f.endswith(file_extension)])

    for idx, filename in enumerate(tqdm(file_list, desc=f"Processing {file_extension} files")):
        new_filename = f"{str(start_id + idx).zfill(6)}{file_extension}"
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, new_filename)

        if file_extension == '.json':
            modify_json_file(source_file, target_file, start_id + idx)
        else:
            shutil.copy2(source_file, target_file)

    return len(file_list)


def modify_json_file(source_file, target_file, new_frame_id):
    with open(source_file, 'r') as f:
        data = json.load(f)

    new_frame_id_str = str(new_frame_id).zfill(6)

    for entry in data:
        entry['veh_frame_id'] = new_frame_id_str
        entry['inf_frame_id'] = new_frame_id_str

    with open(target_file, 'w') as f:
        json.dump(data, f, indent=4)


copy_and_rename_files_cooperative(cooperative_label_json, os.path.join(
    generate_path, 'cooperative', 'label'), '.json', start_frame_id)

update_data_info_car(vehicle_data_info_file,
                     new_sequence_id, num_images, start_frame_id)
update_data_info_road(infrastructure_data_info_file,
                      new_sequence_id, num_images, start_frame_id)
print("Data appending completed.")

