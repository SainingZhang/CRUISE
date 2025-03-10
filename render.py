import random
import torch
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianisualizer
import time
import pandas as pd
import cv2
import numpy as np
from lib.utils.system_utils import searchForMaxIteration

def render_with_edit():
    safe_state(cfg.eval.quiet)
    cfg.mode

    cfg.render.save_image = True
    cfg.render.save_video = True
    generate_new_dataset = False
    generate_path = "generate/new_car"

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.render_actors = True
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        
    save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
        scene.loaded_iter), 'exp_{}'.format(time.strftime('%Y-%m-%d_%H-%M')))
    
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    # import pdb; pdb.set_trace()

    model_keys = gaussians.model_name_id.keys()
    model_keys_list = list(model_keys)
    model_keys_list.remove("background")


    random.seed(time.time())
    # random.seed(1)

    vehicle_models = gaussians.model_keys

    replacement_ratio = 1

    replacement_dict = {}
    for obj in model_keys_list:
        if random.random() < replacement_ratio:  
            replacement_dict[obj] = random.choice(vehicle_models)
    print(replacement_dict)
    
    # replacement_dict = {'obj_3770': 'Lamborghini', 'obj_3772': 'Lamborghini', 'obj_3780': 'Lamborghini', 'obj_3781': 'Lamborghini', 'obj_3798': 'Lamborghini', 'obj_3816': 'iveco-daily-l1h1-2017', 'obj_3817': 'opel-combo-cargo-ru-spec-l1-2021', 'obj_3825': 'white_big_car', 'obj_3826': 'jeep_relight_1', 'obj_3827': 'pickup_relight', 'obj_3828': 'nissan-nv-300-van-lwb-2021', 'obj_3831': 'Lamborghini', 'obj_3832': 'mercedes-benz-s-560-lang-amg-line-v222-2018.fbx', 'obj_3833': 'Lamborghini', 'obj_3834': 'peugeot-boxer-window-van-l1h1-2006-2014', 'obj_3836': 'Lamborghini', 'obj_3837': 'Lamborghini', 'obj_3843': 'Lamborghini'}

    # import pdb; pdb.set_trace()

    with torch.no_grad():
        if not cfg.eval.skip_train:
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            exclude_list = []
            # import pdb; pdb.set_trace()      
            # 读取jsonl文件并找到对应sequence的ego car tracking id
            json_path = "ego_car_tracking.json"
            
            import json

            # TODO 可能报错
            with open(json_path, 'r') as f:
                ego_tracking_dict = json.load(f)

            ego_car_tracking_id = ego_tracking_dict.get(dataset.sequence_id[0], None)  # 获取对应sequence的tracking id
            
            if ego_car_tracking_id is not None:
                ego_car_obj_name = 'obj_' + str(int(ego_car_tracking_id) )
                exclude_list.append(ego_car_obj_name)
            else:
                ego_car_obj_name = None
                print(f'找不到sequence {dataset.sequence_id[0]} 对应的ego car tracking id')
                
            print(gaussians.model_name_id.keys())
            for obj_name, ply_path in replacement_dict.items():
                if ego_car_obj_name:
                    if obj_name == ego_car_obj_name:
                        continue
                gaussians.replace_gaussian_with_custom_actor_new(ply_path, obj_name)
            

    # import pdb; pdb.set_trace()
    
    with torch.no_grad():
        if not cfg.eval.skip_train:
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                torch.cuda.synchronize()
                start_time = time.time()
                # import pdb; pdb.set_trace()
                result = renderer.render_edit(
                    camera, gaussians, exclude_list=exclude_list, scene_info = dataset.scene_info)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                # visualizer.visualize(result, camera)
                visualizer.visualize_combined(result, camera)
                visualizer.visualize_bbox(result, camera)
                
            i2v_xuhr(save_dir)
    
    # visualizer.summarize()
    
def render_with_edit_all():
    safe_state(cfg.eval.quiet)
    cfg.mode

    cfg.render.save_image = True
    cfg.render.save_video = True
    generate_new_dataset = False
    generate_path = "generate/new_car"

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.render_actors = True
        # gaussians.init_render_setup(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        
    save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
        scene.loaded_iter), 'exp_{}'.format(time.strftime('%Y-%m-%d_%H-%M')))
    
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    # import pdb; pdb.set_trace()

    model_keys = gaussians.model_name_id.keys()
    model_keys_list = list(model_keys)
    model_keys_list.remove("background")


    random.seed(time.time())
    # random.seed(1)

    vehicle_models = gaussians.model_keys

    replacement_ratio = 1

    replacement_dict = {}
    for obj in model_keys_list:
        if random.random() < replacement_ratio:  
            replacement_dict[obj] = random.choice(vehicle_models)
    print(replacement_dict)
    
    # replacement_dict = {'obj_3770': 'Lamborghini', 'obj_3772': 'Lamborghini', 'obj_3780': 'Lamborghini', 'obj_3781': 'Lamborghini', 'obj_3798': 'Lamborghini', 'obj_3816': 'iveco-daily-l1h1-2017', 'obj_3817': 'opel-combo-cargo-ru-spec-l1-2021', 'obj_3825': 'white_big_car', 'obj_3826': 'jeep_relight_1', 'obj_3827': 'pickup_relight', 'obj_3828': 'nissan-nv-300-van-lwb-2021', 'obj_3831': 'Lamborghini', 'obj_3832': 'mercedes-benz-s-560-lang-amg-line-v222-2018.fbx', 'obj_3833': 'Lamborghini', 'obj_3834': 'peugeot-boxer-window-van-l1h1-2006-2014', 'obj_3836': 'Lamborghini', 'obj_3837': 'Lamborghini', 'obj_3843': 'Lamborghini'}

    # import pdb; pdb.set_trace()

    with torch.no_grad():
        if not cfg.eval.skip_train:
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            exclude_list = []
            # import pdb; pdb.set_trace()      
            # 读取jsonl文件并找到对应sequence的ego car tracking id
            json_path = "ego_car_tracking.json"
            
            import json

            # TODO 可能报错
            with open(json_path, 'r') as f:
                ego_tracking_dict = json.load(f)
                
            not_render_bbox_list = []

            ego_car_tracking_id = ego_tracking_dict.get(dataset.sequence_id[0], None)  # 获取对应sequence的tracking id
            
            if ego_car_tracking_id is not None:
                if isinstance(ego_car_tracking_id, list):
                    ego_car_obj_names = []
                    for tracking_id in ego_car_tracking_id:
                        ego_car_obj_name = 'obj_' + str(int(tracking_id))
                        ego_car_obj_names.append(ego_car_obj_name)
                        exclude_list.append(ego_car_obj_name)
                        not_render_bbox_list.append(ego_car_obj_name)
                    ego_car_obj_name = ego_car_obj_names
                else:
                    ego_car_obj_name = 'obj_' + str(int(ego_car_tracking_id))
                    exclude_list.append(ego_car_obj_name)
                    not_render_bbox_list.append(ego_car_obj_name)
            else:
                ego_car_obj_name = None
                print(f'找不到sequence {dataset.sequence_id[0]} 对应的ego car tracking id')

            print(gaussians.model_name_id.keys())
            
            # 添加一个逻辑如果是这些车，就直接在这里删了（在 gs model 的 obj 里面）
            
            json_path = "not_replace_car.json"
            with open(json_path, 'r') as f:
                not_replace_car_dict = json.load(f)
                
            
        
            delete_tracking_id_list = not_replace_car_dict.get(dataset.sequence_id[0], None)  # 获取对应sequence的tracking id
            
            # 如果在这个delete_tracking_id_list里面就在 street_gaussian_model 的 attribute 里面删除
            if delete_tracking_id_list :
                for tracking_id in delete_tracking_id_list:
                    gaussians.remove_actor(f'obj_{str(int(tracking_id))}')
                    not_render_bbox_list.append(f'obj_{str(int(tracking_id))}')

            for obj_name, ply_path in replacement_dict.items():
                if ego_car_obj_name:
                    if obj_name == ego_car_obj_name:
                        continue
                try:
                    gaussians.replace_gaussian_with_custom_actor_new(ply_path, obj_name)
                except:
                    continue
            

    # import pdb; pdb.set_trace()
    
    with torch.no_grad():
        if not cfg.eval.skip_train:
            # 创建或清空标注文件
            # annotation_file = os.path.join(save_dir, 'annotations.txt')
            # with open(annotation_file, 'w') as f:
            #     f.write('')  # 清空文件
            modified_actors_file = os.path.join(save_dir, 'modified_actors.json')
            gaussians.save_modified_actors_info(modified_actors_file)
                
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render_edit_all(
                    camera, gaussians, exclude_list=exclude_list, scene_info = dataset.scene_info, not_render_bbox_list=not_render_bbox_list)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                visualizer.visualize_new(result, camera)
                visualizer.visualize_combined(result, camera)
                # import pdb; pdb.set_trace()
                visualizer.visualize_bbox(result, camera)
                
                # all_annotations = gaussians.export_all_annotations(idx, camera)
                # with open(annotation_file, 'a') as f:  # 使用'a'模式追加内容
                #     for ann in all_annotations:
                #         f.write(ann + '\n')
                
                
            i2v_xuhr(save_dir)
    
    # visualizer.summarize()
    
def render_specific_frame():
    safe_state(cfg.eval.quiet)
    cfg.mode

    cfg.render.save_image = True
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.render_actors = True
        # gaussians.init_render_setup(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        
    save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
        scene.loaded_iter), 'exp_{}'.format(time.strftime('%Y-%m-%d_%H-%M')))
    
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    # import pdb; pdb.set_trace()


    with torch.no_grad():
        if not cfg.eval.skip_train:
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            exclude_list = []
            # import pdb; pdb.set_trace()      
            # 读取jsonl文件并找到对应sequence的ego car tracking id

    
    with torch.no_grad():
        if not cfg.eval.skip_train:
                
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                # import pdb; pdb.set_trace()
                frame_id,cam_id = int(camera.image_name.split('_')[0]),int(camera.image_name.split('_')[1])
                if frame_id == 56:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    # import pdb; pdb.set_trace()
                    result = renderer.render_edit_all(
                        camera, gaussians, exclude_list=exclude_list, scene_info = dataset.scene_info)
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                    visualizer.visualize_new(result, camera)
                    visualizer.visualize_combined(result, camera)
                    visualizer.visualize_bbox(result, camera)
                    
    
    # visualizer.summarize()
    
def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []

        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
                scene.loaded_iter), 'test')
            
            os.makedirs(save_dir, exist_ok=True)
            
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            print(gaussians.model_name_id.keys())

            exclude_list = []
            
            # import pdb; pdb.set_trace()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):

                torch.cuda.synchronize()
                start_time = time.time()

                result = renderer.render(
                    camera, gaussians, exclude_list=exclude_list)

                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

                visualizer.visualize_combined(result, camera)
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(
                cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):

                torch.cuda.synchronize()
                start_time = time.time()

                result = renderer.render(camera, gaussians)

                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

                visualizer.visualize(result, camera)

        print(times)
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))

def render_all():
    cfg.render.save_image = True
    # cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        save_dir = os.path.join(
            cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianisualizer(save_dir)

        # import pdb; pdb.set_trace()
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)
            visualizer.visualize(result, camera)

        visualizer.summarize()

def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        save_dir = os.path.join(
            cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianisualizer(save_dir)

        import pdb; pdb.set_trace()
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        
        
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)
            visualizer.visualize(result, camera)

        visualizer.summarize()

def i2v_xuhr(save_dir):
    # Set up fps
    fps=24
    
    # Set input and output path
    input_path = os.path.join(save_dir, 'bbox')
    
    output_dir = os.path.join(save_dir,"video")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on input path
    output_path = os.path.join(output_dir,f"{cfg.model_path.split('/')[-2].split('_')[-1]}_replaced_car_with_bbox_combined.mp4")
    
    # Get the first images to determine dimensions
    img0 = cv2.imread(os.path.join(input_path, '000000_0_bbox.png'))
    img1 = cv2.imread(os.path.join(input_path, '000000_1_bbox.png'))
    
    if img0 is None or img1 is None:
        raise Exception("Could not read first images. Please check the path and file names.")
    
    # Get dimensions
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    # Create video writer
    # Combined width will be sum of individual widths, height will be max of heights
    combined_width = w0 + w1
    combined_height = max(h0, h1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    if not out.isOpened():
        raise Exception(f"Failed to create video writer for {output_path}")
    
    # Process each frame
    num_frames = len([f for f in os.listdir(input_path) if f.endswith('_0_bbox.png')])
    print(f"Processing {num_frames} frames...")
    
    for idx in tqdm(range(num_frames)):
        # Read both images
        img0_path = os.path.join(input_path, f'{idx:06d}_0_bbox.png')
        img1_path = os.path.join(input_path, f'{idx:06d}_1_bbox.png')
        
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        
        if img0 is None or img1 is None:
            print(f"Warning: Could not read images for index {idx}")
            continue
        
        # Resize images to match the height if necessary
        if h0 != combined_height:
            img0 = cv2.resize(img0, (int(w0 * combined_height / h0), combined_height))
        if h1 != combined_height:
            img1 = cv2.resize(img1, (int(w1 * combined_height / h1), combined_height))
        
        # Combine images side by side
        combined_img = np.hstack((img0, img1))
        
        # Write frame
        out.write(combined_img)
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_path}")

def i2v():
    # Set up fps
    fps=24
    
    # Set input and output path
    max_iter = searchForMaxIteration(os.path.join(cfg.model_path, "trained_model"))
    input_path = os.path.join(cfg.model_path, f'train/ours_{max_iter}/2-20-test_ego_car_replacement/bbox')
    output_dir = f'{cfg.model_path}'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on input path
    output_path = os.path.join(output_dir, f"{cfg.model_path.split('/')[-2].split('_')[-1]}_replaced_car_with_bbox_combined.mp4")
    
    # Get the first images to determine dimensions
    img0 = cv2.imread(os.path.join(input_path, '000000_0_bbox.png'))
    img1 = cv2.imread(os.path.join(input_path, '000000_1_bbox.png'))
    
    if img0 is None or img1 is None:
        raise Exception("Could not read first images. Please check the path and file names.")
    
    # Get dimensions
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    # Create video writer
    # Combined width will be sum of individual widths, height will be max of heights
    combined_width = w0 + w1
    combined_height = max(h0, h1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    if not out.isOpened():
        raise Exception(f"Failed to create video writer for {output_path}")
    
    # Process each frame
    num_frames = len([f for f in os.listdir(input_path) if f.endswith('_0_bbox.png')])
    print(f"Processing {num_frames} frames...")
    
    for idx in tqdm(range(num_frames)):
        # Read both images
        img0_path = os.path.join(input_path, f'{idx:06d}_0_bbox.png')
        img1_path = os.path.join(input_path, f'{idx:06d}_1_bbox.png')
        
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        
        if img0 is None or img1 is None:
            print(f"Warning: Could not read images for index {idx}")
            continue
        
        # Resize images to match the height if necessary
        if h0 != combined_height:
            img0 = cv2.resize(img0, (int(w0 * combined_height / h0), combined_height))
        if h1 != combined_height:
            img1 = cv2.resize(img1, (int(w1 * combined_height / h1), combined_height))
        
        # Combine images side by side
        combined_img = np.hstack((img0, img1))
        
        # Write frame
        out.write(combined_img)
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_path}")
    
def draw_bbox_sequence():
    safe_state(cfg.eval.quiet)
    cfg.mode

    cfg.render.save_image = True
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.render_actors = True
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
    save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
        scene.loaded_iter), 'draw_source_bbox')
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    model_keys = gaussians.model_name_id.keys()
    model_keys_list = list(model_keys)
    model_keys_list.remove("background")

    random.seed(time.time())

    with torch.no_grad():
        if not cfg.eval.skip_train:
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            
            for idx, camera in enumerate(tqdm(cameras, desc="Drawing Bounding Boxes")):
                result = renderer.render_source(
                    camera, gaussians, exclude_list=[], scene_info = dataset.scene_info)

                visualizer.visualize_original_with_bbox(result, camera)  # 直接保存带bbox的图像

def draw_bbox_sequence_all():
    safe_state(cfg.eval.quiet)
    cfg.mode

    cfg.render.save_image = True
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        gaussians.render_actors = True
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
    save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
        scene.loaded_iter), 'draw_source_bbox')
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    model_keys = gaussians.model_name_id.keys()
    model_keys_list = list(model_keys)
    model_keys_list.remove("background")

    random.seed(time.time())

    with torch.no_grad():
        if not cfg.eval.skip_train:
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            
            for idx, camera in enumerate(tqdm(cameras, desc="Drawing Bounding Boxes")):
                result = renderer.render_source_all(
                    camera, gaussians, exclude_list=[], scene_info = dataset.scene_info)

                visualizer.visualize_original_with_bbox(result, camera)  # 直接保存带bbox的图像

if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)

    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    elif cfg.mode == 'edit':
        render_with_edit()
    elif cfg.mode == 'edit_all':
        render_with_edit_all()
    elif cfg.mode == 'render_all':
        render_all()
    elif cfg.mode == 'video':
        i2v()
    elif cfg.mode == 'draw_bbox':
        draw_bbox_sequence()
    elif cfg.mode == 'draw_bbox_all':
        draw_bbox_sequence_all()
    elif cfg.mode == 'specific_frame':
        render_specific_frame()
    else:
        raise NotImplementedError()
