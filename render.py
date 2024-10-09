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

def render_with_edit():
    safe_state(cfg.eval.quiet)
    cfg.mode

    cfg.render.save_image = True
    cfg.render.save_video = False
    generate_new_dataset = False
    generate_path = "generate/new_car"

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        
    save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(
        scene.loaded_iter), '002_replace_1_fixed_car')
    os.makedirs(save_dir, exist_ok=True)


    model_keys = gaussians.model_name_id.keys()
    model_keys_list = list(model_keys)
    model_keys_list.remove("background")


    random.seed(time.time())

    vehicle_models = gaussians.model_keys

    replacement_ratio = 1


    replacement_dict = {}
    for obj in model_keys_list:
        if random.random() < replacement_ratio:  
            replacement_dict[obj] = random.choice(vehicle_models)
    print(replacement_dict)

    with torch.no_grad():
        if not cfg.eval.skip_train:
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            print(gaussians.model_name_id.keys())
            for obj_name, ply_path in replacement_dict.items():
                gaussians.replace_gaussian_with_custom_actor(ply_path, obj_name)
            exclude_list = []


    with torch.no_grad():
        if not cfg.eval.skip_train:
            for idx, camera in enumerate(tqdm(cameras[60:62], desc="Rendering Training View")):
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render_edit(
                    camera, gaussians, exclude_list=exclude_list)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                visualizer.visualize(result, camera)


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
                scene.loaded_iter), 'temp')
            os.makedirs(save_dir, exist_ok=True)
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            print(gaussians.model_name_id.keys())

            exclude_list = []

            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):

                torch.cuda.synchronize()
                start_time = time.time()

                result = renderer.render_edit(
                    camera, gaussians, exclude_list=exclude_list)

                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

                # visualizer.visualize(result, camera)
                visualizer.visualize_combined(result, camera)

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

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)
            visualizer.visualize(result, camera)

        visualizer.summarize()



if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)

    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    elif cfg.mode == 'edit':
        render_with_edit()
    else:
        raise NotImplementedError()
