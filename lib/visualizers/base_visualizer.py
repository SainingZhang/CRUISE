import os
import torchvision
import cv2
import torch
import imageio
import numpy as np

from lib.utils.camera_utils import Camera
from lib.utils.img_utils import visualize_depth_numpy
from lib.config import cfg


class BaseVisualizer():
    def __init__(self, save_dir):
        self.result_dir = save_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.save_video = cfg.render.save_video
        self.save_image = cfg.render.save_image
        
        self.rgbs = []
        self.depths = []
        self.diffs = []
        
        self.depth_visualize_func = lambda x: visualize_depth_numpy(x, cmap=cv2.COLORMAP_JET)[0][..., [2, 1, 0]]
        self.diff_visualize_func = lambda x: visualize_depth_numpy(x, cmap=cv2.COLORMAP_TURBO)[0][..., [2, 1, 0]]

    def visualize(self, result, camera: Camera):
        name = camera.image_name
        rgb = result['rgb']

        if self.save_image:
            
            torchvision.utils.save_image(rgb, os.path.join(self.result_dir, f'{name}_rgb.png'))
            torchvision.utils.save_image(camera.original_image[:3], os.path.join(self.result_dir, f'{name}_gt.png'))
        if self.save_video:
            rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs.append(rgb)
            
        self.visualize_diff(result, camera)
        self.visualize_depth(result, camera)
        
    def visualize_new(self, result, camera: Camera):
        name = camera.image_name
        rgb = result['rgb']

        if self.save_image:
            os.makedirs(os.path.join(self.result_dir, 'gt'), exist_ok=True)
            os.makedirs(os.path.join(self.result_dir, 'rgb'), exist_ok=True)
            torchvision.utils.save_image(rgb, os.path.join(self.result_dir, 'rgb', f'{name}_rgb.png'))
            torchvision.utils.save_image(camera.original_image[:3], os.path.join(self.result_dir, 'gt', f'{name}_gt.png'))
        if self.save_video:
            rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs.append(rgb)
            
        self.visualize_diff(result, camera)
        self.visualize_depth(result, camera)
    
    def visualize_diff(self, result, camera: Camera):
        name = camera.image_name
        rgb_gt = camera.original_image[:3]
        rgb = result['rgb'].detach().cpu()  
                
        if hasattr(camera, 'original_mask'):
            mask = camera.original_mask.bool()
        else:
            mask = torch.ones_like(rgb[0]).bool()
            
        rgb = torch.where(mask, rgb, torch.zeros_like(rgb))
        rgb_gt = torch.where(mask, rgb_gt, torch.zeros_like(rgb_gt))
        
        rgb = rgb.permute(1, 2, 0).numpy() # [H, W, 3]
        rgb_gt = rgb_gt.permute(1, 2, 0).numpy() # [H, W, 3]
        diff = ((rgb - rgb_gt) ** 2).sum(axis=-1, keepdims=True) # [H, W, 1]
        
        if self.save_image:
            imageio.imwrite(os.path.join(self.result_dir, f'{name}_diff.png'), self.diff_visualize_func(diff))
        
        if self.save_video:
            self.diffs.append(diff)

    def visualize_depth(self, result, camera: Camera):
        name = camera.image_name
        depth = result['depth']

        depth = depth.detach().permute(1, 2, 0).detach().cpu().numpy() # [H, W, 1]
        
        if self.save_image:
            imageio.imwrite(os.path.join(self.result_dir, f'{name}_depth.png'), self.diff_visualize_func(depth))
        
        if self.save_video:
            self.depths.append(depth)
        
    def save_video_from_frames(self, frames, name, visualize_func=None):
        if len(frames) == 0:
            return
        
        unqiue_cams = sorted(list(set(self.cams)))
        if len(unqiue_cams) == 1:
        
            if visualize_func is not None:
                frames = [visualize_func(frame) for frame in frames]
        
            imageio.mimwrite(os.path.join(self.result_dir, f'{name}.mp4'), frames, fps=cfg.render.fps)
        else:
            if cfg.render.get('concat_cameras', False):
                concat_cameras = cfg.render.concat_cameras
                frames_cam_all = []
                for cam in concat_cameras:
                    frames_cam = [frame for frame, c in zip(frames, self.cams) if c == cam]
                    frames_cam_all.append(frames_cam)
                
                frames_cam_len = [len(frames_cam) for frames_cam in frames_cam_all]
                assert len(list(set(frames_cam_len))) == 1, 'all cameras should have same number of frames'
                num_frames = frames_cam_len[0]

                
                frames_concat_all = []
                for i in range(num_frames):
                    frames_concat = []
                    for j in range(len(concat_cameras)):
                        frames_concat.append(frames_cam_all[j][i])
                    frames_concat = np.concatenate(frames_concat, axis=1)
                    frames_concat_all.append(frames_concat)
                
                if visualize_func is not None:
                    frames_concat_all = [visualize_func(frame) for frame in frames_concat_all]    
        
                imageio.mimwrite(os.path.join(self.result_dir, f'{name}.mp4'), frames_concat_all, fps=cfg.render.fps)
            
            else:
                for cam in unqiue_cams:
                    frames_cam = [frame for frame, c in zip(frames, self.cams) if c == cam]
                    
                    if visualize_func is not None:
                        frames_cam = [visualize_func(frame) for frame in frames_cam]
                    
                    imageio.mimwrite(os.path.join(self.result_dir, f'{name}_{str(cam)}.mp4'), frames_cam, fps=cfg.render.fps)
                    
    def summarize(self):
        if cfg.render.get('save_video', True):
            self.save_video_from_frames(self.rgbs, 'color')
            self.save_video_from_frames(self.depths, 'depth', visualize_func=self.depth_visualize_func)
            self.save_video_from_frames(self.diffs, 'diff', visualize_func=self.diff_visualize_func)

        
    # def visualize_combined(self, result, camera: Camera, idx):
    #     name = camera.image_name
    #     rgb = result['rgb']  # 渲染图像
    #     gt = camera.original_image[:3]  # Ground truth 图像
        
    #     os.makedirs(os.path.join(self.result_dir, 'combined'), exist_ok=True)

    #     if self.save_image:
    #         # 将渲染图像和GT图像拼接
    #         combined_image = torch.cat((gt, rgb.detach().cpu()), dim=-1)  # 水平拼接，如果想垂直拼接使用dim=1

    #         # 保存拼接后的图像
    #         torchvision.utils.save_image(combined_image, os.path.join(self.result_dir, 'combined',f'{idx}_{name}_combined.png'))
        
    #     if self.save_video:
    #         rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    #         self.rgbs.append(rgb)
            
    #     # self.visualize_diff(result, camera)
    #     # self.visualize_depth(result, camera)
        
    #     def visualize_generate_dataset(self, result, camera: Camera, generate_path):
    #         '''
    #         根据当前的视角是哪一个,选择对应的保存路径
    #         '''
    #         name = camera.image_name
            
            
    #         rgb = result['rgb']  # 渲染图像
    #         gt = camera.original_image[:3]  # Ground truth 图像
            
    #         generate_car_path = os.path.join(generate_path, 'vehicle-side','image')
    #         generate_road_path = os.path.join(generate_path, 'infrastructure-side', 'image')
            
    #         os.makedirs(generate_car_path,exist_ok=True)
    #         os.makedirs(generate_road_path,exist_ok=True)
            
    #         os.makedirs(os.path.join(self.result_dir, 'combined'), exist_ok=True)

    #         if self.save_image:
    #             # 将渲染图像和GT图像拼接
                

    #             # 保存拼接后的图像
    #             if 1:
    #                 torchvision.utils.save_image(rgb.detach().cpu(), os.path.join(generate_car_path,f'{name}.png'))
    #             elif 0:
    #                 torchvision.utils.save_image(rgb.detach().cpu(), os.path.join(generate_road_path,f'{name}.png'))
            
    #         if self.save_video:
    #             rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    #             self.rgbs.append(rgb)
                
    def visualize_combined(self, result, camera: Camera):
        """将RGB、GT、Depth和Diff四张图拼接在一起保存"""
        name = camera.image_name
        
        # 获取RGB图像
        rgb = result['rgb'].detach().cpu()
        
        # 获取GT图像
        rgb_gt = camera.original_image[:3]
        
        # 获取深度图
        depth = result['depth'].detach()  # [1, H, W]
        depth = depth.permute(1, 2, 0).cpu().numpy()  # [H, W, 1]
        depth_colored = self.depth_visualize_func(depth)  # [H, W, 3]
        depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        
        # 计算diff图
        if hasattr(camera, 'original_mask'):
            mask = camera.original_mask.bool()
        else:
            mask = torch.ones_like(rgb[0]).bool()
            
        rgb_masked = torch.where(mask, rgb, torch.zeros_like(rgb))
        rgb_gt_masked = torch.where(mask, rgb_gt, torch.zeros_like(rgb_gt))
        
        rgb_np = rgb_masked.permute(1, 2, 0).numpy()  # [H, W, 3]
        rgb_gt_np = rgb_gt_masked.permute(1, 2, 0).numpy()  # [H, W, 3]
        diff = ((rgb_np - rgb_gt_np) ** 2).sum(axis=-1, keepdims=True)  # [H, W, 1]
        diff_colored = self.diff_visualize_func(diff)  # [H, W, 3]
        diff_colored = torch.from_numpy(diff_colored).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        
        # 拼接图像
        # 上排：RGB和GT
        row1 = torch.cat([rgb, rgb_gt], dim=2)
        # 下排：Depth和Diff
        row2 = torch.cat([depth_colored, diff_colored], dim=2)
        # 上下拼接
        combined = torch.cat([row1, row2], dim=1)
        
        if self.save_image:
            os.makedirs(os.path.join(self.result_dir, 'combined'), exist_ok=True)
            torchvision.utils.save_image(
                combined, 
                os.path.join(self.result_dir, 'combined', f'{name}_combined.png')
            )
        
        if self.save_video:
            combined_np = (combined.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            self.rgbs.append(combined_np)

    def visualize_bbox(self, result, camera: Camera):
        """保存带有边界框的渲染结果"""
        name = camera.image_name
        
        # 如果结果中没有带边界框的渲染，直接返回
        if 'rgb_with_bbox' not in result:
            return
        
        # 获取带边界框的RGB图像
        rgb_with_bbox = result['rgb_with_bbox'].detach().cpu()
        
        if self.save_image:
            # 创建专门的bbox可视化目录
            bbox_dir = os.path.join(self.result_dir, 'bbox')
            os.makedirs(bbox_dir, exist_ok=True)
            
            # 保存带边界框的图像
            torchvision.utils.save_image(
                rgb_with_bbox,
                os.path.join(bbox_dir, f'{name}_bbox.png')
            )
        
        if self.save_video:
            # 如果需要保存视频，将帧添加到列表中
            rgb_with_bbox = (rgb_with_bbox.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if not hasattr(self, 'bbox_frames'):
                self.bbox_frames = []
            self.bbox_frames.append(rgb_with_bbox)
            
    def visualize_original_with_bbox(self, result, camera: Camera):
        """保存带有边界框的渲染结果"""
        name = camera.image_name
        
        # 如果结果中没有带边界框的渲染，直接返回
        if 'original_with_bbox' not in result:
            return
        
        # 获取带边界框的RGB图像
        original_with_bbox = result['original_with_bbox'].detach().cpu()
        
        
        # 创建专门的bbox可视化目录
        bbox_dir = os.path.join(self.result_dir, 'bbox')
        os.makedirs(bbox_dir, exist_ok=True)
        
        # 保存带边界框的图像
        torchvision.utils.save_image(
            original_with_bbox,
            os.path.join(bbox_dir, f'{name}_original_with_bbox.png')
        )
        
        # if self.save_video:
        #     # 如果需要保存视频，将帧添加到列表中
        #     rgb_with_bbox = (rgb_with_bbox.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        #     if not hasattr(self, 'original_with_bbox_frames'):
        #         self.original_with_bbox_frames = []
        #     self.original_with_bbox_frames.append(original_with_bbox)