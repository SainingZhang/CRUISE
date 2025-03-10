import torch
from lib.utils.sh_utils import eval_sh
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer
from lib.config import cfg
from lib.utils.general_utils import quaternion_to_matrix, \
    build_scaling_rotation, \
    strip_symmetric, \
    quaternion_raw_multiply, \
    startswith_any, \
    matrix_to_quaternion, \
    quaternion_invert
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask, draw_3d_bbox
import numpy as np
import cv2

class StreetGaussianRenderer():
    def __init__(
        self,         
    ):
        self.cfg = cfg.render
              
    def render_all(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        
        # render all
        render_composition = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # render background
        render_background = self.render_background(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        # render object
        render_object = self.render_object(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_object['rgb']
        result['acc_object'] = render_object['acc']
        
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
    
        return result
    
    def render_object(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):        
        pc.set_visibility(include_list=pc.obj_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_background(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        pc.set_visibility(include_list=['background'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_sky(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):  
        pc.set_visibility(include_list=['sky'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        return result
    
    def render(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
    ):   
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                    
        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color) # 出来之后pc的get_xyz又没有了(首先第一次进去是有的)

        # Step2: render sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)

        return result
    
    def render_edit(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
        scene_info = None,
    ):   
        
        # import pdb; pdb.set_trace()
        
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                
        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # Step2: render sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        # Step3: handle ego car replacement
        if hasattr(viewpoint_camera, 'original_ego_mask') and hasattr(viewpoint_camera, 'original_image'):
            ego_mask = viewpoint_camera.original_ego_mask.cuda()
            original_image = viewpoint_camera.original_image.cuda()
            
            
            result['rgb_without_ego'] = result['rgb'].clone()
            
            
            result['rgb'] = torch.where(ego_mask.repeat(3, 1, 1), original_image, result['rgb'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)
            
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
        # import pdb; pdb.set_trace()
        result = self.render_bbox(viewpoint_camera, pc, result, scene_info)

        return result
    
    
    def render_edit_all(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
        scene_info = None,
        not_render_bbox_list = [],
    ):   
        

        
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                
        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        

        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        


        # Step2: render sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        # Step3: handle ego car replacement
        if hasattr(viewpoint_camera, 'original_ego_mask') and hasattr(viewpoint_camera, 'original_image'):
            ego_mask = viewpoint_camera.original_ego_mask.cuda()
            original_image = viewpoint_camera.original_image.cuda()
            
            
            result['rgb_without_ego'] = result['rgb'].clone()
            
            
            result['rgb'] = torch.where(ego_mask.repeat(3, 1, 1), original_image, result['rgb'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)
            
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
        # import pdb; pdb.set_trace()
        result = self.render_bbox_all(viewpoint_camera, pc, result, scene_info,not_render_bbox_list)

        return result
    
    def render_source(
            self, 
            viewpoint_camera: Camera,
            pc: StreetGaussianModel,
            convert_SHs_python = None, 
            compute_cov3D_python = None, 
            scaling_modifier = None, 
            override_color = None,
            exclude_list = [],
            scene_info = None,
        ):   
            

            include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                
            pc.set_visibility(include_list)
            pc.parse_camera(viewpoint_camera)
            
            result = {'original_image': viewpoint_camera.original_image[:3].cuda()}

            result = self.draw_bbox(viewpoint_camera, pc, result, scene_info)

            return result
    def render_source_all(
            self, 
            viewpoint_camera: Camera,
            pc: StreetGaussianModel,
            convert_SHs_python = None, 
            compute_cov3D_python = None, 
            scaling_modifier = None, 
            override_color = None,
            exclude_list = [],
            scene_info = None,
        ):   
            

            include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                
            pc.set_visibility(include_list)
            pc.parse_camera(viewpoint_camera)
            
            result = {'original_image': viewpoint_camera.original_image[:3].cuda()}

            result = self.draw_bbox_all(viewpoint_camera, pc, result, scene_info)

            return result            
    def render_kernel(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        white_background = cfg.data.white_background,
    ):
        
        if pc.num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier)
        
        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        if cfg.mode == 'train':
            screenspace_points = torch.zeros((pc.num_gaussians, 3), requires_grad=True).float().cuda() + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
        else:
            screenspace_points = None 

        means3D = pc.get_xyz_edit
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                try:
                    shs = pc.get_features
                except:
                    colors_precomp = pc.get_colors(viewpoint_camera.camera_center)
        else:
            colors_precomp = override_color

        # TODO: add more feature here
        feature_names = []
        feature_dims = []
        features = []
        
        if cfg.render.render_normal:
            normals = pc.get_normals(viewpoint_camera)
            feature_names.append('normals')
            feature_dims.append(normals.shape[-1])
            features.append(normals)

        if cfg.data.get('use_semantic', False):
            semantics = pc.get_semantic
            feature_names.append('semantic')
            feature_dims.append(semantics.shape[-1])
            features.append(semantics)
        
        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None
        
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            semantics = features,
        )  
        
        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)
        
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
                
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
            assert semantic_mode in ['logits', 'probabilities']
            if semantic_mode == 'logits': 
                pass # return raw semantic logits
            else:
                rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

            rendered_feature_dict['semantic'] = rendered_semantic
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
        }
        
        result.update(rendered_feature_dict)
        
        return result

    def render_bbox(self, camera, gaussians, result, scene_info):
        """渲染物体边界框"""
        camera_info = scene_info.train_cameras[camera.id]
        H, W = camera.image_height, camera.image_width
        bbox_image = torch.zeros((3, H, W), device='cuda')
        
        if gaussians.include_obj and gaussians.render_actors:
        
            for obj_name in gaussians.graph_obj_list:
                obj_model = getattr(gaussians, obj_name)
                track_id = obj_model.track_id

            
                obj_rot = gaussians.actor_pose.get_tracking_rotation(track_id, camera)
                obj_trans = gaussians.actor_pose.get_tracking_translation(track_id, camera)
                
            
                obj_rot_matrix = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
                obj_pose_vehicle = torch.eye(4, device='cuda')
                obj_pose_vehicle[:3, :3] = obj_rot_matrix
                obj_pose_vehicle[:3, 3] = obj_trans
                
            
                corners_local = torch.cat([obj_model.bbox_corners_local, 
                                         torch.ones_like(obj_model.bbox_corners_local[:, :1])], dim=1)
                
            
                corners_vehicle = corners_local @ obj_pose_vehicle.T
                corners_vehicle = corners_vehicle[:, :3]
                
                ixt = camera_info.K
                ext = camera_info.metadata['extrinsic']
                
                bbox_lines = draw_3d_bbox(
                    corners_3d=corners_vehicle.cpu().numpy(),
                    K=ixt,
                    pose=np.linalg.inv(ext),
                    H=H, W=W,
                    color=(0, 1, 0),
                    thickness=2
                )
                
                bbox_image += torch.from_numpy(bbox_lines).to(bbox_image.device)

        bbox_image = torch.clamp(bbox_image, 0, 1)
        result['rgb_with_bbox'] = result['rgb'] * (1 - bbox_image) + bbox_image
        return result
    
    
    def render_bbox_all(self, camera, gaussians, result, scene_info, not_render_bbox_list):
        """渲染物体边界框"""
        camera_info = scene_info.train_cameras[camera.id]
        H, W = camera.image_height, camera.image_width
        bbox_image = torch.zeros((3, H, W), device='cuda')
        
        if gaussians.include_obj and gaussians.render_actors:
            if gaussians.graph_all_obj_list != gaussians.graph_obj_list:

                print("all", gaussians.graph_all_obj_list)
                print("not static", gaussians.graph_obj_list)
                
        
            for obj_name in gaussians.graph_all_obj_list:
                
                if obj_name in not_render_bbox_list:
                    continue
                
                try:
                    obj_model = getattr(gaussians, obj_name)
                    
                    track_id = obj_model.track_id
                    
                 
                    obj_rot = gaussians.actor_pose.get_tracking_rotation(track_id, camera)
                    obj_trans = gaussians.actor_pose.get_tracking_translation(track_id, camera)
                    
         
                    obj_rot_matrix = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
                    obj_pose_vehicle = torch.eye(4, device='cuda')
                    obj_pose_vehicle[:3, :3] = obj_rot_matrix
                    obj_pose_vehicle[:3, 3] = obj_trans
                    
                  
                    corners_local = torch.cat([obj_model.bbox_corners_local, 
                                            torch.ones_like(obj_model.bbox_corners_local[:, :1])], dim=1)
                    
                    # 坐标变换
                    corners_vehicle = corners_local @ obj_pose_vehicle.T
                    corners_vehicle = corners_vehicle[:, :3]
                    
       
                    ixt = camera_info.K
                    ext = camera_info.metadata['extrinsic']
                    

                    bbox_lines = draw_3d_bbox(
                        corners_3d=corners_vehicle.cpu().numpy(),
                        K=ixt,
                        pose=np.linalg.inv(ext),
                        H=H, W=W,
                        color=(0, 1, 0),
                        thickness=2
                    )
                    
                    bbox_image += torch.from_numpy(bbox_lines).to(bbox_image.device)
                    

                except:
                    obj_model = gaussians.all_actor_manager.get_actor(obj_name)
                
                    track_id = obj_model.track_id
                    
                
                    
                    try:
                        obj_rot = gaussians.all_actor_pose.get_tracking_rotation(track_id, camera)
                        obj_trans = gaussians.all_actor_pose.get_tracking_translation(track_id, camera)
                    except:
                    
                        print(f"Error getting tracking rotation and translation for object {obj_name}")
                        
                    

                    obj_rot_matrix = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
                    obj_pose_vehicle = torch.eye(4, device='cuda')
                    obj_pose_vehicle[:3, :3] = obj_rot_matrix
                    obj_pose_vehicle[:3, 3] = obj_trans
                    
                    bbox = [[-obj_model.bbox[0]/2, -obj_model.bbox[1]/2, -obj_model.bbox[2]/2],
                        [obj_model.bbox[0]/2, obj_model.bbox[1]/2, obj_model.bbox[2]/2]]
                    
                    corners_local = torch.from_numpy(bbox_to_corner3d(bbox)).float().cuda()
                    corners_local = torch.cat([corners_local, torch.ones_like(corners_local[:, :1])], dim=1)
                    
                    corners_vehicle = corners_local @ obj_pose_vehicle.T
                    corners_vehicle = corners_vehicle[:, :3]
                    
                    ixt = camera_info.K
                    ext = camera_info.metadata['extrinsic']
                    
                   
                    bbox_lines = draw_3d_bbox(
                        corners_3d=corners_vehicle.cpu().numpy(),
                        K=ixt,
                        pose=np.linalg.inv(ext),
                        H=H, W=W,
                        color=(0, 1, 0),  # 使用绿色
                        thickness=2
                    )
                    
                   
                    bbox_image += torch.from_numpy(bbox_lines).to(bbox_image.device)
                    
                
        bbox_image = torch.clamp(bbox_image, 0, 1)
        result['rgb_with_bbox'] = result['rgb'] * (1 - bbox_image) + bbox_image
        return result
    
    
    def draw_bbox(self, camera, gaussians, result, scene_info):
        """渲染物体边界框，在原图上也能画框"""
        
        camera_info = scene_info.train_cameras[camera.id]
        H, W = camera.image_height, camera.image_width
        bbox_image = torch.zeros((3, H, W), device='cuda')

       
        text_image = np.zeros((H, W, 3), dtype=np.float32)
        
        for obj_name in gaussians.graph_obj_list:
            obj_model = getattr(gaussians, obj_name)
            track_id = obj_model.track_id

            
            obj_rot = gaussians.actor_pose.get_tracking_rotation(track_id, camera)
            obj_trans = gaussians.actor_pose.get_tracking_translation(track_id, camera)
            
            
            obj_rot_matrix = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
            obj_pose_vehicle = torch.eye(4, device='cuda')
            obj_pose_vehicle[:3, :3] = obj_rot_matrix
            obj_pose_vehicle[:3, 3] = obj_trans
            
           
            bbox = [[-obj_model.bbox[0]/2, -obj_model.bbox[1]/2, -obj_model.bbox[2]/2],
                    [obj_model.bbox[0]/2, obj_model.bbox[1]/2, obj_model.bbox[2]/2]]
            
            corners_local = torch.from_numpy(bbox_to_corner3d(bbox)).float().cuda()
            corners_local = torch.cat([corners_local, torch.ones_like(corners_local[:, :1])], dim=1)
            
            
            corners_vehicle = corners_local @ obj_pose_vehicle.T
            corners_vehicle = corners_vehicle[:, :3]
            
            
            ixt = camera_info.K
            ext = camera_info.metadata['extrinsic']
            
            
            bbox_lines = draw_3d_bbox(
                corners_3d=corners_vehicle.cpu().numpy(),
                K=ixt,
                pose=np.linalg.inv(ext),
                H=H, W=W,
                color=(0, 1, 0),  # 使用绿色
                thickness=2
            )
            
            bbox_image += torch.from_numpy(bbox_lines).to(bbox_image.device)
            
           
            top_center = corners_vehicle[4:5]  
            top_center_homo = np.concatenate([top_center.cpu().numpy(), np.ones((1, 1))], axis=1)
            cam_point = top_center_homo @ np.linalg.inv(ext).T
            cam_point = cam_point[:, :3]
            img_point = cam_point @ ixt.T
            img_point = img_point[:, :2] / img_point[:, 2:]
            img_point = img_point[0].astype(int)
            
            
            cv2.putText(
                text_image,
                f'ID:{track_id}',
                (img_point[0], img_point[1] - 10),  # 文字位置略高于边界框顶部
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # 字体大小
                (255, 0, 0),  # 红色
                1,  # 线条粗细
                cv2.LINE_AA
            )

       
        text_tensor = torch.from_numpy(text_image).permute(2, 0, 1).cuda()
        bbox_image = torch.max(bbox_image, text_tensor) 


        bbox_image = torch.clamp(bbox_image, 0, 1)
        result['original_with_bbox'] = result['original_image'] * (1 - bbox_image) + bbox_image
        return result
    
    def draw_bbox_all(self, camera, gaussians, result, scene_info):
        """渲染物体边界框，在原图上也能画框"""

        camera_info = scene_info.train_cameras[camera.id]
        H, W = camera.image_height, camera.image_width
        bbox_image = torch.zeros((3, H, W), device='cuda')  

       
        text_image = np.zeros((H, W, 3), dtype=np.float32)


        for obj_name in gaussians.graph_all_obj_list:
            obj_model = gaussians.all_actor_manager.get_actor(obj_name)
        
            track_id = obj_model.track_id
            
            try:
                obj_rot = gaussians.all_actor_pose.get_tracking_rotation(track_id, camera)
                obj_trans = gaussians.all_actor_pose.get_tracking_translation(track_id, camera)
            except:
                print(f"Error getting tracking rotation and translation for object {obj_name}")
                
                
            

            obj_rot_matrix = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
            obj_pose_vehicle = torch.eye(4, device='cuda')
            obj_pose_vehicle[:3, :3] = obj_rot_matrix
            obj_pose_vehicle[:3, 3] = obj_trans
            

            bbox = [[-obj_model.bbox[0]/2, -obj_model.bbox[1]/2, -obj_model.bbox[2]/2],
                    [obj_model.bbox[0]/2, obj_model.bbox[1]/2, obj_model.bbox[2]/2]]
            
            corners_local = torch.from_numpy(bbox_to_corner3d(bbox)).float().cuda()
            corners_local = torch.cat([corners_local, torch.ones_like(corners_local[:, :1])], dim=1)
            

            corners_vehicle = corners_local @ obj_pose_vehicle.T
            corners_vehicle = corners_vehicle[:, :3]
            

            ixt = camera_info.K
            ext = camera_info.metadata['extrinsic']
            

            bbox_lines = draw_3d_bbox(
                corners_3d=corners_vehicle.cpu().numpy(),
                K=ixt,
                pose=np.linalg.inv(ext),
                H=H, W=W,
                color=(0, 1, 0),  
                thickness=2
            )
            
            bbox_image += torch.from_numpy(bbox_lines).to(bbox_image.device)
            

            top_center = corners_vehicle[4:5]  
            top_center_homo = np.concatenate([top_center.cpu().numpy(), np.ones((1, 1))], axis=1)
            cam_point = top_center_homo @ np.linalg.inv(ext).T
            cam_point = cam_point[:, :3]
            img_point = cam_point @ ixt.T
            img_point = img_point[:, :2] / img_point[:, 2:]
            img_point = img_point[0].astype(int)
            

            cv2.putText(
                text_image,
                f'ID:{track_id}',
                (img_point[0], img_point[1] - 10),  # 文字位置略高于边界框顶部
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # 字体大小
                (255, 0, 0),  # 红色
                1,  # 线条粗细
                cv2.LINE_AA
            )

        text_tensor = torch.from_numpy(text_image).permute(2, 0, 1).cuda()
        bbox_image = torch.max(bbox_image, text_tensor)  

        
        bbox_image = torch.clamp(bbox_image, 0, 1)
        result['original_with_bbox'] = result['original_image'] * (1 - bbox_image) + bbox_image
        return result