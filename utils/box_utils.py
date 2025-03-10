import numpy as np
import cv2

def get_bound_2d_mask(corners_3d, K, pose, H, W):
    corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
    corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=1e-3, a_max=None)
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:] # TODO 可以参考的投影的代码
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def draw_3d_bbox(corners_3d, K, pose, H, W, color=(0,1,0), thickness=2):
    """绘制3D边界框的线条"""
    # 投影3D点到2D
    corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
    corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=1e-3, a_max=None)
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
    corners_2d = np.round(corners_2d).astype(np.int32)  # 确保转换为整数类型
    
    # 创建图像
    image = np.zeros((H, W, 3), dtype=np.float32)
    
    # 定义边界框的连接线
    lines = [
        # 底面
        (0,1), (1,3), (3,2), (2,0),
        # 顶面
        (4,5), (5,7), (7,6), (6,4),
        # 连接线
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    # 绘制线条
    for (start_idx, end_idx) in lines:
        start_point = tuple(corners_2d[start_idx].tolist())  # 确保转换为tuple
        end_point = tuple(corners_2d[end_idx].tolist())     # 确保转换为tuple
        
        try:
            # 使用cv2.clipLine裁剪线段
            ret, p1, p2 = cv2.clipLine((0, 0, W, H), start_point, end_point)
            if ret:  # 如果线段与图像有交点
                cv2.line(image, p1, p2, color, thickness)
        except Exception as e:
            print(f"Warning: Failed to draw line from {start_point} to {end_point}: {e}")
            continue
    
    # 转换为PyTorch期望的格式 (3, H, W)
    return image.transpose(2, 0, 1)

def scale_to_corrner(scale):
    min_x, min_y, min_z = -scale, -scale, -scale
    max_x, max_y, max_z = scale, scale, scale
    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    
    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def points_to_bbox(points):    
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    bbox = np.array([min_xyz, max_xyz])
    return bbox

def inbbox_points(points, corner3d):
    min_xyz = corner3d[0]
    max_xyz = corner3d[-1]
    return np.logical_and(
        np.all(points >= min_xyz, axis=-1),
        np.all(points <= max_xyz, axis=-1)
    )

