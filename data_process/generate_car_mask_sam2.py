import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def image_filename_to_cam(x: str) -> int:
    """从图片文件名获取相机ID"""
    return int(x.split('.')[0][-1])

def image_filename_to_frame(x: str) -> int:
    """从图片文件名获取帧ID"""
    return int(x.split('.')[0][:6])

def add_to_mask_dict(masks_dict: dict, mask_path: str):
    """将mask添加到字典中"""
    basename = os.path.basename(mask_path)
    cam = image_filename_to_cam(basename)
    frame = image_filename_to_frame(basename)
    mask = cv2.imread(mask_path)
    if frame not in masks_dict:
        masks_dict[frame] = [None] * 3  # FRONT_LEFT, FRONT, FRONT_RIGHT 1, 0, 2
    if cam == 1:
        masks_dict[frame][0] = mask
    elif cam == 0:
        masks_dict[frame][1] = mask
    elif cam == 2:
        masks_dict[frame][2] = mask

def segment_with_text_prompt(
    datadir: str,
    BOX_TRESHOLD: list,
    TEXT_TRESHOLD: float,
    ignore_exists: bool,
    sam2_checkpoint: str,
    sam2_config: str,
    force_cpu: bool = False
):
    """使用SAM2.1进行分割"""
    save_dir = os.path.join(datadir, 'car_mask_1')
    os.makedirs(save_dir, exist_ok=True)

    # 设置设备
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    
    # 初始化模型
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)

    # 获取所有图片文件
    image_dir = os.path.join(datadir, 'images')
    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))
    
    masks_dict = {}
    for image_path in tqdm(image_files):
        image_base_name = image_path.name
        output_mask = os.path.join(save_dir, image_base_name)
        
        if os.path.exists(output_mask) and ignore_exists:
            add_to_mask_dict(masks_dict, output_mask)
            print(f'{output_mask} exists, skip')
            continue
            
        cam = image_filename_to_cam(image_base_name)
        box_threshold = BOX_TRESHOLD[cam]
        
        # 加载和处理图片
        image = Image.open(image_path)
        image_array = np.array(image.convert("RGB"))
        sam2_predictor.set_image(image_array)
        
        # 目标检测
        text_prompt = "car. motorcycle. bicycle. person. tricycle."
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=TEXT_TRESHOLD,
            target_sizes=[image.size[::-1]]
        )
        
        input_boxes = results[0]["boxes"].cpu().numpy()
        print(f'detecting {len(input_boxes)} boxed of car in {str(image_path)}, box_threshold: {box_threshold}')
        
        if len(input_boxes) == 0:
            mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        else:
            # SAM2预测
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            # 添加调试信息
            print(f"Masks type: {type(masks)}")
            print(f"Masks shape: {masks.shape}")
            print(f"Masks dtype: {masks.dtype}")
            
            if masks.ndim == 4:
                masks = masks.squeeze(1)
                
            # 合并所有mask - 添加更多的类型检查和错误处理
            mask = np.zeros(image_array.shape[:2], dtype=bool)
            for i, m in enumerate(masks):
                try:
                    # 打印每个mask的信息
                    print(f"Processing mask {i}")
                    print(f"Mask type: {type(m)}")
                    print(f"Mask shape: {m.shape}")
                    print(f"Mask dtype: {m.dtype}")
                    
                    # 转换为numpy数组
                    if torch.is_tensor(m):
                        m_np = m.cpu().numpy()
                    else:
                        m_np = np.asarray(m)
                    
                    # 确保类型正确
                    m_bool = m_np.astype(bool)
                    
                    # 确保形状匹配
                    if m_bool.shape != mask.shape:
                        print(f"Warning: Shape mismatch - mask: {mask.shape}, m_bool: {m_bool.shape}")
                        continue
                    
                    # 执行位运算
                    mask = np.logical_or(mask, m_bool)
                    
                except Exception as e:
                    print(f"Error processing mask {i}: {str(e)}")
                    continue
            
            # 转换为uint8以保存
            mask = mask.astype(np.uint8) * 255
            
        cv2.imwrite(output_mask, mask)
        add_to_mask_dict(masks_dict, output_mask)
        
    return masks_dict

def visualize_masks(datadir: str):
    """可视化mask结果"""
    image_folder = os.path.join(datadir, "images")
    mask_folder = os.path.join(datadir, "car_mask_sam2_1")
    output_folder = os.path.join(datadir, "car_mask_visualization_sam2_1")
    
    os.makedirs(output_folder, exist_ok=True)
    
    mask_files = sorted(os.listdir(mask_folder))
    
    for mask_file in tqdm(mask_files):
        if not mask_file.endswith(('.jpg', '.png')):
            continue
            
        image_path = os.path.join(image_folder, mask_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        if not os.path.exists(image_path):
            print(f"找不到对应的图片: {image_path}")
            continue
            
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 0, 255]  # BGR格式，红色
        
        overlay = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
        
        output_path = os.path.join(output_folder, mask_file)
        cv2.imwrite(output_path, overlay)
    
    print(f"处理完成！结果保存在: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="使用SAM2.1生成车辆分割mask")
    parser.add_argument("--datadir", type=str, default="/mnt/zhangsn/data/V2X-Seq-SPD-Processed/0022_exp/0022_0_original_all_seperate_with_cooperative_pointcloud", help="数据目录路径")
    parser.add_argument("--box-threshold", type=float, nargs="+", default=[0.3], help="检测框阈值")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="文本阈值")
    parser.add_argument("--ignore-exists", action="store_true", help="是否忽略已存在的mask")
    parser.add_argument("--sam2-checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt", help="SAM2检查点路径")
    parser.add_argument("--sam2-config", type=str, default="./configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2配置文件路径")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU")
    
    args = parser.parse_args()
    
    # 处理box_threshold
    if len(args.box_threshold) == 1:
        box_threshold = args.box_threshold * 5
    else:
        assert len(args.box_threshold) == 5, "box_threshold应该包含5个值或1个值"
        box_threshold = args.box_threshold
        
    print("box_threshold: ", box_threshold)
    
    # 生成mask
    segment_with_text_prompt(
        datadir=args.datadir,
        BOX_TRESHOLD=box_threshold,
        TEXT_TRESHOLD=args.text_threshold,
        ignore_exists=args.ignore_exists,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        force_cpu=args.force_cpu
    )
    
    # 可视化结果
    visualize_masks(args.datadir)

if __name__ == "__main__":
    main() 