import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import logging

# 导入YOLO相关模块
from ultralytics import YOLO
import torch

# 导入配置
from config import TrainingConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmotionDetector:
    """情绪检测器类 - 为实时检测预留接口"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = 'auto',
        imgsz: int = 640,
        half: Optional[bool] = None,
    ):
        """
        初始化情绪检测器

        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            device: 推理设备 ('auto', 'cpu', 'cuda', '0', '1', etc.)
            imgsz: 推理输入尺寸（单边），交给 YOLO 内置 letterbox 处理
            half: 是否使用半精度（仅 CUDA 有效）；None 表示根据设备自动决定
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        self.half = half

        # 情绪标签映射（作为后备，优先使用 self.model.names）
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise',
        }

        # 情绪对应的颜色 (BGR格式)
        self.emotion_colors = {
            'angry': (0, 0, 255),  # 红色
            'disgust': (0, 128, 0),  # 深绿色
            'fear': (128, 0, 128),  # 紫色
            'happy': (0, 255, 255),  # 黄色
            'neutral': (128, 128, 128),  # 灰色
            'sad': (255, 0, 0),  # 蓝色
            'surprise': (0, 165, 255),  # 橙色
        }

        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载YOLO模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置设备
            if self.device == 'auto':
                self.device = '0' if torch.cuda.is_available() else 'cpu'

            # 将模型移动到目标设备
            try:
                self.model.to(self.device)
            except Exception:
                # 某些后端不支持 .to，退回通过参数控制
                pass

            # half 默认根据设备决定
            if self.half is None:
                self.half = bool(torch.cuda.is_available() and (str(self.device) != 'cpu'))
            
            logger.info(f"模型加载成功，使用设备: {self.device}")
            # 优先使用模型自带的类名
            names = getattr(self.model, 'names', None)
            if isinstance(names, (list, tuple, dict)) and len(names) > 0:
                logger.info(f"模型类别数: {len(names)}，使用模型自带类别名")
            else:
                logger.info(f"模型类别数: {len(self.emotion_labels)} (使用内置后备标签)")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
            
        
    def detect_emotions(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人脸情绪
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个结果包含边界框、情绪标签和置信度
        """
        try:
            h, w = image.shape[:2]

            # 直接将原图交给 YOLO，使用其内置 letterbox
            results = self.model(
                image,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                half=self.half,
                verbose=False,
            )

            # 解析结果
            detections = []
            names = getattr(self.model, 'names', None)

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                clses = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i]

                    # 裁剪到图像边界并取整
                    x1 = int(round(max(0, min(x1, w - 1))))
                    y1 = int(round(max(0, min(y1, h - 1))))
                    x2 = int(round(max(0, min(x2, w - 1))))
                    y2 = int(round(max(0, min(y2, h - 1))))

                    class_id = int(clses[i])
                    confidence = float(confs[i])

                    # 优先用模型自带类名，否则用后备映射
                    if isinstance(names, dict):
                        emotion = names.get(class_id, self.emotion_labels.get(class_id, 'unknown'))
                    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                        emotion = names[class_id]
                    else:
                        emotion = self.emotion_labels.get(class_id, 'unknown')

                    detections.append(
                        {
                            'bbox': (x1, y1, x2, y2),
                            'emotion': str(emotion),
                            'confidence': confidence,
                            'class_id': class_id,
                        }
                    )

            return detections

        except Exception as e:
            logger.error(f"情绪检测失败: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            绘制了检测结果的图像
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            emotion = detection['emotion']
            confidence = detection['confidence']
            
            # 获取颜色
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = f"{emotion}: {confidence:.2f}"
            
            # 计算标签背景尺寸
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 选择标签绘制位置（优先在框上方，否则放到框内）
            top_y = y1 - label_h - baseline - 6
            if top_y < 0:
                # 放到框内顶部
                box_bg_top = y1
                box_bg_bottom = min(y1 + label_h + baseline + 6, result_image.shape[0] - 1)
                text_org = (x1, min(y1 + label_h + 2, result_image.shape[0] - 1))
            else:
                box_bg_top = max(0, top_y)
                box_bg_bottom = y1
                text_org = (x1, max(0, y1 - baseline - 2))

            box_bg_right = min(x1 + label_w + 2, result_image.shape[1] - 1)

            # 绘制标签背景
            cv2.rectangle(
                result_image,
                (x1, box_bg_top),
                (box_bg_right, box_bg_bottom),
                color,
                -1,
            )

            # 绘制标签文本
            cv2.putText(
                result_image,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        
        return result_image
    
    def process_image_file(self, input_path: str, output_path: str) -> bool:
        """
        处理单张图片文件
        
        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            
        Returns:
            处理是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"无法读取图像: {input_path}")
                return False
            
            logger.info(f"处理图像: {input_path}")
            logger.info(f"图像尺寸: {image.shape}")
            
            # 检测情绪
            detections = self.detect_emotions(image)
            logger.info(f"检测到 {len(detections)} 个人脸")
            
            # 绘制检测结果
            result_image = self.draw_detections(image, detections)
            
            # 保存结果
            success = cv2.imwrite(output_path, result_image)
            if success:
                logger.info(f"结果已保存到: {output_path}")
                
                # 打印检测结果
                for i, detection in enumerate(detections):
                    emotion = detection['emotion']
                    confidence = detection['confidence']
                    logger.info(f"  人脸 {i+1}: {emotion} (置信度: {confidence:.3f})")
            else:
                logger.error(f"保存结果失败: {output_path}")
                
            return success
            
        except Exception as e:
            logger.error(f"处理图像失败: {e}")
            return False
    
    # 预留的实时检测接口
    def detect_from_camera(
        self,
        camera_id: int = 0,
        show: bool = True,
        save_path: Optional[str] = None,
        frame_stride: int = 1,
    ) -> bool:
        """
        从摄像头进行实时情绪检测

        Args:
            camera_id: 摄像头ID
            show: 是否显示窗口
            save_path: 若提供，则将结果保存为视频文件（.mp4 等）
            frame_stride: 帧间隔（每隔多少帧处理一次）
        Returns:
            是否成功运行
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"无法打开摄像头: {camera_id}")
            return False

        # 读取 FPS（有些摄像头可能返回 0 或 NaN）
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0

        writer = None
        frame_idx = 0
        success = True
        window_name = f"Emotion Camera {camera_id}"

        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info("开始摄像头实时检测，按 'q' 或 ESC 退出...")
        try:
            # 上一次有效检测结果与复用预算（最多复用 2*N 次）
            last_dets: Optional[List[Dict]] = None
            reuse_budget: int = 0
            max_reuse_factor = 2 * max(1, frame_stride)

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("读取摄像头帧失败，可能已断开")
                    success = False
                    break

                process_this = (frame_stride <= 1) or (frame_idx % frame_stride == 0)
                vis = frame
                if process_this:
                    dets = self.detect_emotions(frame)
                    if dets:
                        # 有新结果，更新并重置复用预算
                        last_dets = dets
                        reuse_budget = max_reuse_factor
                        vis = self.draw_detections(frame, dets)
                    else:
                        # 当前推理为空，尝试复用旧结果
                        if last_dets and reuse_budget > 0:
                            vis = self.draw_detections(frame, last_dets)
                            reuse_budget -= 1
                        # 否则保持原帧（不绘制）
                else:
                    # 跳帧时尝试复用旧结果
                    if last_dets and reuse_budget > 0:
                        vis = self.draw_detections(frame, last_dets)
                        reuse_budget -= 1

                # 初始化视频写入器（懒加载）
                if save_path and writer is None:
                    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        logger.error(f"无法打开视频写入器: {save_path}")
                        writer = None

                if writer is not None:
                    writer.write(vis)

                if show:
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):  # ESC 或 q
                        break

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyWindow(window_name)

        if save_path:
            logger.info(f"摄像头检测结束，结果保存至: {save_path}")
        else:
            logger.info("摄像头检测结束")

        return success

    def detect_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        frame_stride: int = 1,
    ) -> bool:
        """
        从视频文件进行情绪检测

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径；为 None 时只显示/不保存
            show: 是否显示窗口
            frame_stride: 帧间隔（每隔多少帧处理一次）
        Returns:
            是否成功运行
        """
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.error(f"无法打开视频写入器: {output_path}")
                writer = None

        frame_idx = 0
        window_name = f"Emotion Video"
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info(f"开始处理视频: {video_path}，按 'q' 或 ESC 退出预览...")
        success = True
        try:
            # 上一次有效检测结果与复用预算（最多复用 2*N 次）
            last_dets: Optional[List[Dict]] = None
            reuse_budget: int = 0
            max_reuse_factor = 2 * max(1, frame_stride)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                process_this = (frame_stride <= 1) or (frame_idx % frame_stride == 0)
                vis = frame
                if process_this:
                    dets = self.detect_emotions(frame)
                    if dets:
                        last_dets = dets
                        reuse_budget = max_reuse_factor
                        vis = self.draw_detections(frame, dets)
                    else:
                        if last_dets and reuse_budget > 0:
                            vis = self.draw_detections(frame, last_dets)
                            reuse_budget -= 1
                else:
                    if last_dets and reuse_budget > 0:
                        vis = self.draw_detections(frame, last_dets)
                        reuse_budget -= 1

                if writer is not None:
                    # 若视频属性不可用，按当前帧尺寸写入
                    if vis.shape[1] != width or vis.shape[0] != height:
                        width, height = vis.shape[1], vis.shape[0]
                        writer.release()
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    writer.write(vis)

                if show:
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyWindow(window_name)

        if output_path:
            logger.info(f"视频处理完成，结果保存至: {output_path}")
        else:
            logger.info("视频处理完成")

        return success


def get_model_path() -> str:
    """获取最佳模型路径"""
    # 优先使用训练结果中的best.pt
    best_model_path = "./emotion_detection/train_v1/weights/best.pt"
    if os.path.exists(best_model_path):
        return best_model_path
    
    # 备用方案：使用根目录的best.pt
    fallback_path = "./best.pt" 
    if os.path.exists(fallback_path):
        return fallback_path
    
    raise FileNotFoundError("未找到训练好的模型文件")


def process_input_directory(detector: EmotionDetector, 
                          input_dir: str = "./input",
                          output_dir: str = "./output") -> None:
    """
    处理输入目录中的所有图片
    
    Args:
        detector: 情绪检测器实例
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取输入目录中的所有图片文件
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"在 {input_dir} 中未找到图片文件")
        logger.info("请将要检测的图片放入 ./input 目录")
        return
    
    logger.info(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每张图片
    success_count = 0
    for image_file in image_files:
        input_path = str(image_file)
        output_filename = f"detected_{image_file.name}"
        output_path = os.path.join(output_dir, output_filename)
        
        if detector.process_image_file(input_path, output_path):
            success_count += 1
        
        print("-" * 50)
    
    logger.info(f"处理完成! 成功处理 {success_count}/{len(image_files)} 张图片")
    logger.info(f"结果保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='人脸情绪检测脚本')
    parser.add_argument('--input', '-i', type=str, default='./input',
                       help='输入目录路径 (默认: ./input)')
    parser.add_argument('--output', '-o', type=str, default='./output', 
                       help='输出目录路径 (默认: ./output)')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='模型文件路径 (默认: 自动查找最佳模型)')
    parser.add_argument('--conf', '-c', type=float, default=0.3,
                       help='置信度阈值 (默认: 0.3)')
    parser.add_argument('--device', '-d', type=str, default='auto',
                       help='推理设备 (默认: auto)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='推理输入尺寸 (默认: 640)')
    parser.add_argument('--half', action='store_true',
                       help='使用半精度推理 (仅CUDA有效)')

    # 实时/视频参数
    parser.add_argument('--camera', action='store_true',
                        help='使用摄像头进行实时检测')
    parser.add_argument('--cam-id', type=int, default=0,
                        help='摄像头ID (默认: 0)')
    parser.add_argument('--video', type=str, default=None,
                        help='输入视频文件路径 (设置该参数时处理视频)')
    parser.add_argument('--show', action='store_true',
                        help='显示实时窗口/视频预览')
    parser.add_argument('--save-out', type=str, default=None,
                        help='输出视频文件路径（若不提供则不保存视频）')
    parser.add_argument('--frame-stride', type=int, default=2,
                        help='帧间隔：每隔多少帧推理一次 (默认: 2)')
    
    args = parser.parse_args()
    
    try:
        # 获取模型路径
        model_path = args.model if args.model else get_model_path()
        logger.info(f"使用模型: {model_path}")
        
        # 创建检测器
        detector = EmotionDetector(
            model_path=model_path,
            conf_threshold=args.conf,
            device=args.device,
            imgsz=args.imgsz,
            half=args.half if args.half else None,
        )
        
        # 路由到不同的来源
        if args.camera:
            # 摄像头
            save_path = args.save_out
            if save_path is None:
                # 默认不保存；如需默认保存，可取消下面注释
                # save_path = os.path.join(args.output, 'camera_out.mp4')
                pass
            detector.detect_from_camera(
                camera_id=args.cam_id,
                show=args.show,
                save_path=save_path,
                frame_stride=args.frame_stride,
            )
        elif args.video:
            # 视频文件
            save_path = args.save_out
            if save_path is None:
                base = os.path.basename(args.video)
                name, _ = os.path.splitext(base)
                os.makedirs(args.output, exist_ok=True)
                save_path = os.path.join(args.output, f"detected_{name}.mp4")
            detector.detect_from_video(
                video_path=args.video,
                output_path=save_path,
                show=args.show,
                frame_stride=args.frame_stride,
            )
        else:
            # 处理输入目录中的图片
            process_input_directory(detector, args.input, args.output)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
