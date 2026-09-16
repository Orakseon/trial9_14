# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 文件说明与导入

"""
perception_yolo.py
识别层：基于 YOLOv11 的交通要素感知模块（结题综合演示 - 识别层）。

功能：
    1. 加载 YOLOv11 权重（默认 yolov11s.pt，与 5_factors/sd 中的模型一致）
    2. 对车头相机（QCar RealSense D435 的 RGB 流）图像执行推理，输出交通要素列表
    3. 将检测框几何信息归一化（横向偏移、中心纵向位置、底边纵向位置、面积占比），
       供决策层判定“目标是否位于本车前方走廊内 / 是否已足够近”
    4. 在图像上绘制检测框、类别、置信度与面积占比，便于演示与调试

交通要素类别（与 yolov11s.pt 训练标签顺序一致，顺序不可更改）：
    0 Cone 交通锥   1 Cow 奶牛        2 Crosswalk 人行横道  3 GREEN 绿灯
    4 People 行人   5 RED 红灯        6 Stop Line 停止线    7 Stop Sign 停止标志
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:                                        # 可选依赖：用于在图像上渲染中文（ultralytics 已依赖 Pillow）
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:                           # pragma: no cover - 环境缺少 Pillow 时退化为英文渲染
    _PIL_AVAILABLE = False

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 交通要素类别定义

# 与 yolov11s.pt 训练标签顺序严格一一对应，禁止调整顺序
YOLO_CLASS_NAMES: Tuple[str, ...] = (
    'Cone',        # 0 交通锥
    'Cow',         # 1 奶牛
    'Crosswalk',   # 2 人行横道
    'GREEN',       # 3 绿灯
    'People',      # 4 行人
    'RED',         # 5 红灯
    'Stop Line',   # 6 停止线
    'Stop Sign',   # 7 停止标志
)

# 类别别名表：把模型 names 中的各种写法归一化到标准名称，避免因大小写/空格差异导致匹配失败
_CLASS_ALIASES: Dict[str, str] = {
    'cone': 'Cone', 'traffic cone': 'Cone', 'trafficcone': 'Cone',
    'cow': 'Cow',
    'crosswalk': 'Crosswalk', 'cross walk': 'Crosswalk', 'zebra crossing': 'Crosswalk',
    'green': 'GREEN', 'green light': 'GREEN', 'greenlight': 'GREEN',
    'people': 'People', 'person': 'People', 'pedestrian': 'People',
    'red': 'RED', 'red light': 'RED', 'redlight': 'RED',
    'stop line': 'Stop Line', 'stopline': 'Stop Line',
    'stop sign': 'Stop Sign', 'stopsign': 'Stop Sign',
}

# 各类别显示颜色（BGR），顺序与 YOLO_CLASS_NAMES 一致
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    'Cone': (50, 50, 255),
    'Cow': (0, 204, 0),
    'Crosswalk': (194, 153, 255),
    'GREEN': (51, 204, 255),
    'People': (204, 102, 255),
    'RED': (255, 153, 0),
    'Stop Line': (255, 0, 0),
    'Stop Sign': (0, 255, 255),
}


class TrafficLabel:
    """交通要素类别常量，供决策层引用，避免在业务代码中硬编码类别字符串。"""
    CONE = 'Cone'
    COW = 'Cow'
    CROSSWALK = 'Crosswalk'
    GREEN = 'GREEN'
    PEOPLE = 'People'
    RED = 'RED'
    STOP_LINE = 'Stop Line'
    STOP_SIGN = 'Stop Sign'
    ALL: Tuple[str, ...] = YOLO_CLASS_NAMES


def _normalize_label(rawName: str) -> str:
    """把模型给出的类别名归一化到标准名称；无法识别时返回原名并保持可用。"""
    if rawName is None:
        return 'Unknown'
    key = str(rawName).strip().lower()
    return _CLASS_ALIASES.get(key, str(rawName).strip())


def _to_numpy(value) -> np.ndarray:
    """把 torch.Tensor 或 array-like 统一转换为 numpy 数组。"""
    if hasattr(value, 'cpu'):
        return value.cpu().numpy()
    return np.asarray(value)


def _default_model_path(filename: str = 'yolov11s.pt') -> str:
    """按优先级搜索 YOLO 权重文件，保证本目录或相邻参考目录下都能直接运行。"""
    thisDir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(thisDir, filename),
        os.path.join(thisDir, '..', '5_factors', 'sd', filename),
        os.path.join(thisDir, '..', '5_factors', 'cache', filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        '未找到 YOLO 权重文件 {}，已搜索：\n  {}'.format(filename, '\n  '.join(candidates))
    )

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 检测结果数据结构

@dataclass
class Detection:
    """单个交通要素检测结果（含决策层所需的归一化几何度量）。"""
    labelId: int            # 模型输出的类别 ID（与 YOLO_CLASS_NAMES 对应）
    label: str              # 归一化后的类别名称
    confidence: float       # 置信度
    x1: int                 # 检测框左边界（像素）
    y1: int                 # 检测框上边界（像素）
    x2: int                 # 检测框右边界（像素）
    y2: int                 # 检测框下边界（像素）
    areaPercent: float      # 检测框面积占整幅图像的百分比
    centerXNorm: float      # 检测框中心横向归一化坐标（0 左边界，1 右边界）
    centerYNorm: float      # 检测框中心纵向归一化坐标（0 上边界，1 下边界）
    bottomYNorm: float      # 检测框底边归一化纵坐标（越大表示越靠近本车）
    imageWidth: int = 0
    imageHeight: int = 0

    @property
    def lateralErrorNorm(self) -> float:
        """目标横向偏离画面中心的程度（>0 表示偏右，<0 表示偏左）。"""
        return self.centerXNorm - 0.5

    @property
    def boxArea(self) -> int:
        """检测框像素面积。"""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def isInCorridor(self, halfWidthNorm: float = 0.32) -> bool:
        """目标中心是否落在本车前方走廊内（判断是否会阻挡行驶轨迹）。"""
        return abs(self.lateralErrorNorm) <= halfWidthNorm

    def isNear(self, bottomYNorm: float, areaPercent: float) -> bool:
        """目标是否已经足够近：底边越过纵向阈值，或面积占比超过阈值。"""
        return (self.bottomYNorm >= bottomYNorm) or (self.areaPercent >= areaPercent)

    def describe(self) -> str:
        """生成用于日志/HUD 的可读描述。"""
        return '{}({:.0%}) 面积{:.2f}% 横向{:+.2f} 底边{:.2f}'.format(
            self.label, self.confidence, self.areaPercent,
            self.lateralErrorNorm, self.bottomYNorm)


@dataclass
class DetectionFrame:
    """一帧识别结果：检测列表 + 带标注图像 + 时间戳与推理耗时等元信息。"""
    detections: List[Detection]
    annotated: np.ndarray
    timestamp: float          # time.monotonic() 时间戳，供决策层判断数据是否过期
    inferenceTime: float      # 本帧 YOLO 推理耗时（秒）
    frameWidth: int
    frameHeight: int
    frameIndex: int = 0

    def byLabel(self, label: str) -> List[Detection]:
        """取出指定类别的全部检测（按面积占比降序，最近的在前）。"""
        return [d for d in self.detections if d.label == label]

    def nearest(self, label: str) -> Optional[Detection]:
        """指定类别中最近的检测（面积占比最大者）。"""
        items = self.byLabel(label)
        return items[0] if items else None

    def labels(self) -> List[str]:
        """本帧出现的所有类别名称。"""
        return sorted({d.label for d in self.detections})

    def countByLabel(self) -> Dict[str, int]:
        """各类别检测数量统计。"""
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        return counts

    def has(self, label: str) -> bool:
        """本帧是否检测到指定类别。"""
        return any(d.label == label for d in self.detections)

    @property
    def isEmpty(self) -> bool:
        return not self.detections

    def isValid(self, maxAge: float, now: Optional[float] = None) -> bool:
        """判断该帧是否仍然有效（决策层据此识别“感知失效”）。"""
        current = time.monotonic() if now is None else now
        return (current - self.timestamp) <= maxAge

    def summaryText(self) -> str:
        """本帧检测摘要，用于 HUD 显示。"""
        if self.isEmpty:
            return '无交通要素'
        counts = self.countByLabel()
        return ', '.join('{} x{}'.format(name, counts[name]) for name in sorted(counts))

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : YOLO 识别模块

class YoloTrafficPerception:
    """YOLOv11 交通要素识别器：封装模型加载、推理与结果可视化。"""

    def __init__(
            self,
            modelPath: Optional[str] = None,
            confThreshold: float = 0.35,
            iouThreshold: float = 0.5,
            imageSize: int = 640,
            device: Optional[str] = None,
            maxDetections: int = 30,
            drawAnnotations: bool = True,
    ):
        """
        参数说明：
            modelPath       权重文件路径，缺省时自动在本目录/相邻参考目录中搜索 yolov11s.pt
            confThreshold   置信度阈值（低于该值的检测被丢弃）
            iouThreshold    NMS 交并比阈值
            imageSize       YOLO 推理输入尺寸（CPU 推理时可适当降低以提升帧率）
            device          推理设备，None 表示自动选择（'cpu' / '0' 等）
            maxDetections   单帧最大检测数量
            drawAnnotations 是否在输出图像上绘制检测框
        """
        self.modelPath = modelPath if modelPath else _default_model_path()
        self.confThreshold = float(confThreshold)
        self.iouThreshold = float(iouThreshold)
        self.imageSize = int(imageSize)
        self.device = device
        self.maxDetections = int(maxDetections)
        self.drawAnnotations = bool(drawAnnotations)

        # 加载权重
        self.model = YOLO(self.modelPath)

        # 读取模型自带类别名并归一化，保证“标签 ID → 类别名称”始终与权重一致
        rawNames = getattr(self.model, 'names', None) or {}
        if isinstance(rawNames, dict):
            self.rawClassNames = [str(rawNames[key]) for key in sorted(rawNames)]
        else:
            self.rawClassNames = [str(name) for name in rawNames]
        self.classNames = [_normalize_label(name) for name in self.rawClassNames]
        if not self.classNames:
            self.classNames = list(YOLO_CLASS_NAMES)
            self.rawClassNames = list(YOLO_CLASS_NAMES)

        self.frameIndex = 0
        self.lastFrame: Optional[DetectionFrame] = None

    def labelOf(self, labelId: int) -> str:
        """类别 ID → 标准类别名称（带边界保护，避免索引越界）。"""
        if 0 <= labelId < len(self.classNames):
            return self.classNames[labelId]
        return 'Unknown-{}'.format(labelId)

    def warmUp(self, imageSize: int = 320) -> float:
        """用空白图预热模型（首次推理包含一次性加载开销），返回预热耗时（秒）。"""
        dummy = np.zeros((imageSize, imageSize, 3), dtype=np.uint8)
        startTime = time.perf_counter()
        self.model.predict(
            source=dummy,
            imgsz=self.imageSize,
            device=self.device,
            verbose=False,
            save=False,
        )
        return time.perf_counter() - startTime

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    def detect(self, frame: np.ndarray, annotate: Optional[bool] = None) -> DetectionFrame:
        """
        对一帧 BGR 图像执行推理。

        参数：
            frame     车头相机图像（不做 resize，由 ultralytics 内部完成 letterbox 缩放）
            annotate  是否绘制检测框，None 表示使用构造参数
        返回：
            DetectionFrame 对象
        """
        if frame is None:
            raise ValueError('detect() 收到空图像，请检查相机数据流。')
        doAnnotate = self.drawAnnotations if annotate is None else bool(annotate)
        startTime = time.perf_counter()
        height, width = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.confThreshold,
            iou=self.iouThreshold,
            imgsz=self.imageSize,
            max_det=self.maxDetections,
            device=self.device,
            verbose=False,
            save=False,
        )

        detections: List[Detection] = []
        boxes = None
        if results and getattr(results[0], 'boxes', None) is not None:
            boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = _to_numpy(boxes.xyxy)
            classIds = _to_numpy(boxes.cls).astype(int)
            confidences = _to_numpy(boxes.conf).astype(float)
            # 逐框解析：类别 ID 与坐标一一对应，drawing/统计都基于同一索引，避免标签错位
            for index in range(len(classIds)):
                x1 = int(np.clip(xyxy[index][0], 0, width - 1))
                y1 = int(np.clip(xyxy[index][1], 0, height - 1))
                x2 = int(np.clip(xyxy[index][2], 0, width - 1))
                y2 = int(np.clip(xyxy[index][3], 0, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                labelId = int(classIds[index])
                areaPercent = (x2 - x1) * (y2 - y1) / float(width * height) * 100.0
                detections.append(Detection(
                    labelId=labelId,
                    label=self.labelOf(labelId),
                    confidence=float(confidences[index]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    areaPercent=round(areaPercent, 3),
                    centerXNorm=(x1 + x2) / 2.0 / width,
                    centerYNorm=(y1 + y2) / 2.0 / height,
                    bottomYNorm=y2 / float(height),
                    imageWidth=width,
                    imageHeight=height,
                ))

        # 颜色二次校验：过滤 YOLO 将“信号灯背面”或“远处绿灯”误判为 RED 的情况
        detections = [d for d in detections
                      if d.label != 'RED' or self._verifyRedLight(frame, d)]

        # 动态障碍（行人/奶牛）二次过滤：路面/路肩常被低置信度误判，增加宽高比与置信度双重校验
        detections = [d for d in detections if not (
            d.label in ('People', 'Cow') and
            (d.confidence < 0.60 or                              # 对 People/Cow 提高置信度门槛
             (d.x2 - d.x1) / max(d.y2 - d.y1, 1) > 2.5)         # 过宽的框不像是行人/奶牛（路面纹理）
        )]

        # 按面积占比降序排列：最近的交通要素排在最前，便于决策层直接取用
        detections.sort(key=lambda item: item.areaPercent, reverse=True)

        annotated = frame.copy()
        if doAnnotate:
            for det in detections:
                self._drawDetection(annotated, det)

        self.frameIndex += 1
        detectionFrame = DetectionFrame(
            detections=detections,
            annotated=annotated,
            timestamp=time.monotonic(),
            inferenceTime=time.perf_counter() - startTime,
            frameWidth=width,
            frameHeight=height,
            frameIndex=self.frameIndex,
        )
        self.lastFrame = detectionFrame
        return detectionFrame

    @staticmethod
    def _drawDetection(image: np.ndarray, det: Detection) -> None:
        """在图像上绘制单个检测框与标签（类别 + 置信度 + 面积占比）。"""
        color = CLASS_COLORS.get(det.label, (0, 255, 0))
        cv2.rectangle(image, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        text = '{} {:.0%} {:.1f}%'.format(det.label, det.confidence, det.areaPercent)
        (textWidth, textHeight), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(det.y1 - textHeight - baseline - 4, 0)
        cv2.rectangle(image, (det.x1, top),
                      (det.x1 + textWidth + 6, top + textHeight + baseline + 4),
                      color, -1)
        cv2.putText(image, text, (det.x1 + 3, top + textHeight + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _verifyRedLight(image: np.ndarray, det: Detection,
                        minRedRatio: float = 0.08) -> bool:
        """HSV 颜色二次验证：检测框内红色像素占比 ≥ minRedRatio 才认为是真红灯。
        用于过滤 YOLO 将“信号灯背面外壳”或“远处绿灯”误判为 RED 的情况。
        """
        if image is None or det.y2 <= det.y1 or det.x2 <= det.x1:
            return True  # 无法校验时放行，避免因边缘情况漏掉真红灯
        h, w = image.shape[:2]
        x1 = max(0, det.x1)
        y1 = max(0, det.y1)
        x2 = min(w, det.x2)
        y2 = min(h, det.y2)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return True
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 红色在 HSV 中跨越 0°：低红 (0~12°) 和高红 (170~180°)
        maskLow = cv2.inRange(hsv, (0, 50, 60), (12, 255, 255))
        maskHigh = cv2.inRange(hsv, (170, 50, 60), (180, 255, 255))
        redPixels = cv2.countNonZero(maskLow) + cv2.countNonZero(maskHigh)
        redRatio = redPixels / (roi.shape[0] * roi.shape[1])
        return redRatio >= minRedRatio

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 图像叠加显示工具（供主程序在相机画面上显示决策与控制状态）

# 中文字体候选路径（Windows 自带）
_CN_FONT_CANDIDATES: Tuple[str, ...] = (
    r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
    r'C:\Windows\Fonts\simhei.ttf',    # 黑体
    r'C:\Windows\Fonts\simsun.ttc',    # 宋体
)
_CN_FONT_CACHE: Dict[int, object] = {}


def _get_cn_font(fontSizePx: int):
    """按字号获取中文字体（带缓存）；不可用时返回 None。"""
    if not _PIL_AVAILABLE:
        return None
    if fontSizePx in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[fontSizePx]
    font = None
    for fontPath in _CN_FONT_CANDIDATES:
        if os.path.isfile(fontPath):
            try:
                font = ImageFont.truetype(fontPath, int(fontSizePx))
                break
            except Exception:
                font = None
    _CN_FONT_CACHE[fontSizePx] = font
    return font


def draw_status_panel(
        image: np.ndarray,
        lines: Sequence[str],
        origin: Tuple[int, int] = (10, 12),
        fontSizePx: int = 18,
        lineHeight: int = 26,
        alpha: float = 0.55,
) -> np.ndarray:
    """
    在图像左上角绘制半透明状态面板（决策状态、速度、指令等）。

    说明：使用 Pillow + 系统中文字体渲染，可正常显示中文；
         若环境缺少 Pillow，则退化为 OpenCV 内置字体（仅 ASCII 正常显示）。
    """
    if image is None or not lines:
        return image
    x0, y0 = int(origin[0]), int(origin[1])
    font = _get_cn_font(fontSizePx)

    if font is not None:
        pilImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGBA')
        overlay = Image.new('RGBA', pilImage.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        textWidth = max(draw.textlength(text, font=font) for text in lines)
        boxWidth = int(textWidth) + 20
        boxHeight = lineHeight * len(lines) + 12
        draw.rectangle([x0, y0, x0 + boxWidth, y0 + boxHeight],
                       fill=(0, 0, 0, int(255 * float(alpha))))
        for index, text in enumerate(lines):
            draw.text((x0 + 10, y0 + 6 + index * lineHeight), text,
                      font=font, fill=(255, 255, 255, 255))
        composed = Image.alpha_composite(pilImage, overlay).convert('RGB')
        image[:] = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)
        return image

    # 退化路径：OpenCV 内置字体
    scale = max(fontSizePx / 30.0, 0.3)
    (textWidth, textHeight), baseline = cv2.getTextSize(
        max(lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    boxWidth = textWidth + 16
    boxHeight = lineHeight * len(lines) + 12
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + boxWidth, y0 + boxHeight), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    for index, text in enumerate(lines):
        position = (x0 + 8, y0 + 8 + textHeight + index * lineHeight)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_corridor_guide(
        image: np.ndarray,
        corridorHalfWidthNorm: float = 0.32,
        nearBottomYNorm: float = 0.72,
        color: Tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    """
    绘制“前方走廊”与“近距阈值线”辅助线，用于现场标定决策层阈值。

    走廊：以画面中心为中心、宽度为 2*corridorHalfWidthNorm 的竖直带状区域，
          落在其中的目标被认为可能阻挡本车行驶轨迹。
    阈值线：目标检测框底边越过该线时视为“距离已近”。
    """
    if image is None:
        return image
    height, width = image.shape[:2]
    left = int((0.5 - corridorHalfWidthNorm) * width)
    right = int((0.5 + corridorHalfWidthNorm) * width)
    for x in (left, right):
        cv2.line(image, (x, 0), (x, height - 1), color, 1, cv2.LINE_AA)
    cv2.line(image, (0, int(nearBottomYNorm * height)),
             (width - 1, int(nearBottomYNorm * height)), color, 1, cv2.LINE_AA)
    return image

#endregion




