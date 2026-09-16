# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 文件说明与导入

"""
lidar_perception.py
Lidar 激光雷达互补感知模块：对 QCar2 激光雷达数据进行网格聚类障碍检测，
并将检测到的障碍物转换为 YOLO 兼容的 Detection 对象，供决策层直接使用。

参考实现：8_radar/0/QCar2_lidar_point_cloud.py 中的障碍检测算法。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pal.products.qcar import QCarLidar

try:
    from perception_yolo import Detection
except ImportError:
    Detection = None

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


#region : 配置参数

@dataclass
class LidarConfig:
    """Lidar 感知参数，方便现场标定。"""
    numMeasurements: int = 1000
    measurementMode: int = 2
    interpolationMode: int = 0
    detectionRadius: float = 4.0
    gridResolution: float = 0.05
    minClusterPoints: int = 12
    minDistance: float = 0.15
    validAngleHalf: float = np.deg2rad(75)
    maxRange: float = 5.0
    lateralScale: float = 0.32
    virtualWidth: int = 640
    virtualHeight: int = 480


DEFAULT_LIDAR_CONFIG = LidarConfig()

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


#region : 障碍检测结果

@dataclass
class LidarObstacle:
    """单个 lidar 障碍物（车身坐标系，x=前，y=左）。"""
    x: float
    y: float
    distance: float
    bearingRad: float
    clusterSize: int = 0
    maxIntensityInCluster: float = 0.0


@dataclass
class LidarFrame:
    """一帧 lidar 数据处理结果。"""
    obstacles: List[LidarObstacle] = field(default_factory=list)
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timestamp: float = 0.0

    @property
    def obstacleCount(self) -> int:
        return len(self.obstacles)

    def describe(self) -> str:
        items = []
        for i, o in enumerate(self.obstacles, start=1):
            items.append('O{}: {:.1f}m @{:.0f}deg'.format(
                i, o.distance, np.degrees(o.bearingRad)))
        return '; '.join(items) if items else 'no obstacle'

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : LidarProcessor 主类

class LidarProcessor:
    """QCar2 激光雷达感知处理器。"""

    def __init__(self, config: Optional[LidarConfig] = None):
        self.cfg = config if config is not None else DEFAULT_LIDAR_CONFIG
        self._lidar: Optional[QCarLidar] = None
        self._distances: np.ndarray = np.zeros(0)
        self._angles: np.ndarray = np.zeros(0)

    def open(self) -> bool:
        try:
            self._lidar = QCarLidar(
                numMeasurements=self.cfg.numMeasurements,
                rangingDistanceMode=self.cfg.measurementMode,
                interpolationMode=self.cfg.interpolationMode,
            )
            return True
        except Exception as e:
            print('[Lidar] init failed:', e)
            self._lidar = None
            return False

    def terminate(self) -> None:
        if self._lidar is not None:
            try:
                self._lidar.terminate()
            except Exception:
                pass
            self._lidar = None

    @property
    def isReady(self) -> bool:
        return self._lidar is not None

    def read(self) -> bool:
        if self._lidar is None:
            return False
        try:
            self._lidar.read()
            self._distances = np.array(self._lidar.distances)
            self._angles = np.array(self._lidar.angles)
            return True
        except Exception:
            return False

    def getCartesian(self) -> tuple:
        if self._distances.size == 0:
            return np.zeros(0), np.zeros(0)
        anglesCar = self._wrap_2pi(2.5 * np.pi - self._angles)
        valid = (self._distances > self.cfg.minDistance) & (
            self._distances < self.cfg.detectionRadius)
        valid &= (np.abs(anglesCar - np.pi) <= self.cfg.validAngleHalf)
        if not np.any(valid):
            return np.zeros(0), np.zeros(0)
        d = self._distances[valid]
        a = anglesCar[valid]
        return d * np.cos(a), d * np.sin(a)

    # ---------------------------------------------------------------- 障碍检测（网格聚类）
    def detect(self) -> LidarFrame:
        cfg = self.cfg
        obstacles: List[LidarObstacle] = []
        distances = self._distances
        angles = self._angles

        if distances.size == 0:
            return LidarFrame(obstacles=obstacles, distances=distances,
                              angles=angles)

        anglesCar = self._wrap_2pi(2.5 * np.pi - angles)
        valid = (distances > cfg.minDistance) & (distances < cfg.detectionRadius)
        valid &= (np.abs(anglesCar - np.pi) <= cfg.validAngleHalf)
        if not np.any(valid):
            return LidarFrame(obstacles=obstacles, distances=distances,
                              angles=angles)

        dValid = distances[valid]
        aValid = anglesCar[valid]
        xValid = dValid * np.cos(aValid)
        yValid = dValid * np.sin(aValid)

        cellMap = {}
        for k in range(len(xValid)):
            cx = int(np.floor(xValid[k] / cfg.gridResolution))
            cy = int(np.floor(yValid[k] / cfg.gridResolution))
            cellMap.setdefault((cx, cy), []).append(k)

        visited = set()
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        for key in list(cellMap.keys()):
            if key in visited:
                continue
            stack = [key]
            compCells = []
            visited.add(key)
            while stack:
                c = stack.pop()
                compCells.append(c)
                for d in nbrs:
                    nb = (c[0]+d[0], c[1]+d[1])
                    if nb in cellMap and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

            indices = []
            for c in compCells:
                indices.extend(cellMap.get(c, []))

            if len(indices) >= cfg.minClusterPoints:
                cx = float(np.mean(xValid[indices]))
                cy = float(np.mean(yValid[indices]))
                obstacles.append(LidarObstacle(
                    x=cx, y=cy,
                    distance=float(np.hypot(cx, cy)),
                    bearingRad=float(np.arctan2(cy, cx)),
                    clusterSize=len(indices),
                ))

        return LidarFrame(obstacles=obstacles, distances=distances,
                          angles=angles)
# ---------------------------------------------------------------- 转换为 YOLO Detection 列表
    def to_detections(self, obstacles: List[LidarObstacle]) -> list:
        """将 lidar 障碍物列表转换为 YOLO 兼容的 Detection 对象列表。
        映射策略：lateralNorm=-y*scale, bottomYNorm=1-dist/maxRange, area~clusterSize/(dist+offset)。"""
        if Detection is None:
            return []
        cfg = self.cfg
        dets = []
        for o in obstacles:
            lateralNorm = -o.y * cfg.lateralScale
            cxNorm = float(np.clip(0.5 + lateralNorm, 0.02, 0.98))
            byNorm = float(np.clip(1.0 - o.distance / cfg.maxRange, 0.05, 0.98))
            cyNorm = float(np.clip(byNorm * 0.85, 0.02, 0.95))
            area = float(np.clip(2.5 / (o.distance + 0.3), 0.02, 0.95))
            conf = float(np.clip(0.85 * o.clusterSize / 30.0, 0.55, 0.99))
            bw, bh = 0.03, 0.04
            dets.append(Detection(
                labelId=4, label='People', confidence=conf,
                x1=max(0, int((cxNorm - bw) * cfg.virtualWidth)),
                y1=max(0, int((cyNorm - bh) * cfg.virtualHeight)),
                x2=min(cfg.virtualWidth, int((cxNorm + bw) * cfg.virtualWidth)),
                y2=min(cfg.virtualHeight, int(byNorm * cfg.virtualHeight)),
                areaPercent=area, centerXNorm=cxNorm,
                centerYNorm=cyNorm, bottomYNorm=byNorm,
                imageWidth=cfg.virtualWidth, imageHeight=cfg.virtualHeight,
            ))
        return dets

    @staticmethod
    def _wrap_2pi(angles: np.ndarray) -> np.ndarray:
        return np.mod(angles, 2 * np.pi)

#endregion