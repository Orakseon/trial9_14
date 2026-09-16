# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 文件说明与导入

"""
QCar2_YOLO_Decision_Control.py
结题综合演示主程序：YOLO 交通要素识别 + 决策状态机 + QCar 车辆控制。

层次结构：
    perception_yolo.py              识别层：YOLOv11 推理交通要素
                                    （红绿灯 / 行人 / 锥桶 / 停止标志 / 斑马线 / 停止线）
    decision_layer.py               决策层：交通规则状态机（巡航 / 减速 / 停车等待 / 确认安全后绕行）
    control_layer.py                控制层：速度 PI 控制 + Stanley 路径跟踪（支持绕行横向偏移）
    本文件                          主程序：相机与车辆接口、线程调度、加锁数据交换、可视化

运行前提：
    1. 在 QLabs 中已加载包含红绿灯、斑马线行人、锥桶、停止标志的 SDCS 路网场景；
       按任务要求，本程序只搭建“识别-决策-控制”层，不进行场景构建。
    2. 参考路径由 SDCSRoadMap 生成（nodeSequence=[0, 23, 0]），车辆初始位姿取自路网节点；
       若需要程序自动搭建演示场景（斑马线 5 条 + 信号灯 4 个）并把车辆复位到路网起点，
       请把 resetVehiclePoseAtStart 置为 True（将调用本目录的 setup_trial.py；
       若该文件不存在，则回退到 0/qlabs_setup_task01.py 只做车辆复位）。
       也可以先单独运行 python setup_trial.py 搭好场景，再运行本程序。
    3. 实车（IS_PHYSICAL_QCAR 为 True）运行时需按提示完成 GPS 标定。

线程模型：
    识别线程：按相机帧率读取 RealSense RGB 图像 → YOLO 推理 → 发布最新识别结果（加锁）
    控制线程：100 Hz 读取编码器/GPS → EKF 状态估计 → 决策 → 速度/转向控制 → 写入指令
    主线程：刷新 MultiScope 曲线 + 在相机画面上叠加决策与指令信息
            （cv2 窗口在主线程创建/刷新，避免 GUI 跨线程问题）
"""

import importlib
import os
import signal
import time
from threading import Lock, Thread
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, QCarRealSense, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images

from perception_yolo import (
    DetectionFrame,
    YoloTrafficPerception,
    draw_corridor_guide,
    draw_status_panel,
)
from decision_layer import DecisionParameters, DecisionResult, TrafficDecision
from control_layer import (
    OffsetShaper,
    SpeedController,
    SpeedRampLimiter,
    SteeringController,
)

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 实验参数配置

# ===== 时间参数 =====
tf = 180                     # 演示总时长（秒）
startDelay = 1.0            # 启动延迟，用于滤波器收敛（秒）
controllerUpdateRate = 100  # 控制循环频率（Hz），不建议超过 500
sampleTime = 1 / controllerUpdateRate

# ===== 速度控制器参数（与 0/QCar2_Steering_Control.py 保持一致的量级）=====
K_p = 0.1                   # 比例增益
K_i = 1.0                   # 积分增益
maxThrottle = 0.1           # 油门限幅
integratorLimit = 1.0       # 积分限幅（抗饱和）
stopSpeedThreshold = 0.03   # 停车保持速度阈值（目标速度为 0 且车速低于该值时输出零油门，避免倒退）
accelLimit = 0.45           # 期望速度斜坡：最大加速度（m/s²）
decelLimit = 0.90           # 期望速度斜坡：最大减速度（m/s²）

# ===== 转向控制器参数 =====
enableSteeringControl = True
K_stanley = 1.0             # Stanley 增益
nodeSequence = [0, 23, 0]   # SDCS 路网路径节点序列（闭环路线）
lookAheadDistance = 0.2     # 控制点前视距离（m）
maxSteeringAngleDeg = 30    # 前轮转角限幅（°）
stanleySpeedFloor = 0.10    # Stanley 速度分母下限（m/s），避免停车时转向增益发散
offsetRateLimit = 0.60      # 绕行横向偏移变化率上限（m/s）

# ===== 油门缩放（转弯/锥桶避让时降油门，避免车速过冲）=====
turnThrottleScale = 0.90          # 转弯时油门缩放（直线巡航的90%）
turnSteeringThresholdRad = 0.087  # 转弯判定前轮转角阈值（rad，≈5°）
coneAvoidThrottleScale = 0.80     # 锥桶避让时油门额外缩放（在减速限速基础上再降）

# ===== 连续转向降速（直道巡航 0.40，连续打方向时降至 0.30）=====
sustainedSteeringFrames = 10     # 连续转向帧数阈值（100 Hz 下 ≈ 0.1 s）
steeringCruiseSpeed = 0.35       # 连续转向时的巡航速度（m/s）

# ===== 识别（YOLO）参数 =====
yoloModelPath = None        # None 表示自动搜索本目录/5_factors 下的 yolov11s.pt
confThreshold = 0.50        # 置信度阈值（0.45→0.50，过滤路面/路肩→People/Cow 误判）
iouThreshold = 0.5          # NMS 交并比阈值
yoloImageSize = 480         # YOLO 推理尺寸（降低到 480 提升帧率；仿真红绿灯较大，精度损失可控）
yoloDevice = None           # None 自动选择；CPU 推理可显式写 'cpu'
maxDetections = 30          # 单帧最大检测数量
cameraWidth = 640           # 车头相机 RGB 分辨率（仿真下固定 640x480）
cameraHeight = 480
cameraFrameRate = 30
warmUpModel = True          # 启动时用空白图预热，避免第一帧卡顿
adaptiveDetectPeriod = True # 依据实测推理耗时自适应识别周期
detectPeriodMin = 0.05      # 识别周期下限（秒）
detectPeriodMax = 0.33      # 识别周期上限（秒），确保最低约 3 Hz

# ===== 决策参数 =====
# 各阈值以“面积占比 / 归一化坐标”表达，与相机分辨率无关；
# 现场标定时可打开可视化画面上的走廊辅助线（drawCorridorGuide）对照调整。
decisionParams = DecisionParameters(
    cruiseSpeed=0.45,                 # 巡航速度（直道更快；连续转向时自动降至 steeringCruiseSpeed）
    blindSpeed=0.15,                  # 感知失效时的谨慎速度（改为 0.0 则停车等待）
    crosswalkSpeed=0.20,              # 斑马线/前方交通要素限速
    coneSlowSpeed=0.20,               # 远距锥桶限速（0.30→0.20，更早减速）
    bypassSpeed=0.12,                 # 绕行锥桶限速（0.25→0.12，进一步降低绕行速度）
    detectionTimeout=1.5,             # 识别结果有效期（秒）
    corridorHalfWidth=0.32,           # 前方走廊半宽（画面宽度比例）
    trafficLightCorridorHalfWidth=0.38,
    lightMaxCenterYNorm=0.78,
    redMinAreaPercent=0.10,           # 红灯最小面积占比
    greenMinAreaPercent=0.10,         # 绿灯最小面积占比
    lightConfirmFrames=3,             # 红/绿灯连续确认帧数
    redLostReleaseTime=6.0,           # 红灯消失且未见绿灯的宽限时间
    stopSignMinAreaPercent=0.10,
    stopSignHoldTime=2.5,             # 停止标志强制停车时间
    stopSignResetTime=3.0,            # 停止标志闭锁复位时间
    dynamicStopBottomYNorm=0.62,      # 行人/奶牛“距离已近”的底边阈值
    dynamicStopAreaPercent=0.80,
    clearConfirmTime=1.0,             # 障碍离开走廊后的畅通确认时间（0.60→1.0，加长回正后行驶）
    coneNearBottomYNorm=0.60,         # 锥桶“距离已近”的底边阈值（0.72→0.60，更早触发停车）
    coneStopAreaPercent=0.45,         # 锥桶"距离已近"的面积阈值（0.90→0.45，更容易触发）
    coneObserveTime=1.20,             # 近距锥桶停车观察时间
    bypassOffset=0.35,                # 绕行横向偏移量（m）
    bypassMinClearance=0.18,          # 锥桶横向偏离超过该值视为不挡路
)

# ===== 可视化与记录 =====
showScope = True                      # 是否显示 MultiScope 曲线与轨迹图
showCameraWindow = True               # 是否显示相机识别画面
drawCorridorGuide = True              # 是否在相机画面上绘制走廊/近距辅助线
cameraWindowName = 'QCar2 YOLO Perception'
recordVideo = False                   # 是否录制演示视频（识别画面）
videoPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_record.mp4')
videoFps = 10

# ===== 窗口显示尺寸 =====
# 综合演示（MultiScope）与相机窗口的显示尺寸。
# 相机采集分辨率仿真模式下固定 640x480，显示与之一致，但窗口允许手动缩放。
scopeWindowSize = (1400, 900)
cameraDisplaySize = (640, 480)

# ===== 车辆位姿复位（可选）=====
# True 时程序启动会调用同目录下的 qlabs_setup_task01.py 把车辆复位到路网起点，
# 结束时会关闭 QLabs 连接。若场景由队友另行搭建，保持 False 即可。
resetVehiclePoseAtStart = False

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 初始设置

if enableSteeringControl:
    # 参考路径：SDCS 路网节点序列生成的中心线（不是场景构建，仅几何路径）
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    waypointSequence = None
    initialPose = np.array([0.0, 0.0, 0.0])

# GPS 标定参考位姿
calibrationPose = [float(initialPose[0]), float(initialPose[1]), float(initialPose[2])]

def load_scene_module():
    """
    按优先级加载场景脚本：

        1. 本目录的 setup_trial.py —— 搭建演示场景（斑马线 5 条 + 信号灯 4 个）并生成车辆；
        2. 0/qlabs_setup_task01.py —— 仅生成/复位车辆（旧版示例脚本）。

    返回模块对象；都不可用时返回 None（跳过场景/位姿复位，使用 QLabs 当前场景）。
    """
    for moduleName in ('setup_trial', 'qlabs_setup_task01'):
        try:
            return importlib.import_module(moduleName)
        except Exception as error:
            print('[提示] 场景脚本 {} 不可用：{}'.format(moduleName, error))
    print('[警告] 未找到可用的场景脚本（setup_trial.py / qlabs_setup_task01.py），'
          '跳过场景搭建与车辆位姿复位。')
    return None


sceneModule = None
if not IS_PHYSICAL_QCAR:
    calibrate = False
    if resetVehiclePoseAtStart:
        sceneModule = load_scene_module()
        if sceneModule is not None:
            try:
                sceneModule.setup(
                    initialPosition=[float(initialPose[0]), float(initialPose[1]), 0],
                    initialOrientation=[0, 0, float(initialPose[2])],
                )
            except Exception as error:
                print('[警告] 场景搭建 / 车辆位姿复位失败（可忽略，使用 QLabs 当前场景）：', error)
else:
    calibrate = 'y' in input('是否需要重新标定？(y/n)')

# 安全退出标志：Ctrl+C 或主循环结束时置位，各线程据此退出
global KILL_THREAD
KILL_THREAD = False


def sig_handler(*args):
    """Ctrl+C 中断处理：置位退出标志，交由各线程安全收尾。"""
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 线程安全的数据交换

class SharedState:
    """识别线程 / 控制线程 / 主线程之间的加锁数据交换区。"""

    def __init__(self):
        self.lock = Lock()
        self.detectionFrame: Optional[DetectionFrame] = None   # 最新识别结果
        self.inferenceTime: float = 0.0                        # 最近一帧推理耗时
        self.perceptionPeriod: float = 0.0                     # 当前识别周期
        self.perceptionError: str = ''                         # 识别线程异常信息
        self.decision: Optional[DecisionResult] = None         # 最新决策结果
        self.telemetry: Dict[str, object] = {}                # 最新控制遥测（含状态字符串）
        self.decisionModule: Optional[TrafficDecision] = None  # 决策模块引用（用于结束总结）

    def publishDetection(self, detectionFrame: Optional[DetectionFrame],
                          period: float = 0.0, error: str = '') -> None:
        with self.lock:
            self.detectionFrame = detectionFrame
            self.perceptionPeriod = period
            self.perceptionError = error
            if detectionFrame is not None:
                self.inferenceTime = detectionFrame.inferenceTime
            elif error:
                self.inferenceTime = 0.0

    def readDetection(self) -> Tuple[Optional[DetectionFrame], float, str]:
        with self.lock:
            return self.detectionFrame, self.perceptionPeriod, self.perceptionError

    def publishDecision(self, decision: DecisionResult, telemetry: Dict[str, object]) -> None:
        with self.lock:
            self.decision = decision
            self.telemetry = telemetry

    def readDecision(self) -> Tuple[Optional[DecisionResult], Dict[str, object]]:
        with self.lock:
            return self.decision, dict(self.telemetry)

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 识别线程

def perceptionLoop(shared: SharedState, camera: QCarRealSense,
                   perception: YoloTrafficPerception) -> None:
    """
    识别线程：读取车头相机图像 → YOLO 推理 → 发布最新识别结果。

    说明：
        1. 相机图像按原始分辨率送入 YOLO（不做 resize，避免拉伸导致目标形变）；
        2. 推理耗时较大（尤其 CPU 推理）时自适应降低识别频率，保证控制线程 100 Hz 实时性；
        3. 任何异常都不会中断线程，只把错误信息发布出去，决策层会按“感知失效”处理。
    """
    global KILL_THREAD
    period = detectPeriodMin
    while not KILL_THREAD:
        cycleStart = time.perf_counter()

        # ---- 读取相机 ----
        try:
            timestamp = camera.read_RGB()
        except Exception as error:
            shared.publishDetection(None, period, '相机读取异常：{}'.format(error))
            time.sleep(0.2)
            continue
        if timestamp is None or timestamp < 0:
            time.sleep(0.01)
            continue

        # ---- 推理：先复制图像，避免推理期间相机缓冲被下一帧覆盖 ----
        frame = camera.imageBufferRGB.copy()
        try:
            detectionFrame = perception.detect(frame)
        except Exception as error:
            shared.publishDetection(None, period, 'YOLO 推理异常：{}'.format(error))
            time.sleep(0.1)
            continue

        # ---- 自适应识别周期：取 1.2 倍实测推理耗时，限制在设定区间内 ----
        if adaptiveDetectPeriod:
            period = float(np.clip(detectionFrame.inferenceTime * 1.2,
                                   detectPeriodMin, detectPeriodMax))
        else:
            period = detectPeriodMin

        shared.publishDetection(detectionFrame, period)

        sleepTime = period - (time.perf_counter() - cycleStart)
        if sleepTime > 0:
            time.sleep(sleepTime)

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 控制线程

def controlLoop(shared: SharedState) -> None:
    """控制线程：传感器读取 → EKF 状态估计 → 决策 → 速度/转向控制 → 指令写入 → 可视化采样。"""
    global KILL_THREAD
    u = 0.0
    delta = 0.0
    count = 0
    countMax = max(int(controllerUpdateRate / 10), 1)   # 可视化采样频率 10 Hz
    _steerCount = 0                       # 连续转向帧计数器

    # ---- 控制器初始化 ----
    speedController = SpeedController(
        kp=K_p, ki=K_i, maxThrottle=maxThrottle, integratorLimit=integratorLimit,
        stopSpeedThreshold=stopSpeedThreshold)
    speedRamp = SpeedRampLimiter(
        accelLimit=accelLimit, decelLimit=decelLimit, initialSpeed=0.0)
    offsetShaper = OffsetShaper(rateLimit=offsetRateLimit, initialOffset=0.0)
    steeringController = None
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley,
            cyclic=True,
            maxSteeringAngle=np.deg2rad(maxSteeringAngleDeg),
            speedFloor=stanleySpeedFloor,
        )
    decisionModule = TrafficDecision(params=decisionParams)
    shared.decisionModule = decisionModule

    # ---- QCar 接口初始化 ----
    # 注意：QCar / QCarGPS 的构造函数在连接失败时只会打印错误、不会抛异常，
    # 真正的失败会在随后的 read() 中抛 HILError，因此这里主动包 try/except 兜底。
    qcar = None
    gps = None
    try:
        qcar = QCar(readMode=1, frequency=controllerUpdateRate)
        if enableSteeringControl or calibrate:
            ekf = QCarEKF(x_0=initialPose)
            gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
        else:
            gps = memoryview(b'')
    except Exception as error:
        KILL_THREAD = True
        print('[错误] QCar/GPS 接口初始化失败：{}'.format(error))
        print('       控制线程退出，请确认 QCar2 实时模型与 GPS 服务已启动。')
        return

    try:
        t0 = time.time()
        t = 0.0
        while (t < tf + startDelay) and (not KILL_THREAD):
            # ---- 循环计时 ----
            tp = t
            t = time.time() - t0
            dt = t - tp

            # ---- 传感器读取与状态估计 ----
            try:
                qcar.read()
            except Exception as error:
                print('[错误] QCar 传感器读取失败：{}'.format(error))
                print('       控制线程退出，请确认 QCar2 实时模型正在运行。')
                break
            th = 0.0
            pVehicle = np.zeros(2)
            pControl = np.zeros(2)
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2],
                    ])
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])
                th = float(ekf.x_hat[2, 0])
                pVehicle = np.array([float(ekf.x_hat[0, 0]), float(ekf.x_hat[1, 0])])
                pControl = pVehicle + np.array([np.cos(th), np.sin(th)]) * lookAheadDistance
            v = qcar.motorTach

            # ---- 决策：使用识别线程发布的最新识别结果 ----
            detectionFrame, perceptionPeriod, perceptionError = shared.readDetection()
            decision = decisionModule.update(detectionFrame, dt, v)

            # ---- 控制律与指令写入 ----
            vRefShaped = 0.0
            offset = 0.0
            if t < startDelay:
                u = 0.0
                delta = 0.0
                speedRamp.reset(0.0)
                offsetShaper.reset(0.0)
            else:
                # 期望速度/偏移先做斜坡限制，再交给 PI 与 Stanley 跟踪
                offset = offsetShaper.update(decision.lateralOffset, dt)
                if steeringController is not None:
                    delta = steeringController.update(pControl, th, v, offset)
                else:
                    delta = 0.0

                # 连续转向检测：前轮转角超过阈值则累计帧数
                if abs(delta) > turnSteeringThresholdRad:
                    _steerCount += 1
                else:
                    _steerCount = 0
                # 连续打方向时巡航速度降至 steeringCruiseSpeed（0.30 m/s）
                if _steerCount >= sustainedSteeringFrames and decision.state == TrafficDecision.STATE_CRUISE:
                    decision.vRef = min(decision.vRef, steeringCruiseSpeed)

                vRefShaped = speedRamp.update(decision.vRef, dt)
                u = speedController.update(v, vRefShaped, dt)

                # 转弯时降低油门至直线巡航的 90%，避免过弯速度过冲
                if abs(delta) > turnSteeringThresholdRad:
                    u *= turnThrottleScale
                # 遇锥桶执行避让时再额外降低油门（与转弯叠加）
                if decision.state in (
                    TrafficDecision.STATE_SLOW_CONE,
                    TrafficDecision.STATE_BYPASS_CONE,
                    TrafficDecision.STATE_STOP_CONE,
                ):
                    u *= coneAvoidThrottleScale
            qcar.write(u, delta)

            # ---- 遥测与可视化采样（10 Hz）----
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                if showScope:
                    steeringScope.axes[4].sample(t_plot, [[pVehicle[0], pVehicle[1]]])
                    steeringScope.axes[0].sample(t_plot, [v, vRefShaped])
                    steeringScope.axes[1].sample(t_plot, [delta])
                    steeringScope.axes[2].sample(t_plot, [offset])
                    steeringScope.axes[3].sample(
                        t_plot, [float(decisionModule.codeOf(decision.state))])
                    arrow.setPos(pVehicle[0], pVehicle[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)
                shared.publishDecision(decision, {
                    't': t_plot,
                    'v': float(v),
                    'vRef': float(vRefShaped),
                    'u': float(u),
                    'delta': float(delta),
                    'offset': float(offset),
                    'state': decision.state,
                    'stateCode': float(decisionModule.codeOf(decision.state)),
                    'reason': decision.reason,
                    'countdown': float(decision.countdown),
                    'x': float(pVehicle[0]),
                    'y': float(pVehicle[1]),
                    'th': float(th),
                    'period': float(perceptionPeriod),
                    'inference': float(shared.inferenceTime),
                    'blind': 1.0 if detectionFrame is None else 0.0,
                    'perceptionError': perceptionError,
                })
                count = 0

        # 退出前确保车辆停止（连接已断开时写指令也会失败，这里兜底忽略）
        try:
            qcar.read_write_std(throttle=0, steering=0)
        except Exception:
            pass
    finally:
        # 释放车辆/GPS 资源；用 hasattr 保护，避免 GPS 初始化半途失败时
        # QCarGPS.terminate() 因缺少 _lidar_client 属性而二次报错。
        for resource in (gps, qcar):
            if resource is None or not hasattr(resource, 'terminate'):
                continue
            try:
                resource.terminate()
            except Exception:
                pass

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 实验主流程

if __name__ == '__main__':

    #region : Scope 可视化配置（风格与 0/QCar2_Steering_Control.py 一致）
    fps = 10 if IS_PHYSICAL_QCAR else 30

    showScope = bool(showScope and enableSteeringControl)
    if showScope:
        steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='YOLO 识别-决策-控制综合演示',
            fps=fps
        )
        steeringScope.graphicsLayoutWidget.resize(*scopeWindowSize)
        # 左列：时间曲线（速度 / 转角 / 绕行偏移 / 决策状态码）
        steeringScope.addAxis(
            row=0, col=0, timeWindow=tf, yLabel='速度 [m/s]', yLim=(0, 1.2))
        steeringScope.axes[0].attachSignal(name='实际速度')
        steeringScope.axes[0].attachSignal(name='期望速度')
        steeringScope.addAxis(
            row=1, col=0, timeWindow=tf, yLabel='前轮转角 [rad]', yLim=(-0.6, 0.6))
        steeringScope.axes[1].attachSignal(name='前轮转角')
        steeringScope.addAxis(
            row=2, col=0, timeWindow=tf, yLabel='绕行横向偏移 [m]', yLim=(-0.6, 0.6))
        steeringScope.axes[2].attachSignal(name='横向偏移')
        steeringScope.addAxis(
            row=3, col=0, timeWindow=tf, yLabel='决策状态码', yLim=(-1, 9))
        steeringScope.axes[3].attachSignal(name='决策状态')
        # 决策状态码是 0~8 的整数，默认自动刻度只剩 0/5 两个点；
        # 这里显式设置 0~8 整数刻度并附状态简写，便于对照曲线。
        stateTicks = [
            (0, '0 巡航'), (1, '1 斑马线'), (2, '2 锥桶减速'),
            (3, '3 绕行锥桶'), (4, '4 锥桶停车'), (5, '5 行人停车'),
            (6, '6 停止标志'), (7, '7 红灯'), (8, '8 感知失效'),
        ]
        steeringScope.axes[3].plot.getAxis('left').setTicks([stateTicks])
        steeringScope.axes[3].xLabel = '时间 [s]'

        # 右列：轨迹图（叠加 SDCS 城市俯视图与参考路径）
        steeringScope.addXYAxis(
            row=0, col=1, rowSpan=4,
            xLabel='x 位置 [m]', yLabel='y 位置 [m]',
            xLim=(-2.5, 2.5), yLim=(-1, 5))
        im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125, 2365),
            rotation=180,
            levels=(0, 255))
        steeringScope.axes[4].images[0].setImage(image=im)
        referencePath = pg.PlotDataItem(
            pen={'color': (85, 168, 104), 'width': 2},
            name='Reference')
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
        steeringScope.axes[4].attachSignal(name='Estimated', width=2)
        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196, 78, 82], 'width': 1},
            brush=[196, 78, 82])
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    #region : 识别器与相机初始化
    print('=' * 70)
    print('QCar2  YOLO 识别 - 决策 - 控制 综合演示')
    print('=' * 70)

    perception = YoloTrafficPerception(
        modelPath=yoloModelPath,
        confThreshold=confThreshold,
        iouThreshold=iouThreshold,
        imageSize=yoloImageSize,
        device=yoloDevice,
        maxDetections=maxDetections,
    )
    print('YOLO 权重文件：', perception.modelPath)
    print('类别映射：', ', '.join(
        '{}:{}'.format(index, name) for index, name in enumerate(perception.classNames)))
    if warmUpModel:
        print('模型预热用时：{:.2f} s'.format(perception.warmUp()))

    camera = None
    try:
        camera = QCarRealSense(
            mode='RGB',
            frameWidthRGB=cameraWidth,
            frameHeightRGB=cameraHeight,
            frameRateRGB=cameraFrameRate,
            readMode=0,          # 非阻塞读取：无新帧时 read_RGB() 返回 -1，识别线程可据此检查退出标志
        )
        # QCarRealSense 的构造函数内部会吞掉 MediaError（只打印不抛出），
        # 因此必须主动校验 RGB 视频流是否真正打开，否则相机不可用时会被误报为“已就绪”。
        if getattr(camera, 'streamRGB', None) is None:
            raise RuntimeError('RGB 视频流未能打开（请确认 QCar2 实时模型已启动）')
        print('车头 RealSense 相机已就绪（{}x{} @ {} Hz）'.format(
            cameraWidth, cameraHeight, cameraFrameRate))
    except Exception as error:
        camera = None
        print('[警告] RealSense 相机初始化失败：{}'.format(error))
        print('       识别功能不可用，决策层将按“感知失效”以 {:.2f} m/s 谨慎行驶。'.format(
            decisionParams.blindSpeed))

    shared = SharedState()
    if camera is None:
        shared.publishDetection(None, 0.0, '相机不可用')
    #endregion

    #region : 启动线程并运行演示
    perceptionThread = None
    if camera is not None:
        perceptionThread = Thread(
            target=perceptionLoop, args=(shared, camera, perception), daemon=True)
        perceptionThread.start()

    controlThread = Thread(target=controlLoop, args=(shared,))
    controlThread.start()

    videoWriter = None
    if showCameraWindow and recordVideo:
        videoWriter = cv2.VideoWriter(
            videoPath, cv2.VideoWriter_fourcc(*'mp4v'), videoFps,
            (cameraWidth, cameraHeight))

    # 相机窗口：用可缩放窗口并按 cameraDisplaySize 放大显示（采集分辨率不变）
    if showCameraWindow:
        cv2.namedWindow(cameraWindowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cameraWindowName, *cameraDisplaySize)

    # ---- 主循环：刷新曲线与相机画面（cv2 窗口在主线程刷新）----
    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            if showScope:
                MultiScope.refreshAll()

            detectionFrame, perceptionPeriod, perceptionError = shared.readDetection()
            _, telemetry = shared.readDecision()

            if showCameraWindow and detectionFrame is not None:
                display = detectionFrame.annotated
                if drawCorridorGuide:
                    draw_corridor_guide(
                        display,
                        decisionParams.corridorHalfWidth,
                        decisionParams.dynamicStopBottomYNorm)
                lines = [
                    '决策状态: {}'.format(telemetry.get('state', '-')),
                    '决策原因: {}'.format(telemetry.get('reason', '-')),
                    '速度: {:.2f} / {:.2f} m/s   油门: {:.3f}'.format(
                        telemetry.get('v', 0.0), telemetry.get('vRef', 0.0),
                        telemetry.get('u', 0.0)),
                    '前轮转角: {:+.3f} rad   绕行偏移: {:+.2f} m'.format(
                        telemetry.get('delta', 0.0), telemetry.get('offset', 0.0)),
                    '识别结果: {}'.format(detectionFrame.summaryText()),
                    '推理耗时: {:.0f} ms   识别周期: {:.2f} s'.format(
                        detectionFrame.inferenceTime * 1000, perceptionPeriod),
                ]
                if perceptionError:
                    lines.append('感知异常: {}'.format(perceptionError))
                draw_status_panel(display, lines)
                cv2.imshow(cameraWindowName, display)
                if videoWriter is not None:
                    videoWriter.write(display)
            elif showCameraWindow:
                # 相机不可用或识别线程尚未产出结果时，用提示画面代替，避免窗口闪烁
                placeholder = np.zeros((cameraHeight, cameraWidth, 3), dtype=np.uint8)
                draw_status_panel(placeholder, [
                    '相机不可用或识别结果缺失',
                    perceptionError if perceptionError else '等待识别线程输出...',
                    '决策状态: {}'.format(telemetry.get('state', '-')),
                ])
                cv2.imshow(cameraWindowName, placeholder)

            if showCameraWindow:
                cv2.waitKey(1)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('\n收到中断信号，正在安全停车并退出...')
    finally:
        KILL_THREAD = True
    #endregion

    #region : 资源释放与演示总结
    controlThread.join(timeout=2.0)
    if perceptionThread is not None:
        perceptionThread.join(timeout=2.0)

    if videoWriter is not None:
        videoWriter.release()
        print('演示视频已保存：', videoPath)
    if camera is not None:
        try:
            camera.terminate()
        except Exception as error:
            print('[警告] 相机资源释放异常：', error)

    print('-' * 70)
    if shared.decisionModule is not None:
        print('演示结束，决策状态变化记录（共 {} 条）：'.format(
            len(shared.decisionModule.transitionLog)))
        print(shared.decisionModule.historyText())
    else:
        print('演示结束，决策状态变化记录（共 0 条）')
    print('-' * 70)

    if (not IS_PHYSICAL_QCAR) and resetVehiclePoseAtStart and (sceneModule is not None):
        try:
            sceneModule.terminate()
        except Exception as error:
            print('[警告] QLabs 实时模型释放异常：', error)

    # 演示结束后保持综合演示窗口与相机画面，按任意键关闭
    print('\n演示结束，窗口将保持打开。按 ESC 或关闭窗口退出...')
    while True:
        if showScope:
            MultiScope.refreshAll()
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        # 窗口被用户手动关闭则退出
        if showCameraWindow:
            try:
                if cv2.getWindowProperty(cameraWindowName, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break
    if showCameraWindow:
        cv2.destroyAllWindows()
    #endregion

#endregion






