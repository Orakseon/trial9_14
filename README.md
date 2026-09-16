# QCar2 YOLO 识别 - 决策 - 控制 综合演示（trial9_14）

本目录为结题综合演示程序：**用 YOLO 识别交通要素（红绿灯 / 斑马线行人 / 锥桶 / 停止标志等），
由决策层按交通规则给出行为（巡航、减速、停车等待、确认安全后绕行），再由控制层完成
QCar 的速度 PI 控制与 Stanley 路径跟踪**。

按任务要求：**不进行场景构建**，只搭建“识别 - 决策 - 控制”层。

---

## 1. 文件结构

| 文件 | 层次 | 说明 |
| --- | --- | --- |
| `QCar2_YOLO_Decision_Control.py` | 主程序 | 相机/车辆接口、线程调度、加锁数据交换、MultiScope 与相机画面可视化 |
| `setup_trial.py` | 场景 | 运行一次即可搭建虚拟场景：斑马线 5 条（`crosswalk.spawn_degrees`）+ 信号灯 4 个（`trafficLight.spawn_id_degrees`），可选生成车辆并启动实时模型 |
| `perception_yolo.py` | 识别层 | YOLOv11 推理与结果解析（类别、置信度、面积占比、归一化几何量）、走廊辅助线工具 |
| `decision_layer.py` | 决策层 | 交通规则状态机，输出期望速度 `vRef` 与绕行横向偏移 `lateralOffset` |
| `control_layer.py` | 控制层 | 期望速度斜坡、速度 PI、绕行偏移斜坡、Stanley 转向控制器 |
| `lidar_perception.py` | 感知层 | 激光雷达互补感知：网格聚类障碍检测 → 转换为 YOLO 兼容 Detection 对象 |
| `yolov11s.pt` | 权重 | 与 `../5_factors/sd/YOLO_object_detection.py` 使用的同一套 8 类权重 |

类别映射（与权重严格一致，顺序不可更改）：
`0 Cone(交通锥) / 1 Cow(奶牛) / 2 Crosswalk(人行横道) / 3 GREEN(绿灯) / 4 People(行人) /
5 RED(红灯) / 6 Stop Line(停止线) / 7 Stop Sign(停止标志)`

---

## 2. 运行方法

1. **场景**（二选一）：
   * **运行本目录的场景脚本（推荐）**，自动生成 5 条斑马线 + 4 个信号灯：

     ```powershell
     cd C:\Users\Obbha\Desktop\QCar_finale\trial9_14
     python setup_trial.py                      # 交通要素 + 车辆，并保持实时模型运行（Ctrl+C 结束）
     python setup_trial.py --no-vehicle         # 只搭建交通要素，不生成车辆
     python setup_trial.py --color red          # 信号灯初始为红灯（green/red/yellow/none）
     python setup_trial.py --cycle --interval 6 # 运行中每 6 s 红/绿交替，便于演示停车与通行
     ```

     保持该脚本运行，另开一个终端运行主程序（QLabs 支持多连接）。
   * 或使用队友已搭好的 SDCS 路网场景：把 `resetVehiclePoseAtStart` 置为 `True`，
     主程序启动时会自动调用 `setup_trial.setup(...)` 重建场景并把车辆复位到路网起点；
     保持默认 `False` 则完全不改动现有场景。
2. **环境**：Python 3.11 + `pal`（Quanser 库，位于 `C:\Users\Obbha\Documents\Quanser\0_libraries\python`）
   + `ultralytics` + `torch` + `opencv-python` + `pyqtgraph`。
3. **启动**：

   ```powershell
   cd C:\Users\Obbha\Desktop\QCar_finale\trial9_14
   python QCar2_YOLO_Decision_Control.py
   ```

4. **结束**：按 `Ctrl+C`，程序会安全停车（写入 `throttle=0, steering=0`）、释放相机与窗口，
   并打印本次演示的决策状态变化记录。

> 若需要程序自动把车辆复位到路网起点：把 `0/qlabs_setup_task01.py` 拷贝到本目录，
> 并把 `resetVehiclePoseAtStart` 置为 `True`（默认 `False`，即使用 QLabs 中的当前位姿）。

---

## 3. 层次与接口

```
车头 RealSense RGB 图像
        │
        ▼                              ┌──────────────────┐
[识别层] YoloTrafficPerception         │  QCarLidar (20Hz) │
        │    DetectionFrame            └────────┬─────────┘
        │                                      │ LidarObstacle[]
        │                             ┌────────▼─────────┐
        │                             │ lidar_perception  │
        │                             │ to_detections()   │
        │                             └────────┬─────────┘
        │                           Detection(label=People)
        │                                      │
        ▼                                      ▼
[决策层] TrafficDecision.update(detectionFrame + lidarDets, dt, v)
        │    DecisionResult: state / vRef / lateralOffset / reason / countdown
        ▼
[控制层] SpeedRampLimiter → SpeedController → throttle
        │   OffsetShaper → SteeringController（Stanley + 绕行偏移）→ steering
        ▼
      QCar.write(throttle, steering)
```

线程模型：

| 线程 | 频率 | 职责 |
| --- | --- | --- |
| 识别线程 `perceptionLoop` | 相机帧率，按推理耗时自适应（0.05~0.5 s） | 取图 → YOLO 推理 → 发布最新 `DetectionFrame`（加锁） |
| 控制线程 `controlLoop` | 100 Hz | 读编码器/GPS → EKF → 决策 → 控制 → 写指令；10 Hz 刷新曲线 |
| 主线程 | ~100 Hz | `MultiScope.refreshAll()` + 相机画面叠加状态信息（cv2 窗口只在主线程刷新） |

---

## 4. 决策状态机

优先级由高到低（`decision_layer.py` 中的状态常量）：

| 状态 | 触发条件 | 行为 |
| --- | --- | --- |
| `BLIND` | 识别结果缺失或超过 `detectionTimeout` 未更新 | 以 `blindSpeed`（默认 0.15 m/s）谨慎行驶 |
| `STOP_RED` | 红灯连续 `lightConfirmFrames` 帧确认（闭锁） | 停车等待；绿灯连续确认后放行；红灯驶出视野且未见绿灯超过 `redLostReleaseTime` 后谨慎通过 |
| `STOP_SIGN` | 停止标志进入本车道视野 | 强制停车 `stopSignHoldTime`（默认 2.5 s），之后放行；标志未离开视野前不重复触发 |
| `STOP_PEDESTRIAN` | 行人/奶牛位于前方走廊内且距离已近 | 停车等待，障碍离开走廊并保持 `clearConfirmTime` 后恢复 |
| `STOP_CONE` | 锥桶近距且位于本车轨迹上 | 停车观察 `coneObserveTime`（默认 1.2 s） |
| `BYPASS_CONE` | 观察结束且前方无动态障碍（或锥桶明显偏离轨迹） | 以 `bypassSpeed` 限速，并沿路径左法向偏移 `bypassOffset` 绕行；通过后自动回正 |
| `SLOW_CROSSWALK` | 检测到斑马线，或走廊内存在尚未逼近的行人 | 以 `crosswalkSpeed`（默认 0.2 m/s）限速通过 |
| `SLOW_CONE` | 远处锥桶 | 以 `coneSlowSpeed`（默认 0.3 m/s）减速 |
| `CRUISE` | 前方无影响目标 | 以 `cruiseSpeed`（默认 0.5 m/s）巡航 |

设计要点：

1. **连续帧确认 + 闭锁**：红/绿灯、停止标志、动态障碍都要求多帧确认或离开确认，
   避免单帧误检导致车辆频繁启停；
2. **安全的保守策略**：感知失效时按最保守方式处理（低速或停车），且所有停车决策都以
   障碍“已离开走廊”为恢复条件；
3. **绕行前先确认安全**：只有“停车观察结束 + 走廊内无行人/奶牛”才进入绕行。

---

## 5. 场景要素（setup_trial.py）

坐标约定：**坐标表中的数值是 SDCS 路网世界坐标**（与 `SDCSRoadMap` 的 `nodeSequence=[0,23,0]` 同一坐标系，
例如节点 0 为 (0.00, 0.13, -90°)）；QLabs 场景坐标 = 世界坐标 × `QLABS_SCALE`(=10)，与
`0/qlabs_setup_task01.py` 中 `location=[p*10 for p in initialPosition]` 的车辆生成约定一致。

### 要素一：斑马线 5 条 —— `crosswalk.spawn_degrees(...)`

| # | 名称 | 世界坐标 (x, y, z) | 朝向 (roll, pitch, yaw°) | 生成位置（QLabs） |
| --- | --- | --- | --- | --- |
| 1 | 中央北侧斑马线 | (0.0, 1.5, 0.0) | (0, 0, 0) | (0.00, 15.00, 0.005) |
| 2 | 中央南侧斑马线 | (0.0, 0.9, 0.0) | (0, 0, 0) | (0.00, 9.00, 0.005) |
| 3 | 中央西侧斑马线 | (-0.5, 1.2, 0.0) | (0, 0, 90) | (-5.00, 12.00, 0.005) |
| 4 | 中央东侧斑马线 | (0.5, 1.2, 0.0) | (0, 0, 90) | (5.00, 12.00, 0.005) |
| 5 | 右上方倾斜斑马线 | (0.8, 3.1, 0.0) | (0, 0, 45) | (8.00, 31.00, 0.005) |

### 要素二：信号灯 4 个 —— `trafficLight.spawn_id_degrees(...)`

| actorNumber | 名称 | 控制车流 | 面向 | 世界坐标 | 朝向 yaw° | 生成位置（QLabs） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 路口西侧信号灯 | 北向南 | +Y（北） | (-0.7, 1.3, 0.0) | 90 | (-7.00, 13.00, 0.0) |
| 2 | 路口东侧信号灯 | 东向西 | +X（东） | (0.8, 1.3, 0.0) | 0 | (8.00, 13.00, 0.0) |
| 3 | 路口南侧偏左信号灯 | 西向东 | -X（西） | (-0.4, 0.5, 0.0) | 180 | (-4.00, 5.00, 0.0) |
| 4 | 路口南侧偏右信号灯 | 南向北 | -Y（南） | (0.4, 0.5, 0.0) | 270 | (4.00, 5.00, 0.0) |

说明：

* 朝向角含义：`0° = 面向 +X（东）`、`90° = 面向 +Y（北）`、`180° = 面向 -X（西）`、`270° = 面向 -Y（南）`；
  灯头面向来车方向，符合右侧通行原则（`configuration=0`）。
* 信号灯初始颜色由 `--color` 指定（默认绿灯）；也可在程序内调用
  `setup_trial.set_light_color(scene['traffic_lights'], 'red', actorNumbers=[1, 4])` 只把指定路口的灯设为红灯。
* 车辆默认生成在路网节点 0（世界坐标 (0.00, 0.13)，航向 -90°）对应 QLabs (0.00, 1.30)；
  可用 `setup_trial.setup(initialPosition=[x, y, 0], initialOrientation=[0, 0, yawRad])` 自定义。
* 场景比例不同时（例如使用 1:1 场景），只需调整 `QLABS_SCALE` 或坐标表中的数值。

## 6. 现场标定建议

所有判据都用“归一化坐标 / 面积占比”表达，与相机分辨率无关，调整时参考相机画面上的辅助线：

* 打开主程序即会绘制**走廊边界线**（`corridorHalfWidth`）与**近距阈值线**（`dynamicStopBottomYNorm`）；
* 目标框**底边越过阈值线**或**面积占比超过阈值**即判为“距离已近”，可用于标定
  `dynamicStopBottomYNorm` / `coneNearBottomYNorm` / `coneStopAreaPercent`；
* 红绿灯按“面积占比 + 横向位于本车道 + 中心位于画面上部”判定：
  `redMinAreaPercent` / `greenMinAreaPercent` / `trafficLightCorridorHalfWidth` / `lightMaxCenterYNorm`；
* 若误检导致频繁停车，可提高 `confThreshold`（0.35 → 0.5）或提高面积阈值；
* CPU 推理较慢时可降低 `yoloImageSize`（640 → 480/320）提升识别帧率。

---

## 7. 环境依赖说明（pal / hal / qvl 与 RTMODELS_DIR）

本目录用到的 Quanser Python 库**都不是 PyPI 包**，由 Quanser QCar 2 / Autonomous Vehicles SDK
（Quanser Interactive Labs，QLabs）安装器统一安装到：

```
C:\Users\<用户名>\Documents\Quanser\0_libraries\python\
    ├─ pal\   Platform Abstraction Layer：QCar / QCarGPS / QCarRealSense、MultiScope、SDCS_CITYSCAPE …
    ├─ hal\   硬件抽象层扩展：QCarEKF、SDCSRoadMap …
    ├─ qvl\   QLabs API：QuanserInteractiveLabs、QLabsQCar2、QLabsCrosswalk、QLabsTrafficLight、QLabsRealTime …
    └─ pit\   Python Interface Toolkit
```

安装器同时为当前用户设置 3 个环境变量（本机实测）：

| 变量 | 本机值 | 作用 |
| --- | --- | --- |
| `PYTHONPATH` | `C:\Users\<用户名>\Documents\Quanser\0_libraries\python` | 让 `import pal / hal / qvl` 能被找到 |
| `QAL_DIR` | `C:\Users\<用户名>\Documents\Quanser` | SDK 根目录 |
| `RTMODELS_DIR` | `...\Quanser\0_libraries\resources\rt_models` | 实时模型（`QCar2_Workspace` 等）所在目录 |

### `pal.resources.rtmodels` 是什么

它**只是一个路径常量模块**（`pal\resources\rtmodels.py`，全文 6 个常量）：

```python
import os
__rtModelDirPath = os.environ['RTMODELS_DIR']      # ← 直接读环境变量，缺失即抛 KeyError
QCAR            = <RTMODELS_DIR>/QCar/QCar_Workspace
QCAR2           = <RTMODELS_DIR>/QCar2/QCar2_Workspace    # 本机 = ...\rt_models\QCar2\QCar2_Workspace
QCAR_STUDIO / QCAR2_STUDIO / QBOT_PLATFORM / QDRONE2 …
```

作用只有一个：把“实时模型文件名”传给 `QLabsRealTime().start_real_time_model(...)`
（该接口形参说明就是“model 文件名，不含扩展名”，内部拼成 `{modelName}.rt-linux_x86_64` 交给 `quarc_run`）。

### 为什么别的电脑上会找不到

| 报错现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'pal'` | 该机器没装 Quanser SDK，或没把 `0_libraries\python` 加入 `PYTHONPATH`（PyPI 上同名的 `pal` 包与本 SDK 无关，不能替代） | 安装 QLabs / QCar SDK；或把 SDK 的 `0_libraries\python` 目录追加到 `PYTHONPATH` |
| `KeyError: 'RTMODELS_DIR'` | SDK 装了，但当前进程没有该环境变量（从非 Quanser 快捷方式启动的终端 / VS Code，或环境变量被清掉） | `setx RTMODELS_DIR "...\Quanser\0_libraries\resources\rt_models"`，重启终端与 VS Code |
| `ModuleNotFoundError: No module named 'qvl'` | QLabs 的 Python API 未安装 | 安装 Quanser Interactive Labs（含 Python API），或使用 SDK 自带的 Python 环境 |

在目标机器的新终端里先做 3 行自检：

```powershell
echo $env:PYTHONPATH ; echo $env:RTMODELS_DIR
python -c "import pal, hal, qvl; print(pal.__file__); print(qvl.__file__)"
```

### 本仓库的处理方式

* **`setup_trial.py` 已去除 pal 依赖**：不再 `import pal.resources.rtmodels`，而是按同一规则自行拼装实时模型路径，
  环境变量缺失时退化为只传模型名（官方示例 `terminate_real_time_model('QCar2_Workspace')` 就是这种用法）：

  ```python
  RT_MODEL_DIR = os.environ.get('RTMODELS_DIR')
  QCAR2_RT_MODEL = (os.path.normpath(os.path.join(RT_MODEL_DIR, 'QCar2', 'QCar2_Workspace'))
                    if RT_MODEL_DIR else 'QCar2_Workspace')
  ```

  因此该脚本只依赖 `qvl`（QLabs API），在没有安装 PAL 的机器上也能单独搭建场景。
* **主程序 `QCar2_YOLO_Decision_Control.py` 仍需要完整 SDK**：它使用 `pal.products.qcar`（QCar / QCarGPS /
  QCarRealSense）、`pal.utilities.scope`（MultiScope）、`hal.content.qcar_functions.QCarEKF`、
  `hal.products.mats.SDCSRoadMap`，这些都是 SDK 内容，无法绕过。
* 若只需要 VS Code 的代码提示（不运行），可在 `.vscode/settings.json` 加：

  ```json
  { "python.analysis.extraPaths": ["C:\\Users\\<用户名>\\Documents\\Quanser\\0_libraries\\python"] }
  ```

* 仓库里其它官方示例（如 `0/qlabs_setup_task01.py`、`10_turn/…`）使用 `import pal.resources.rtmodels as rtmodels`，
  在新机器上报错时按上表安装/配置即可，也可改成上面的 `QCAR2_RT_MODEL` 写法。

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| 提示“RealSense 相机初始化失败” | 检查 QLabs 场景中 QCar2 是否带 D435 相机、`video3dPort`(18965) 是否被占用；程序会自动降级为“感知失效 + 低速行驶” |
| 识别帧率过低（CPU 推理） | 降低 `yoloImageSize`、提高 `detectPeriodMin`，或使用带 CUDA 的环境并把 `yoloDevice` 设为 `'0'` |
| 车辆在路口不停车 | 确认红灯在本车道视野内、面积占比达到 `redMinAreaPercent`；可用辅助线核对走廊范围 |
| 停车后不恢复行驶 | 检查障碍是否仍在走廊内、绿灯是否被识别到；宽限时间参数 `redLostReleaseTime`、`clearConfirmTime` 可适当放宽 |
| 绕行幅度过大/过小 | 调整 `bypassOffset`（默认 0.35 m）与 `K_stanley` |
| 相机画面上的中文显示为方块/乱码 | 程序默认用 Pillow + 系统中文字体（`C:\Windows\Fonts\msyh.ttc`）渲染状态面板；若环境缺少 Pillow 会自动退化为 OpenCV 英文渲染（功能不受影响） |
| 停车时车辆有轻微倒退倾向 | 已在速度控制器中加入“停车保持”判据（`stopSpeedThreshold`）与积分净化（`integralPurgeError`），必要时可调大阈值 |
| 运行 `setup_trial.py` 提示 “Unable to connect to QLabs” | 先在 QLabs 中打开并加载 **QCar Cityscape**（SDCS 路网）场景，再运行脚本；脚本会自动清空上一次生成的 actor 与实时模型 |
| 斑马线 / 信号灯的位置或大小与预期不符 | 坐标表给的是 SDCS 世界坐标，脚本按 `QLABS_SCALE = 10` 换算为 QLabs 坐标（与车辆生成约定一致）；若使用其它比例的场景，修改 `QLABS_SCALE` 或坐标表数值即可 |
| 信号灯不按预期变化 | `setup_trial.py` 默认把所有灯设为绿灯；用 `--cycle` 让脚本自动红/绿交替，或在演示脚本中调用 `setup_trial.set_light_color(..., actorNumbers=[...])` |
---

## 9. 行人生成功能（自 v2）

`Untitled-2.py` 已集成来自 `setup3.py` 的行人生成逻辑：

| 组件 | 说明 |
| --- | --- |
| `PEDESTRIAN_PATHS` | 5 条行人路径（SDCS 坐标），与 5 条斑马线位置一一对应 |
| `spawn_people(qlabs, scale)` | 用 `QLabsPerson.spawn_id()` 生成 5 个行人，actorNumber=100~104 |
| `start_people(people)` | 为每个行人启动 daemon 线程，以 WALK 速度在 start/end 之间往返走动 |

行人线程在 Ctrl+C 退出时随进程结束，QLabs 的 `destroy_all_spawned_actors()` 会一并清理。

---

## 10. 信号灯防误识别（自 v2）

两个典型误检的解决方案：

| 误检场景 | 置信度 | 根因 | 对策 |
| --- | --- | --- | --- |
| 远处绿灯 → RED | ~40% | 小目标像素不足，模型混淆 | `confThreshold` 0.35 → **0.45**，直接拦截低置信度误判 |
| 信号灯背面外壳 → RED | ~75% | 暗色外壳偏红，模型未见背面样本 | **HSV 颜色二次验证**：`_verifyRedLight()` 检查检测框内红色像素占比 ≥ 8% 才放行 |

颜色验证逻辑：裁剪 RED 检测框 → BGR→HSV → 统计红色像素（H∈[0,12]∪[170,180]，S≥50，V≥60）→ 占比 < 8% 则丢弃。
真红灯发光区域红色占比 >30%，背面外壳/远处绿灯 < 5%，阈值 8% 有足够安全裕度。

---

## 11. 性能优化记录（自 v2）

| 参数 | 旧值 | 新值 | 效果 |
| --- | --- | --- | --- |
| `yoloImageSize` | 640 | 480 | 推理面积降至 56%，速度提升约 ×1.7 |
| `detectPeriodMax` | 0.5 s | 0.33 s | 保底 ≥ 3 Hz |
| 自适应乘数 | ×1.5 | ×1.2 | 减少等待时间 |
| `lightConfirmFrames` | 3 帧 | 7 帧 | 信号灯确认更稳健，减少误触发 |
| `confThreshold` | 0.35 | 0.45 | 过滤低置信度误判 |
| 相机窗口 | 固定 960×720 | 640×480（可缩放） | 默认尺寸恢复，保留拖动调整能力 |

---

## 12. 锥桶生成集成（setup_static.py，自 v2）

`setup_static.py` 参考 `Untitled-3.py` 的锥桶生成逻辑，新增锥桶场景要素：

| 组件 | 说明 |
| --- | --- |
| `CONE_POSITIONS` | 4 个锥桶在 SDCS 世界坐标系下的位置（初始沿用 Untitled-3 参考坐标） |
| `spawn_cones(qlabs, verbose=True)` | 用 `QLabsTrafficCone.spawn_id_degrees()` 生成锥桶，actorNumber=200~203 |
| setup() 调用 | 在信号灯生成之后、车辆生成之前执行 `cones = spawn_cones(qlabs, ...)` |
| 返回字典 | 新增 `'cones': cones` 字段，供调用方获取生成状态 |
| 失败统计 | `failedCones` 跟踪生成失败的锥桶，摘要与警告消息均包含锥桶计数 |

锥桶位置如需根据实际路口几何调整，直接修改 `CONE_POSITIONS` 列表中的 SDCS 坐标即可；若需适配 `QLABS_SCALE` 与其他要素保持一致，脚本中所有要素共用同一缩放倍数。
---

## 13. 锥桶避障响应调优与误检防御（自 v2）

### 锥桶近距阈值调整

原有锥桶"距离已近"的判断条件过于保守，导致车辆接近锥桶时才触发减速/停车，来不及避让。两处关键阈值同步下调：

| 参数 | 旧值 | 新值 | 效果 |
| --- | --- | --- | --- |
| `coneSlowSpeed` | 0.30 m/s | **0.20 m/s** | 远处锥桶即减到更低速度 |
| `coneNearBottomYNorm` | 0.72 | **0.60** | 底边在画面更低位置即判定为"已近"（更早触发） |
| `coneStopAreaPercent` | 0.90 | **0.45** | 锥桶面积占比更小时即判定为"已近"（更容易触发停车/绕行） |

**涉及文件**（两处需同步修改）：
- `decision_layer.py`：`DecisionParameters` 类的默认字段值（第 78–79, 84 行）
- `QCar2_YOLO_Decision_Control.py`：`decisionParams` 实例化参数（第 114, 130–131 行）

### 误判防御：People/Cow 二次过滤

在空旷路面上，YOLO 偶尔将路面纹理/路肩误判为 People 或 Cow（置信度 ~0.45–0.55）。防御措施分两层：

| 层级 | 策略 | 说明 |
| --- | --- | --- |
| **主阈值** | `confThreshold` 0.45 → 0.50 | 全局置信度门槛，拦截大量低置信度误判 |
| **二次过滤** | 类别特定校验 | 在 `perception_yolo.py` 的 `detect()` 中，对 People/Cow 额外要求：<br>• 置信度 ≥ **0.60**（独立于全局阈值）<br>• 宽高比 ≤ **2.5:1**（过滤横向宽幅路面纹理伪检） |

宽高比校验的逻辑：真实行人/奶牛的检测框接近方形（w/h ≈ 0.5–1.5），而路面纹理误检框往往宽高比 > 3。使用 `max(y2-y1, 1)` 防止除零错误。

```
detections = [d for d in detections if not (
    d.label in ('People', 'Cow') and
    (d.confidence < 0.60 or
     (d.x2 - d.x1) / max(d.y2 - d.y1, 1) > 2.5)
)]

---

## 14. Lidar 激光雷达互补感知（自 v3）

Lidar 与 YOLO 视觉互补：检测到 YOLO 可能遗漏的障碍物（低光/背光/复杂纹理），
自动转换为 `People` 标签的 `Detection` 对象注入决策层。

### 架构

| 组件 | 说明 |
| --- | --- |
| `lidar_perception.py` | 独立的雷达感知模块，含 `LidarProcessor` 类 |
| `QCarLidar` | pal 库标准接口，100 点/扫描，模式 2（标准测距） |
| 检测频率 | 每 5 个控制周期一次（100 Hz → **20 Hz**），平衡精度与计算开销 |
| 融合方式 | Lidar 障碍物以 `People` 标签注入 `DetectionFrame.detections`，决策层自动按行人逻辑处理 |

### 障碍检测算法

1. **坐标转换**：`anglesCar = wrap_2pi(2.5π - angles)`，`x=d·cos(a)`，`y=d·sin(a)`（车身系：x=前，y=左）
2. **筛选**：距离 0.15~4 m，正前方 ±75° 锥形区域
3. **网格聚类**：5 cm 分辨率 → 8 连通 BFS → 簇内点数 ≥ 12 判定为障碍
4. **质心提取**：取簇内点的均值坐标 (cx, cy) → 距离 + 方位角

### 坐标映射（Lidar → 归一化）

| Lidar 物理量 | 归一化坐标 | 公式 |
| --- | --- | --- |
| 纵向距离 d (m) | `bottomYNorm` | `1.0 − d / 5.0`（越近越大） |
| 横向偏移 y (m) | `lateralErrorNorm` | `−y × 0.32`（1 m ≈ 0.32，走廊半宽对应 1 m） |
| 簇大小 + 距离 | `areaPercent` | `2.5 / (d + 0.3)`（越近越大） |

### 决策效果

- 障碍物在走廊内（横向 ≤1 m）且距离 < ~2 m → **停车等待** (`STOP_PEDESTRIAN`)
- 障碍离开后 0.6 s → **自动放行**
- Lidar 初始化失败 / 不可用时 → **零干预退化为纯视觉模式**

### 配置参数（`QCar2_YOLO_Decision_Control.py`）

```python
enableLidar = True           # 是否启用激光雷达（False 则纯视觉）
lidarDetectInterval = 5      # 检测间隔（控制周期数，100Hz 下 5 次 ≈ 20Hz）
```

Lidar 障碍检测参数在 `lidar_perception.py` 的 `LidarConfig` 中集中管理：`detectionRadius`、`gridResolution`、`minClusterPoints`、`validAngleHalf`、`maxRange` 等。
```
