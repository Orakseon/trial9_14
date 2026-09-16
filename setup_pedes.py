# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 文件说明与导入
"""
setup_trial.py
结题综合演示 - 虚拟场景搭建脚本（QLabs / QCar Cityscape 场景）
本脚本按任务给定坐标生成演示所需的交通要素（运行一次即可在 QLabs 中把场景搭好）：
    要素一：斑马线 5 条        —— crosswalk.spawn_degrees(...)
    要素二：信号灯 4 个        —— trafficLight.spawn_id_degrees(...)
                                ✅ 新规则：信号灯先放置在四条斑马线围成矩形四个顶角，
                                   再沿顶角指向矩形中心对角线方向向内收缩0.005m；
                                   信号灯不在斑马线上
坐标系约定（重要）：
    坐标表中的数值为 SDCS 路网世界坐标（与 SDCSRoadMap 的 nodeSequence=[0,23,0] 同一坐标系，
    SDCS 节点位姿约为 ±3 m，例如节点 0 为 (0.00, 0.13, -90°)）。
    QLabs 场景坐标 = SDCS 世界坐标 × QLABS_SCALE，与 0/qlabs_setup_task01.py 中
    “location=[p*10 for p in initialPosition]” 的车辆生成约定完全一致，因此本脚本默认
    QLABS_SCALE = 10.0（车辆、斑马线、信号灯使用同一换算关系，场景比例才会正确）。
    python setup_trial.py                 # 搭建场景（含车辆）并保持运行，Ctrl+C 退出
    python setup_trial.py --no-vehicle    # 只搭交通要素，不生成车辆
    python setup_trial.py --color red     # 信号灯初始为红灯（green/red/yellow/none）
    python setup_trial.py --cycle         # 运行中自动周期切换信号灯颜色（便于演示停车/通行）
"""
import argparse
import math
import sys
import time
from qvl.crosswalk import QLabsCrosswalk
from qvl.free_camera import QLabsFreeCamera
from qvl.qcar2 import QLabsQCar2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.traffic_light import QLabsTrafficLight
from qvl.traffic_cone import QLabsTrafficCone
import threading
from qvl.person import QLabsPerson
import pal.resources.rtmodels as rtmodels
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 场景参数
QLABS_SCALE = 10.0              # SDCS 世界坐标 -> QLabs 场景坐标 的放大倍数
CROSSWALK_Z_LIFT = 0.005        # 斑马线抬升量（避免与地面 z-fighting，可按需改 0.0）
TITLE = 'Final Trial: YOLO Perception - Decision - Control'
NODE_SEQUENCE = [0, 23, 0]      # 与主程序一致的演示路线（仅用于取车辆初始位姿）
# 信号灯初始颜色：'green' / 'red' / 'yellow' / 'none'
INITIAL_LIGHT_COLOR = 'green'
# 运行中自动切换信号灯（--cycle 打开）：切换间隔（秒）
LIGHT_CYCLE_INTERVAL = 6.0

# ========= 路口矩形几何参数（由4条路口斑马线确定） =========
# 四条斑马线围成矩形边界 SDCS坐标
RECT_X_WEST  = -0.4
RECT_X_EAST  = 0.6
RECT_Y_SOUTH = 0.5
RECT_Y_NORTH = 1.4

# 矩形中心
RECT_CENTER_X = (RECT_X_WEST + RECT_X_EAST) / 2.0
RECT_CENTER_Y = (RECT_Y_SOUTH + RECT_Y_NORTH) / 2.0

SHRINK_DIST = 0.005   # 沿对角线向内收缩固定距离 0.005 m
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 要素定义表（坐标均为 SDCS 世界坐标）
# 要素一：斑马线 5 条
CROSSWALK_SPECS = [
    dict(name='中央北侧斑马线', location=[0.1, 1.4, 0.0], rotation=[0, 0, 0]),
    dict(name='中央南侧斑马线', location=[0.1, 0.5, 0.0], rotation=[0, 0, 0]),
    dict(name='中央西侧斑马线', location=[-0.4, 0.9, 0.0], rotation=[0, 0, 90]),
    dict(name='中央东侧斑马线', location=[0.6, 0.9, 0.0], rotation=[0, 0, 90]),
    dict(name='右上方倾斜斑马线', location=[0.9, 3.7, 0.0], rotation=[0, 0, 17]),
]

# 要素：行人路径（SDCS 世界坐标，与斑马线位置对应，生成时乘以 QLABS_SCALE）
PEDESTRIAN_PATHS = [
    dict(name='北侧行人', start=[-0.25, 1.4, 0.005], end=[0.55, 1.4, 0.005], yaw=0),
    dict(name='南侧行人', start=[-0.25, 0.5, 0.005], end=[0.55, 0.5, 0.005], yaw=0),
    dict(name='西侧行人', start=[-0.4, 0.60, 0.005], end=[-0.4, 1.30, 0.005], yaw=math.pi/2),
    dict(name='东侧行人', start=[0.6, 0.57, 0.005], end=[0.6, 1.28, 0.005], yaw=math.pi/2),
    dict(name='右上斜向行人', start=[0.55, 3.59, 0.005], end=[1.25, 3.81, 0.005], yaw=math.radians(17)),
]

# 要素：锥桶（QLabs 坐标，从 Untitled-3.py 参考位置）
CONE_POSITIONS = [
    [-20.0, 37.0, 0.25],
    [-20.0, 36.0, 0.25],
    [-17.5, 26.5, 0.25],
    [-17.5, 25.5, 0.25],
]

# 信号灯颜色名称 -> QLabsTrafficLight 颜色常量
_LIGHT_COLOR_NAMES = {
    'none': 'COLOR_NONE',
    'red': 'COLOR_RED',
    'yellow': 'COLOR_YELLOW',
    'green': 'COLOR_GREEN',
}

# 信号灯配置：矩形4个顶角
# 格式：(actorNumber, name, controls, facing, light_yaw_deg, corner_x, corner_y)
LIGHT_BIND_CONFIG = [
    # 西北顶角
    (1, '路口西北侧信号灯', '北向南车流', '+Y（北）', 90, RECT_X_WEST, RECT_Y_NORTH),
    # 东北顶角
    (2, '路口东北侧信号灯', '东向西车流', '+X（东）', 0, RECT_X_EAST, RECT_Y_NORTH),
    # 西南顶角
    (3, '路口西南侧信号灯', '西向东车流', '-X（西）', 180, RECT_X_WEST, RECT_Y_SOUTH),
    # 东南顶角
    (4, '路口东南侧信号灯', '南向北车流', '-Y（南）', 270, RECT_X_EAST, RECT_Y_SOUTH),
]

def generate_traffic_light_specs():
    """
    自动计算4个信号灯SDCS坐标
    几何规则：
    1. 初始位置：四条斑马线围成矩形的四个顶角
    2. 沿顶角指向矩形中心的对角线方向，向内收缩固定距离 SHRINK_DIST =0.005 m
    3. 信号灯不会落在斑马线上
    """
    traffic_specs = []
    for (actorNumber, name, controls, facing, light_yaw, cx, cy) in LIGHT_BIND_CONFIG:
        # 顶角指向中心向量
        dx = RECT_CENTER_X - cx
        dy = RECT_CENTER_Y - cy
        diag_len = math.hypot(dx, dy)
        # 单位向量
        ux = dx / diag_len
        uy = dy / diag_len
        # 向内移动SHRINK_DIST
        lx = cx + SHRINK_DIST * ux
        ly = cy + SHRINK_DIST * uy
        traffic_specs.append(dict(
            name=name,
            actorNumber=actorNumber,
            controls=controls,
            facing=facing,
            location=[lx, ly, 0.0],
            rotation=[0,0,light_yaw]
        ))
    return traffic_specs

TRAFFIC_LIGHT_SPECS = generate_traffic_light_specs()
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 通用工具
def to_qlabs_position(location, zLift=0.0):
    """
    SDCS 世界坐标 -> QLabs 场景坐标。
    换算关系与 0/qlabs_setup_task01.py 中车辆生成一致（location=[p*10 for p in initialPosition]）：
        QLabs.x = SDCS.x * QLABS_SCALE
        QLabs.y = SDCS.y * QLABS_SCALE
        QLabs.z = SDCS.z + zLift
    """
    return [
        float(location[0]) * QLABS_SCALE,
        float(location[1]) * QLABS_SCALE,
        float(location[2]) + float(zLift),
    ]

def to_qlabs_yaw(rotationDeg):
    """取朝向角（度）；本场景中各要素 roll/pitch 均为 0。"""
    return float(rotationDeg[2])

def light_color_constant(trafficLight, colorName):
    """颜色名称 -> QLabsTrafficLight 的颜色常量（如 'green' -> COLOR_GREEN）。"""
    key = str(colorName).strip().lower()
    if key not in _LIGHT_COLOR_NAMES:
        raise ValueError('未知的信号灯颜色：{}（可选：{}）'.format(
            colorName, ', '.join(_LIGHT_COLOR_NAMES.keys())))
    return getattr(trafficLight, _LIGHT_COLOR_NAMES[key])

def _spawn_result(result):
    """解析 spawn_degrees / spawn_id_degrees 的返回值，统一为 (status, actorNumber)。"""
    if isinstance(result, tuple):
        status = result[0]
        actorNumber = result[1] if len(result) > 1 else None
        return status, actorNumber
    return result, None

def _status_text(status):
    """把生成状态码转成可读文本。"""
    if status is None:
        return '已发送（未等待确认）'
    if status == 0:
        return '成功'
    return '失败（状态码 {}）'.format(status)

def default_vehicle_pose():
    """
    车辆默认初始位姿：取 SDCS 路网 nodeSequence 起点的节点位姿（与主程序一致）。
    返回：(SDCS 世界坐标 [x, y, 0], QLabs 车辆姿态 [roll, pitch, yaw(rad)])
    """
    try:
        from hal.products.mats import SDCSRoadMap
        roadmap = SDCSRoadMap(leftHandTraffic=False)
        nodePose = roadmap.get_node_pose(NODE_SEQUENCE[0]).squeeze()
        return ([float(nodePose[0]), float(nodePose[1]), 0.0],
                [0.0, 0.0, float(nodePose[2])])
    except Exception as error:      # 路网不可用时退化为已知的节点 0 位姿
        print('[提示] 读取 SDCS 路网失败（{}），使用默认车辆位姿 (0.00, 0.13, -90°)。'.format(error))
        return [0.0, 0.13, 0.0], [0.0, 0.0, -math.pi / 2]
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 要素一 斑马线（crosswalk.spawn_degrees）
def spawn_crosswalks(qlabs, verbose=True):
    """
    生成 5 条斑马线。
    使用接口：crosswalk.spawn_degrees(location=..., rotation=..., scale=..., configuration=...,
                                      waitForConfirmation=...)（由 QLabs 自动分配 actorNumber）
    参数：
        qlabs    已连接的 QuanserInteractiveLabs 对象
        verbose  是否打印每条斑马线的生成结果
    返回：
        handles 列表，元素为 dict(name, location, rotation, status, actorNumber)
    """
    crosswalk = QLabsCrosswalk(qlabs)
    handles = []
    if verbose:
        print('\n[要素一] 生成斑马线（crosswalk.spawn_degrees）：')
    for index, spec in enumerate(CROSSWALK_SPECS, start=1):
        location = to_qlabs_position(spec['location'], CROSSWALK_Z_LIFT)
        rotation = spec['rotation']
        status, actorNumber = _spawn_result(crosswalk.spawn_degrees(
            location=location,
            rotation=rotation,
            scale=[1, 1, 1],
            configuration=0,
            waitForConfirmation=True,
        ))
        handles.append(dict(
            name=spec['name'],
            location=location,
            rotation=rotation,
            status=status,
            actorNumber=actorNumber,
        ))
        if verbose:
            print('  [{}/{}] {:<10s} 朝向 {:>3.0f}°  QLabs 坐标 (x={:7.2f}, y={:7.2f}, z={:5.3f})  {}'.format(
                index, len(CROSSWALK_SPECS), spec['name'], to_qlabs_yaw(rotation),
                location[0], location[1], location[2], _status_text(status)))
    return handles
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 要素二 信号灯（trafficLight.spawn_id_degrees）
def spawn_traffic_lights(qlabs, initialColor=INITIAL_LIGHT_COLOR, verbose=True):
    """
    生成 4 个信号灯并设置初始颜色。
    ✅ 位置规则：信号灯初始放在矩形四个顶角，沿对角线向中心收缩0.005m
                信号灯位于斑马线外侧，不会压在斑马线上
    使用接口：trafficLight.spawn_id_degrees(actorNumber=..., location=..., rotation=...,
                                            configuration=0, waitForConfirmation=...)，
    configuration=0 为右侧通行布置；朝向按“面向来车方向”给定：
        0° = 面向 +X（东）    90° = 面向 +Y（北）
        180° = 面向 -X（西）  270° = 面向 -Y（南）
    参数：
        qlabs        已连接的 QuanserInteractiveLabs 对象
        initialColor 初始颜色：'green' / 'red' / 'yellow' / 'none'
        verbose      是否打印每个信号灯的生成结果
    返回：
        handles 列表，元素为 dict(name, actorNumber, controls, facing, location, rotation,
                                   status, colorOk, handle)
    """
    handles = []
    if verbose:
        print('\n[要素二] 生成信号灯（trafficLight.spawn_id_degrees，右侧通行、面向来车方向）：')
        print(f'  >> 几何约束：信号灯初始放置于路口矩形四角，沿对角线向矩形中心收缩距离 = {SHRINK_DIST:.3f} m')
        print(f'     矩形中心 SDCS坐标：({RECT_CENTER_X:.3f}, {RECT_CENTER_Y:.3f})，信号灯不在斑马线上')
    for spec in TRAFFIC_LIGHT_SPECS:
        location = to_qlabs_position(spec['location'])
        trafficLight = QLabsTrafficLight(qlabs)
        status, _ = _spawn_result(trafficLight.spawn_id_degrees(
            actorNumber=spec['actorNumber'],
            location=location,
            rotation=spec['rotation'],
            configuration=0,
            waitForConfirmation=True,
        ))
        colorOk = trafficLight.set_color(
            color=light_color_constant(trafficLight, initialColor),
            waitForConfirmation=True)
        handles.append(dict(
            name=spec['name'],
            actorNumber=spec['actorNumber'],
            controls=spec['controls'],
            facing=spec['facing'],
            location=location,
            rotation=spec['rotation'],
            status=status,
            colorOk=colorOk,
            handle=trafficLight,
        ))
        if verbose:
            print('  [{}] {:<10s} 控制{:<7s} 面向 {:<7s} 朝向 {:>3.0f}°  QLabs 坐标 (x={:7.2f}, y={:7.2f})  生成{}  初始{}'.format(
                spec['actorNumber'], spec['name'], spec['controls'], spec['facing'],
                to_qlabs_yaw(spec['rotation']), location[0], location[1],
                _status_text(status), initialColor))
    return handles

def set_light_color(lightHandles, colorName, actorNumbers=None):
    """
    切换信号灯颜色（演示红灯停车 / 绿灯通行时可在运行中调用）。
    参数：
        lightHandles  setup() 返回的 scene['traffic_lights']
        colorName     'green' / 'red' / 'yellow' / 'none'
        actorNumbers  None 表示全部信号灯；或传入列表（如 [1, 4]）只切换指定信号灯
    返回：
        成功设置的信号灯数量
    """
    count = 0
    for item in lightHandles:
        if actorNumbers is not None and item['actorNumber'] not in actorNumbers:
            continue
        handle = item['handle']
        if handle.set_color(color=light_color_constant(handle, colorName),
                            waitForConfirmation=True):
            count += 1
    return count
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 锥桶生成

def spawn_cones(qlabs, verbose=True):
    """生成 4 个交通锥桶（QLabsTrafficCone.spawn_id_degrees）。"""
    if verbose:
        print('\n[要素：锥桶] 生成锥桶（QLabsTrafficCone.spawn_id_degrees）：')
    cone = QLabsTrafficCone(qlabs)
    handles = []
    for i, position in enumerate(CONE_POSITIONS):
        status = cone.spawn_id_degrees(
            actorNumber=200 + i,
            location=position,
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            configuration=0,
            waitForConfirmation=True,
        )
        handle = dict(actorNumber=200 + i, position=position, status=status)
        handles.append(handle)
        if verbose:
            print('  [{}/{}] actor={}  pos=({:.1f}, {:.1f}, {:.2f})  {}'.format(
                i + 1, len(CONE_POSITIONS), 200 + i,
                position[0], position[1], position[2],
                'OK' if status in (None, 0) else 'FAIL({})'.format(status)))
    return handles

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 车辆与实时模型
def spawn_vehicle(qlabs, initialPosition=None, initialOrientation=None, verbose=True):
    """
    生成 QCar2（actorNumber=0）、自由相机并接管视角。
    参数：
        qlabs              已连接的 QuanserInteractiveLabs 对象
        initialPosition    SDCS 世界坐标 [x, y, z]；None 表示取路网起点（节点 0）
        initialOrientation QLabs 车辆姿态（弧度）[roll, pitch, yaw]；None 表示取路网起点航向
    返回：
        vehicleHandle（QLabsQCar2），同时返回使用的位姿
    """
    if initialPosition is None or initialOrientation is None:
        defaultPosition, defaultOrientation = default_vehicle_pose()
        if initialPosition is None:
            initialPosition = defaultPosition
        if initialOrientation is None:
            initialOrientation = defaultOrientation
    location = to_qlabs_position(initialPosition)
    vehicle = QLabsQCar2(qlabs)
    vehicle.spawn_id(
        actorNumber=0,
        location=location,
        rotation=initialOrientation,
        waitForConfirmation=True,
    )
    camera = QLabsFreeCamera(qlabs)
    camera.spawn()
    vehicle.possess()
    if verbose:
        print('\n[车辆] QCar2 生成于 QLabs (x={:.2f}, y={:.2f})，航向 {:.1f}°'
              '（对应 SDCS 坐标 x={:.3f}, y={:.3f}, 航向 {:.1f}°）'.format(
                  location[0], location[1], math.degrees(initialOrientation[2]),
                  float(initialPosition[0]), float(initialPosition[1]),
                  math.degrees(initialOrientation[2])))
    return vehicle, location, initialOrientation
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 行人生成

def spawn_people(qlabs, scale, verbose=True):
    """生成 5 个行人，每个在各自的斑马线上来回行走。"""
    people = []
    if verbose:
        print('\n[要素三] 生成行人（QLabsPerson.spawn_id）：')
    for i, data in enumerate(PEDESTRIAN_PATHS):
        person = QLabsPerson(qlabs)
        location = [data['start'][0] * scale, data['start'][1] * scale, data['start'][2]]
        status = person.spawn_id(
            actorNumber=100 + i,
            location=location,
            rotation=[0, 0, data['yaw']],
            scale=[1, 1, 1],
            configuration=6,
            waitForConfirmation=True,
        )
        handle = {
            'person': person,
            'start': [data['start'][0] * scale, data['start'][1] * scale, data['start'][2]],
            'end': [data['end'][0] * scale, data['end'][1] * scale, data['end'][2]],
            'yaw': data['yaw'],
            'name': data['name'],
            'status': status,
        }
        people.append(handle)
        if verbose:
            print('  [{}/{}] {:<10s}  start=({:.1f}, {:.1f})  end=({:.1f}, {:.1f})  {}'.format(
                i + 1, len(PEDESTRIAN_PATHS), data['name'],
                handle['start'][0], handle['start'][1],
                handle['end'][0], handle['end'][1],
                'OK' if status in (None, 0) else 'FAIL({})'.format(status)))
    return people


def _pedestrian_loop(data):
    """单个行人循环：在 start 和 end 之间往返走动。"""
    person = data['person']
    while True:
        person.move_to(location=data['end'], speed=person.WALK, waitForConfirmation=True)
        time.sleep(0.4)
        person.move_to(location=data['start'], speed=person.WALK, waitForConfirmation=True)
        time.sleep(0.3)


def start_people(people):
    """为每个行人启动独立的 daemon 线程，实现并行往返。"""
    for p in people:
        t = threading.Thread(target=_pedestrian_loop, args=(p,))
        t.daemon = True
        t.start()

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 场景搭建主函数
def setup(
        initialPosition=None,
        initialOrientation=None,
        spawnVehicle=True,
        initialLightColor=INITIAL_LIGHT_COLOR,
        startRealTimeModel=True,
        verbose=True,
):
    """
    连接 QLabs 并搭建结题演示场景（斑马线 5 条 + 信号灯 4 个，可选生成车辆）。
    参数：
        initialPosition        车辆初始位置（SDCS 世界坐标）；None 取路网起点
        initialOrientation     车辆初始姿态（弧度）；None 取路网起点航向
        spawnVehicle           是否生成车辆（含自由相机与实时模型）
        initialLightColor      信号灯初始颜色 'green' / 'red' / 'yellow' / 'none'
        startRealTimeModel     是否启动 QCar2 实时模型（仿真车辆可被 pal 库控制）
        verbose                是否打印搭建明细
    返回：
        scene 字典：qlabs / crosswalks / traffic_lights / vehicle / ...
    """
    qlabs = QuanserInteractiveLabs()
    print('Connecting to QLabs...')
    if not qlabs.open('localhost'):
        print('Unable to connect to QLabs')
        sys.exit()
    print('Connected to QLabs')
    # 清理上一次生成的 actor 与实时模型，保证场景干净
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    QLabsSystem(qlabs).set_title_string(TITLE)
    # ---- 要素一：斑马线 5 条 ----
    crosswalks = spawn_crosswalks(qlabs, verbose=verbose)
    # ---- 要素二：信号灯 4 个 ----
    trafficLights = spawn_traffic_lights(qlabs, initialColor=initialLightColor, verbose=verbose)
    # ---- 锥桶 4 个 ----
    cones = spawn_cones(qlabs, verbose=verbose)
    # ---- 要素三：行人 5 个 ----
    people = spawn_people(qlabs, QLABS_SCALE, verbose=verbose)
    start_people(people)
    # ---- 车辆与实时模型 ----
    vehicle = None
    vehicleLocation = None
    vehicleOrientation = None
    if spawnVehicle:
        vehicle, vehicleLocation, vehicleOrientation = spawn_vehicle(
            qlabs, initialPosition, initialOrientation, verbose=verbose)
        if startRealTimeModel:
            QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
            if verbose:
                print('[实时模型] QCar2 实时模型已启动，可运行主程序进行识别-决策-控制演示。')
    failedCrosswalks = [item['name'] for item in crosswalks
                        if item['status'] not in (None, 0)]
    failedLights = [item['name'] for item in trafficLights
                    if item['status'] not in (None, 0)]
    failedPeople = [item['name'] for item in people
                    if item['status'] not in (None, 0)]
    failedCones = [item['position'] for item in cones
                    if item['status'] not in (None, 0)]
    if verbose:
        print('\n场景搭建完成：斑马线 {}/{} 条，信号灯 {}/{} 个，锥桶 {}/{} 个，行人 {}/{} 个{}。'.format(
            len(crosswalks) - len(failedCrosswalks), len(crosswalks),
            len(trafficLights) - len(failedLights), len(trafficLights),
            len(cones) - len(failedCones), len(cones),
            len(people) - len(failedPeople), len(people),
            '，车辆 1 台' if vehicle is not None else '，未生成车辆'))
        if failedCrosswalks or failedLights or failedCones or failedPeople:
            print('[警告] 生成失败的要素：斑马线 {}  信号灯 {}  锥桶 {}  行人 {}'.format(failedCrosswalks, failedLights, failedCones, failedPeople))
    return {
        'qlabs': qlabs,
        'crosswalks': crosswalks,
        'traffic_lights': trafficLights,
        'cones': cones,
        'people': people,
        'vehicle': vehicle,
        'vehicle_location': vehicleLocation,
        'vehicle_orientation': vehicleOrientation,
        'initial_light_color': initialLightColor,
        'scale': QLABS_SCALE,
    }

def terminate():
    """停止 QCar2 实时模型（与 0/qlabs_setup_task01.py 的 terminate 保持一致）。"""
    QLabsRealTime().terminate_real_time_model(rtmodels.QCAR2)
#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 独立运行入口
def parse_arguments():
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(
        description='结题演示 - 虚拟场景搭建：斑马线 5 条 + 信号灯 4 个（可选生成车辆）')
    parser.add_argument('--no-vehicle', dest='spawnVehicle', action='store_false',
                        help='只搭建交通要素，不生成车辆')
    parser.add_argument('--no-rt', dest='startRealTimeModel', action='store_false',
                        help='不启动 QCar2 实时模型（仅静态预览场景）')
    parser.add_argument('--color', dest='color', default=INITIAL_LIGHT_COLOR,
                        choices=list(_LIGHT_COLOR_NAMES.keys()),
                        help='信号灯初始颜色（默认 {}）'.format(INITIAL_LIGHT_COLOR))
    parser.add_argument('--cycle', dest='cycle', action='store_true',
                        help='运行中自动切换信号灯颜色（红/绿交替），便于演示红灯停车与绿灯通行')
    parser.add_argument('--interval', dest='interval', type=float, default=LIGHT_CYCLE_INTERVAL,
                        help='信号灯自动切换间隔（秒，默认 {}）'.format(LIGHT_CYCLE_INTERVAL))
    return parser.parse_args()

def main():
    """搭建场景并保持运行（Ctrl+C 退出）。"""
    args = parse_arguments()
    scene = setup(
        spawnVehicle=args.spawnVehicle,
        initialLightColor=args.color,
        startRealTimeModel=args.startRealTimeModel,
    )
    print('\n提示：')
    print('  1) 场景已就绪，可运行主程序 QCar2_YOLO_Decision_Control.py 进行识别-决策-控制演示；')
    print('  2) 运行中可用 set_light_color(scene["traffic_lights"], "red") 切换信号灯颜色；')
    print('  3) 本脚本会保持运行以维持实时模型，按 Ctrl+C 结束。')
    lightIsGreen = (args.color == 'green')
    nextToggle = time.time() + args.interval
    try:
        while True:
            time.sleep(0.2)
            if args.cycle and time.time() >= nextToggle:
                lightIsGreen = not lightIsGreen
                colorName = 'green' if lightIsGreen else 'red'
                count = set_light_color(scene['traffic_lights'], colorName)
                print('  [{}] 信号灯切换为 {}（{} 个）'.format(
                    time.strftime('%H:%M:%S'), '绿灯' if lightIsGreen else '红灯', count))
                nextToggle = time.time() + args.interval
    except KeyboardInterrupt:
        print('\n收到中断，正在收尾...')
    finally:
        terminate()
        scene['qlabs'].close()
        print('Done!')

if __name__ == '__main__':
    main()
#endregion
