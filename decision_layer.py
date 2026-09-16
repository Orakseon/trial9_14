# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 文件说明与导入

"""
decision_layer.py
决策层：把识别层输出的交通要素转换为车辆行为指令（期望速度 + 绕行横向偏移）。

行为状态机（优先级由高到低）：
    BLIND           感知数据缺失/过期      → 低速谨慎行驶（无法确认安全时按最保守策略处理）
    STOP_RED        红灯                  → 停车等待，直到检测到绿灯（或红灯长时间消失并放行）
    STOP_SIGN       停止标志              → 强制停车 stopSignHoldTime 秒后通过，并对该标志闭锁
    STOP_OBSTACLE   行人/奶牛进入前方走廊  → 停车等待，直到障碍离开走廊 clearConfirmTime 秒
    STOP_CONE       锥桶近距              → 停车观察 coneObserveTime 秒
    BYPASS_CONE     确认安全后绕行         → 限速 + 沿路径法向横向偏移绕开锥桶，通过后自动回正
    SLOW_CROSSWALK  斑马线/前方远处行人    → 限速通过
    SLOW_CONE       远距锥桶              → 减速通过
    CRUISE          正常巡航              → 以巡航速度沿参考路径行驶

设计要点：
    1. 所有判定都带“连续帧确认”与“闭锁/复位”机制，避免单帧误检导致车辆频繁启停；
    2. 停车类决策输出 vRef=0，低速类决策输出限速值，绕行类决策额外输出横向偏移；
    3. 决策层不直接操作执行器，只输出期望量，交由控制层（control_layer.py）平滑跟踪。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from perception_yolo import Detection, DetectionFrame, TrafficLabel

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 决策参数

@dataclass
class DecisionParameters:
    """
    决策层可调参数集合。

    说明：以“面积占比 / 归一化坐标”表达距离与位置，与相机分辨率无关；
          现场标定时可借助主程序画面上的走廊辅助线（draw_corridor_guide）对照调整。
    """
    # ===== 速度决策（m/s）=====
    cruiseSpeed: float = 0.50          # 正常巡航速度
    blindSpeed: float = 0.15           # 感知失效时的谨慎行驶速度
    crosswalkSpeed: float = 0.20       # 通过斑马线/前方远处行人时的速度
    coneSlowSpeed: float = 0.20        # 远处锥桶减速后的速度
    bypassSpeed: float = 0.25          # 绕行锥桶时的速度

    # ===== 感知有效性 =====
    detectionTimeout: float = 1.5      # 识别帧超过该时长未更新则认为感知失效（秒）

    # ===== 走廊几何（归一化）=====
    corridorHalfWidth: float = 0.32    # 前方走廊半宽（画面宽度比例），用于判断目标是否挡路

    # ===== 红绿灯 =====
    trafficLightCorridorHalfWidth: float = 0.38   # 认定“本车道信号灯”的横向半宽
    lightMaxCenterYNorm: float = 0.78             # 灯光类目标中心必须位于画面上部（排除地面误检）
    redMinAreaPercent: float = 0.10               # 红灯最小面积占比（过小视为远处，暂不处理）
    greenMinAreaPercent: float = 0.10             # 绿灯最小面积占比
    lightConfirmFrames: int = 7                   # 红/绿灯连续确认帧数（约 7 帧 ≈ 2.3 秒@3Hz）
    redLostReleaseTime: float = 6.0               # 红灯消失且未见绿灯的宽限时间，超时后谨慎通过（秒）

    # ===== 停止标志 =====
    stopSignMinAreaPercent: float = 0.10          # 停止标志最小面积占比
    stopSignHoldTime: float = 2.5                 # 检测到停止标志后的强制停车时间（秒）
    stopSignResetTime: float = 3.0                # 标志离开视野后解除闭锁所需时间（秒）

    # ===== 动态障碍（行人/奶牛）=====
    dynamicStopBottomYNorm: float = 0.62          # 动态障碍底边越过该值视为“距离已近”
    dynamicStopAreaPercent: float = 0.80          # 动态障碍面积占比超过该值视为“距离已近”
    clearConfirmTime: float = 0.60                # 障碍离开走廊后需保持“畅通”的确认时间（秒）

    # ===== 锥桶绕行 =====
    coneNearBottomYNorm: float = 0.60             # 锥桶底边越过该值视为“距离已近”（0.72→0.60，更早触发）
    coneStopAreaPercent: float = 0.45             # 锥桶面积占比超过该值视为“距离已近”（0.90→0.45，更容易触发）
    coneObserveTime: float = 1.20                 # 近距锥桶的停车观察时间（秒）
    bypassOffset: float = 0.35                    # 绕行横向偏移量（m，沿参考路径左法向为正）
    bypassMinClearance: float = 0.18              # 锥桶横向距离超过该值时认为已有足够绕行空间（归一化）

    # ===== 其它 =====
    dtMax: float = 0.20                           # 单次决策时间步上限（防止线程调度抖动导致计时跳变）


#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 决策输出

@dataclass
class DecisionResult:
    """决策层输出：行为状态 + 期望速度 + 绕行横向偏移 + 可读原因。"""
    state: str                              # 行为状态名称
    vRef: float                             # 期望速度（m/s）
    lateralOffset: float = 0.0              # 绕行横向偏移（m，沿参考路径左法向为正）
    reason: str = ''                        # 决策原因（HUD/日志显示）
    countdown: float = 0.0                  # 剩余停车/观察时间（秒，HUD 显示）
    activeLabels: List[str] = field(default_factory=list)   # 本周期参与决策的交通要素

    @property
    def isStop(self) -> bool:
        """是否属于“停车等待”类决策。"""
        return self.vRef <= 1e-6

    def describe(self) -> str:
        return '{} vRef={:.2f}m/s offset={:+.2f}m {}'.format(
            self.state, self.vRef, self.lateralOffset, self.reason)

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 决策状态机

@dataclass
class TrafficSnapshot:
    """单周期交通要素快照（由 _analyze 生成，供各优先级判据使用）。"""
    redVisible: bool = False
    greenVisible: bool = False
    redLostRelease: bool = False          # 本轮是否因“红灯驶出视野且未见绿灯”而放行
    signVisible: bool = False
    cones: List[Detection] = field(default_factory=list)             # 走廊内的锥桶（按面积降序）
    dynamicInCorridor: List[Detection] = field(default_factory=list)  # 走廊内的动态障碍
    dynamicDanger: List[Detection] = field(default_factory=list)      # 走廊内且距离已近的动态障碍
    crosswalkVisible: bool = False
    stopLineVisible: bool = False
    activeLabels: List[str] = field(default_factory=list)


class TrafficDecision:
    """交通要素决策状态机：输入识别结果，输出期望速度与绕行横向偏移。"""

    # ===== 行为状态常量 =====
    STATE_CRUISE = 'CRUISE'
    STATE_SLOW_CROSSWALK = 'SLOW_CROSSWALK'
    STATE_SLOW_CONE = 'SLOW_CONE'
    STATE_BYPASS_CONE = 'BYPASS_CONE'
    STATE_STOP_CONE = 'STOP_CONE'
    STATE_STOP_PEDESTRIAN = 'STOP_PEDESTRIAN'
    STATE_STOP_SIGN = 'STOP_SIGN'
    STATE_STOP_RED = 'STOP_RED'
    STATE_BLIND = 'BLIND'

    # 状态 → 数值编码（用于 MultiScope 曲线显示）
    STATE_CODES: Dict[str, int] = {
        'CRUISE': 0,
        'SLOW_CROSSWALK': 1,
        'SLOW_CONE': 2,
        'BYPASS_CONE': 3,
        'STOP_CONE': 4,
        'STOP_PEDESTRIAN': 5,
        'STOP_SIGN': 6,
        'STOP_RED': 7,
        'BLIND': 8,
    }

    def __init__(self, params: Optional[DecisionParameters] = None):
        self.p = params if params is not None else DecisionParameters()
        self.transitionLog: List[Tuple[float, str, str]] = []   # (时间, 状态, 原因)
        self._logStart = time.monotonic()
        self.reset()

    # ---------------------------------------------------------------- 初始化
    def reset(self) -> None:
        """复位全部闭锁与计时器（重新开始演示时调用）。"""
        # 红绿灯
        self._redCount = 0
        self._greenCount = 0
        self._redLatched = False
        self._lastRedSeen: Optional[float] = None
        # 停止标志
        self._stopSignLatched = False
        self._stopSignHoldStart: Optional[float] = None
        self._lastStopSignSeen: Optional[float] = None
        # 动态障碍
        self._obstacleStopActive = False
        self._obstacleSource = ''
        self._obstacleClearElapsed = 0.0
        # 锥桶
        self._coneObserveElapsed = 0.0
        self._coneBypassActive = False
        self._coneBypassSign = 0.0
        self._coneClearElapsed = 0.0
        # 输出缓存
        self._prevState = ''
        self._lastResult = DecisionResult(
            state=self.STATE_CRUISE, vRef=self.p.cruiseSpeed, reason='初始化')

    # ---------------------------------------------------------------- 只读接口
    @property
    def state(self) -> str:
        return self._lastResult.state

    @property
    def lastResult(self) -> DecisionResult:
        return self._lastResult

    @property
    def isRedLatched(self) -> bool:
        """红灯闭锁标志（供 HUD 显示交通灯判据状态）。"""
        return self._redLatched

    def codeOf(self, state: str) -> int:
        return self.STATE_CODES.get(state, -1)

    def historyText(self, limit: Optional[int] = None) -> str:
        """状态变化历史摘要（用于演示结束后的打印；limit 为 None 表示全部）。"""
        entries = self.transitionLog if limit is None else self.transitionLog[-limit:]
        lines = []
        for timestamp, state, reason in entries:
            lines.append('  [{:6.2f}s] {:<16s} {}'.format(timestamp - self._logStart, state, reason))
        return '\n'.join(lines) if lines else '  （无状态变化）'

    # ---------------------------------------------------------------- 主入口
    def update(self, detectionFrame: Optional[DetectionFrame], dt: float,
               speed: float = 0.0) -> DecisionResult:
        """
        每个控制周期调用一次。

        参数：
            detectionFrame  识别层最新一帧结果（None 表示尚未产生）
            dt              本周期时间步（秒）
            speed           当前车速（m/s，供将来扩展 TTC 等判据）
        返回：
            DecisionResult
        """
        now = time.monotonic()
        dt = float(min(max(dt, 1e-3), self.p.dtMax))

        # ---- 1. 感知有效性：数据缺失或过期则保守低速行驶 ----
        if detectionFrame is None or not detectionFrame.isValid(self.p.detectionTimeout, now):
            return self._publish(
                self.STATE_BLIND, self.p.blindSpeed, 0.0,
                '感知数据缺失或过期，低速谨慎行驶')

        # ---- 2. 生成本周期交通要素快照（含各类闭锁与计时更新）----
        snapshot = self._analyze(detectionFrame, now, dt)

        # ---- 3. 按优先级判定行为 ----
        if self._redLatched:
            return self._redDecision(snapshot)

        signResult = self._stopSignDecision(snapshot, now)
        if signResult is not None:
            return signResult

        obstacleResult = self._obstacleDecision(snapshot)
        if obstacleResult is not None:
            return obstacleResult

        coneResult = self._coneDecision(snapshot, dt)
        if coneResult is not None:
            return coneResult

        if snapshot.crosswalkVisible or snapshot.dynamicInCorridor:
            return self._publish(
                self.STATE_SLOW_CROSSWALK, self.p.crosswalkSpeed, 0.0,
                '斑马线或前方交通要素，减速通过',
                activeLabels=snapshot.activeLabels)

        return self._publish(
            self.STATE_CRUISE, self.p.cruiseSpeed, 0.0,
            '前方无影响目标，正常巡航',
            activeLabels=snapshot.activeLabels)

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # ---------------------------------------------------------------- 要素解析
    def _analyze(self, detFrame: DetectionFrame, now: float, dt: float) -> TrafficSnapshot:
        """解析本帧识别结果，更新各类闭锁/计时器，并生成本周期快照。"""
        p = self.p
        snapshot = TrafficSnapshot()

        # ===== 红绿灯：连续帧确认 + 闭锁（避免单帧误检导致启停抖动）=====
        reds = [d for d in detFrame.byLabel(TrafficLabel.RED)
                if self._isSignalCandidate(d, p.redMinAreaPercent)]
        greens = [d for d in detFrame.byLabel(TrafficLabel.GREEN)
                  if self._isSignalCandidate(d, p.greenMinAreaPercent)]
        snapshot.redVisible = len(reds) > 0
        snapshot.greenVisible = len(greens) > 0

        if snapshot.redVisible:
            self._redCount += 1
            self._greenCount = 0
            self._lastRedSeen = now
        elif snapshot.greenVisible:
            self._greenCount += 1
            self._redCount = 0
        else:
            self._redCount = max(0, self._redCount - 1)
            self._greenCount = max(0, self._greenCount - 1)

        if self._redCount >= p.lightConfirmFrames and not self._redLatched:
            self._redLatched = True
            self._logTransition(now, self.STATE_STOP_RED,
                                '红灯连续 {} 帧确认，进入停车等待'.format(self._redCount))

        if self._redLatched and self._greenCount >= p.lightConfirmFrames:
            self._redLatched = False
            self._logTransition(now, self.STATE_CRUISE,
                                '绿灯连续 {} 帧确认，放行'.format(self._greenCount))

        # 红灯长时间消失且未见绿灯（灯光已驶出视野）：宽限时间后谨慎通过
        if (self._redLatched and not snapshot.redVisible and not snapshot.greenVisible
                and self._lastRedSeen is not None
                and (now - self._lastRedSeen) > p.redLostReleaseTime):
            self._redLatched = False
            snapshot.redLostRelease = True
            self._logTransition(now, self.STATE_CRUISE,
                                '红灯驶出视野且未见绿灯，宽限超时后谨慎通过')

        # ===== 停止标志：进入视野即触发停车，闭锁至标志离开视野 =====
        signs = [d for d in detFrame.byLabel(TrafficLabel.STOP_SIGN)
                 if self._isSignalCandidate(d, p.stopSignMinAreaPercent)]
        snapshot.signVisible = len(signs) > 0
        if snapshot.signVisible:
            self._lastStopSignSeen = now
            if not self._stopSignLatched:
                self._stopSignLatched = True
                self._stopSignHoldStart = now
                self._logTransition(now, self.STATE_STOP_SIGN,
                                    '检测到停止标志，开始停车确认')
        elif (self._stopSignLatched and self._lastStopSignSeen is not None
              and (now - self._lastStopSignSeen) > p.stopSignResetTime):
            # 标志离开视野足够久，解除闭锁，允许下一圈再次触发
            self._stopSignLatched = False
            self._stopSignHoldStart = None

        # ===== 动态障碍（行人/奶牛）：走廊内且距离已近则停车等待 =====
        for label in (TrafficLabel.PEOPLE, TrafficLabel.COW):
            for det in detFrame.byLabel(label):
                if not det.isInCorridor(p.corridorHalfWidth):
                    continue
                snapshot.dynamicInCorridor.append(det)
                if det.isNear(p.dynamicStopBottomYNorm, p.dynamicStopAreaPercent):
                    snapshot.dynamicDanger.append(det)

        if snapshot.dynamicDanger:
            if not self._obstacleStopActive:
                self._obstacleStopActive = True
                self._obstacleSource = snapshot.dynamicDanger[0].label
                self._logTransition(now, self.STATE_STOP_PEDESTRIAN,
                                    '{}进入前方走廊，停车等待'.format(self._obstacleSource))
            self._obstacleClearElapsed = 0.0
        elif self._obstacleStopActive:
            # 障碍已离开走廊，需保持“畅通”一段时间后才恢复行驶
            self._obstacleClearElapsed += dt
            if self._obstacleClearElapsed >= p.clearConfirmTime:
                self._obstacleStopActive = False
                self._obstacleSource = ''
                self._obstacleClearElapsed = 0.0

        # ===== 锥桶（仅统计落在前方走廊内的）=====
        snapshot.cones = [d for d in detFrame.byLabel(TrafficLabel.CONE)
                          if d.isInCorridor(p.corridorHalfWidth)]

        # ===== 斑马线 / 停止线 =====
        snapshot.crosswalkVisible = len(detFrame.byLabel(TrafficLabel.CROSSWALK)) > 0
        snapshot.stopLineVisible = len(detFrame.byLabel(TrafficLabel.STOP_LINE)) > 0

        # ===== 本周期出现的全部类别（HUD/日志用）=====
        snapshot.activeLabels = detFrame.labels()
        return snapshot

    def _isSignalCandidate(self, det: Detection, minAreaPercent: float) -> bool:
        """
        判断检测结果是否可视为“本车前方车道内的信号灯/标志”。

        条件：面积占比达到阈值（足够近）、横向位于本车道走廊内、纵向位于画面上部
              （排除地面上的同色误检）。
        """
        return (det.areaPercent >= minAreaPercent
                and abs(det.lateralErrorNorm) <= self.p.trafficLightCorridorHalfWidth
                and det.centerYNorm <= self.p.lightMaxCenterYNorm)

    # ---------------------------------------------------------------- 各类行为判据
    def _redDecision(self, snapshot: TrafficSnapshot) -> DecisionResult:
        """红灯闭锁期间：保持停车，并给出当前判据说明。"""
        if snapshot.redVisible:
            reason = '红灯，停车等待绿灯'
            if snapshot.stopLineVisible:
                reason += '（已捕获停止线）'
        elif snapshot.greenVisible:
            reason = '绿灯亮起，等待确认后放行'
        else:
            reason = '红灯闭锁中（信号灯暂不可见），保持停车'
        return self._publish(self.STATE_STOP_RED, 0.0, 0.0, reason,
                             activeLabels=snapshot.activeLabels)

    def _stopSignDecision(self, snapshot: TrafficSnapshot,
                          now: float) -> Optional[DecisionResult]:
        """停止标志：强制停车 stopSignHoldTime 秒，计时结束后放行（闭锁保持到标志离开视野）。"""
        if self._stopSignHoldStart is None:
            return None
        elapsed = now - self._stopSignHoldStart
        if elapsed < self.p.stopSignHoldTime:
            remaining = self.p.stopSignHoldTime - elapsed
            return self._publish(
                self.STATE_STOP_SIGN, 0.0, 0.0,
                '停止标志：停车确认（剩余 {:.1f}s）'.format(remaining),
                countdown=remaining, activeLabels=snapshot.activeLabels)
        # 停车确认结束：允许继续行驶（闭锁仍保持，避免同一标志重复触发）
        self._stopSignHoldStart = None
        self._logTransition(now, self.STATE_CRUISE, '停止标志停车确认完成，继续行驶')
        return None

    def _obstacleDecision(self, snapshot: TrafficSnapshot) -> Optional[DecisionResult]:
        """动态障碍（行人/奶牛）位于前方走廊且距离已近：停车等待其离开。"""
        if not self._obstacleStopActive:
            return None
        remaining = max(self.p.clearConfirmTime - self._obstacleClearElapsed, 0.0)
        return self._publish(
            self.STATE_STOP_PEDESTRIAN, 0.0, 0.0,
            '{}在车前方，停车等待其离开走廊'.format(self._obstacleSource or '动态障碍'),
            countdown=remaining, activeLabels=snapshot.activeLabels)

    def _coneDecision(self, snapshot: TrafficSnapshot, dt: float) -> Optional[DecisionResult]:
        """
        锥桶处理：
            1. 近距且位于本车轨迹上   → 停车观察 coneObserveTime 秒；
               观察结束且前方无动态障碍 → 切换为绕行（限速 + 横向偏移）
            2. 近距但明显偏离本车轨迹 → 无需停车，直接带偏移通过
            3. 远距                   → 减速通过
            4. 走廊内无锥桶           → 绕行收尾（保持偏移约 clearConfirmTime）后回正
        """
        p = self.p
        cones = snapshot.cones

        if cones:
            self._coneClearElapsed = 0.0
            nearCone = next((d for d in cones
                             if d.isNear(p.coneNearBottomYNorm, p.coneStopAreaPercent)), None)

            if nearCone is not None and not self._coneBypassActive:
                if abs(nearCone.lateralErrorNorm) > p.bypassMinClearance:
                    # 锥桶明显偏在一侧，本车行驶轨迹基本不受影响 → 直接带偏移通过
                    self._coneBypassActive = True
                    self._coneBypassSign = self._bypassSign(nearCone)
                    self._logTransition(
                        time.monotonic(), self.STATE_BYPASS_CONE,
                        '锥桶偏离本车轨迹（横向 {:+.2f}），直接绕行'.format(nearCone.lateralErrorNorm))
                else:
                    # 锥桶就在车前 → 先停车观察
                    self._coneObserveElapsed += dt
                    if self._coneObserveElapsed < p.coneObserveTime:
                        remaining = p.coneObserveTime - self._coneObserveElapsed
                        return self._publish(
                            self.STATE_STOP_CONE, 0.0, 0.0,
                            '锥桶近距，停车观察（剩余 {:.1f}s）'.format(remaining),
                            countdown=remaining, activeLabels=snapshot.activeLabels)
                    # 观察结束：确认前方无动态障碍后启动绕行
                    if not snapshot.dynamicInCorridor:
                        self._coneBypassActive = True
                        self._coneBypassSign = self._bypassSign(nearCone)
                        self._logTransition(
                            time.monotonic(), self.STATE_BYPASS_CONE,
                            '确认安全，向{}侧绕行（横向偏移 {:+.2f} m）'.format(
                                '左' if self._coneBypassSign > 0 else '右',
                                self._coneBypassSign * p.bypassOffset))

            if self._coneBypassActive:
                offset = self._coneBypassSign * p.bypassOffset
                return self._publish(
                    self.STATE_BYPASS_CONE, p.bypassSpeed, offset,
                    '绕行锥桶，横向偏移 {:+.2f} m'.format(offset),
                    activeLabels=snapshot.activeLabels)

            return self._publish(
                self.STATE_SLOW_CONE, p.coneSlowSpeed, 0.0,
                '前方远处锥桶，减速通过', activeLabels=snapshot.activeLabels)

        # ---- 走廊内已无锥桶 ----
        if self._coneBypassActive:
            self._coneClearElapsed += dt
            if self._coneClearElapsed < p.clearConfirmTime:
                offset = self._coneBypassSign * p.bypassOffset
                return self._publish(
                    self.STATE_BYPASS_CONE, p.bypassSpeed, offset,
                    '绕行收尾，等待障碍完全让开', activeLabels=snapshot.activeLabels)
            # 绕行完成：偏移交由控制层平滑回正
            self._coneBypassActive = False
            self._coneBypassSign = 0.0
            self._logTransition(time.monotonic(), self.STATE_CRUISE, '锥桶绕行完成，回归巡航')
        self._coneObserveElapsed = 0.0
        return None

    def _bypassSign(self, cone: Detection) -> float:
        """
        计算绕行偏移方向（沿参考路径左法向为正）。

        锥桶偏左 → 向右让行（负偏移）；锥桶偏右 → 向左让行（正偏移）。
        """
        return 1.0 if cone.lateralErrorNorm >= 0 else -1.0

    # ---------------------------------------------------------------- 输出封装
    def _publish(self, state: str, vRef: float, lateralOffset: float = 0.0,
                 reason: str = '', countdown: float = 0.0,
                 activeLabels: Optional[List[str]] = None) -> DecisionResult:
        """组装决策输出，并在状态发生变化时记录日志。"""
        result = DecisionResult(
            state=state,
            vRef=float(vRef),
            lateralOffset=float(lateralOffset),
            reason=reason,
            countdown=float(max(countdown, 0.0)),
            activeLabels=list(activeLabels) if activeLabels else [],
        )
        self._lastResult = result
        if state != self._prevState:
            self._logTransition(time.monotonic(), state, reason)
            self._prevState = state
        return result

    def _logTransition(self, timestamp: float, state: str, reason: str) -> None:
        """记录状态/闭锁事件，最多保留 500 条。"""
        self.transitionLog.append((timestamp, state, reason))
        if len(self.transitionLog) > 500:
            del self.transitionLog[:-500]

#endregion





