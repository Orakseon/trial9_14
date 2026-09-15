# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : 文件说明与导入

"""
control_layer.py
控制层：把决策层给出的期望量转换为 QCar 的执行器指令。

包含：
    SpeedRampLimiter   期望速度斜坡限制（限制加/减速度，避免停车/起步突兀）
    SpeedController    速度 PI 控制器（带积分抗饱和与油门限幅）
    OffsetShaper       绕行横向偏移斜坡限制（避免转向指令突变）
    SteeringController Stanley 路径跟踪控制器（支持横向偏移绕行，增益随速度自适应）

约定：
    lateralOffset 沿参考路径左法向为正（正值表示参考点向路径左侧平移）。
"""

from __future__ import annotations

import numpy as np

from pal.utilities.math import wrap_to_pi

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 速度相关模块

class SpeedRampLimiter:
    """
    期望速度斜坡限制器。

    决策层会在“巡航 0.5 m/s”与“停车 0 m/s”之间切换，若直接把阶跃给到 PI 控制器，
    会造成急加速/急减速。本类按给定加速度/减速度上限对期望速度做斜坡处理。
    """

    def __init__(self, accelLimit: float = 0.45, decelLimit: float = 0.90,
                 initialSpeed: float = 0.0):
        """
        参数：
            accelLimit    允许的最大加速度（m/s²）
            decelLimit    允许的最大减速度（m/s²）
            initialSpeed  初始期望速度（m/s）
        """
        self.accelLimit = float(abs(accelLimit))
        self.decelLimit = float(abs(decelLimit))
        self.initialSpeed = float(initialSpeed)
        self.value = float(initialSpeed)

    def reset(self, initialSpeed: float = None) -> None:
        """复位为初始速度。"""
        self.value = self.initialSpeed if initialSpeed is None else float(initialSpeed)

    def update(self, vTarget: float, dt: float) -> float:
        """把一个周期的期望速度限制在可达范围内，返回平滑后的期望速度。"""
        vTarget = float(vTarget)
        dt = float(max(dt, 1e-4))
        maxRise = self.accelLimit * dt
        maxFall = self.decelLimit * dt
        delta = np.clip(vTarget - self.value, -maxFall, maxRise)
        self.value = self.value + float(delta)
        return self.value


class SpeedController:
    """
    QCar 速度 PI 控制器。

    u = K_p * e + K_i * ∫e dt ，输出为油门（throttle），并做限幅与积分抗饱和。
    """

    def __init__(self, kp: float = 0.1, ki: float = 1.0, maxThrottle: float = 0.1,
                 integratorLimit: float = 1.0, stopSpeedThreshold: float = 0.03,
                 integralPurgeError: float = 0.08):
        """
        参数：
            kp                  比例增益
            ki                  积分增益
            maxThrottle         油门输出限幅（QCar 推荐不超过 0.1~0.2）
            integratorLimit     积分项限幅（防止长时间停车时积分饱和）
            stopSpeedThreshold  停车保持判据的速度阈值（m/s）：目标速度为 0 且车速低于该值时直接给零油门
            integralPurgeError  积分净化阈值（m/s）：车速与目标速度之差超过该值且积分方向相反时清零积分
        """
        self.kp = float(kp)
        self.ki = float(ki)
        self.maxThrottle = float(maxThrottle)
        self.integratorLimit = float(abs(integratorLimit))
        self.stopSpeedThreshold = float(abs(stopSpeedThreshold))
        self.integralPurgeError = float(abs(integralPurgeError))
        self.ei = 0.0

    def reset(self) -> None:
        """清零积分项。"""
        self.ei = 0.0

    def update(self, v: float, vRef: float, dt: float) -> float:
        """返回本周期油门指令。"""
        dt = float(max(dt, 1e-4))
        error = float(vRef) - float(v)
        # 停车保持：目标速度为零且车辆已基本停住 → 直接零油门并清零积分，
        # 避免 PI 反向积分输出负油门导致车辆倒退
        if abs(float(vRef)) <= 1e-3 and abs(float(v)) < self.stopSpeedThreshold:
            self.ei = 0.0
            return 0.0
        # 积分净化：仅在“确有急减速/急加速需求”时清掉方向相反的积分残留，
        # 保证制动/起步响应及时；日常速度跟踪中的小幅积分保留，用于消除稳态误差
        if error < -self.integralPurgeError and self.ei > 0.0:
            self.ei = 0.0
        elif error > self.integralPurgeError and self.ei < 0.0:
            self.ei = 0.0
        # 先积分后限幅：积分项保持在允许范围内
        self.ei = float(np.clip(self.ei + error * dt,
                                -self.integratorLimit, self.integratorLimit))
        u = self.kp * error + self.ki * self.ei
        u = float(np.clip(u, -self.maxThrottle, self.maxThrottle))
        # 抗饱和：输出饱和且误差方向与积分增长方向一致时，回退本次积分
        if (u >= self.maxThrottle and error > 0) or (u <= -self.maxThrottle and error < 0):
            self.ei = float(np.clip(self.ei - error * dt,
                                    -self.integratorLimit, self.integratorLimit))
        return u


class OffsetShaper:
    """绕行横向偏移斜坡限制器：把决策层的偏移目标平滑成连续轨迹。"""

    def __init__(self, rateLimit: float = 0.60, initialOffset: float = 0.0):
        """
        参数：
            rateLimit      偏移变化率上限（m/s）
            initialOffset  初始偏移（m）
        """
        self.rateLimit = float(abs(rateLimit))
        self.value = float(initialOffset)

    def reset(self, initialOffset: float = 0.0) -> None:
        self.value = float(initialOffset)

    def update(self, target: float, dt: float) -> float:
        """返回限制变化率后的横向偏移。"""
        dt = float(max(dt, 1e-4))
        maxStep = self.rateLimit * dt
        self.value = self.value + float(np.clip(float(target) - self.value, -maxStep, maxStep))
        return self.value

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 转向控制模块

class SteeringController:
    """
    Stanley 路径跟踪转向控制器（在 0/QCar2_Steering_Control.py 基准实现上扩展）。

    扩展点：
        1. lateralOffset 参数：把参考路径上的最近点沿路径左法向平移，
           用于锥桶绕行等横向避让动作（不改变 Stanley 本身的收敛特性）；
        2. speedFloor 参数：Stanley 前馈项 arctan2(k*ect, speed) 在低速时增益极大，
           停车等待（v≈0）时会产生剧烈转向，这里对速度分母做下限保护。
    """

    def __init__(self, waypoints, k: float = 1.0, cyclic: bool = True,
                 maxSteeringAngle: float = np.pi / 6, speedFloor: float = 0.10):
        """
        参数：
            waypoints        参考路径点阵，形状 (2, N)
            k                Stanley 增益 K_stanley
            cyclic           路径是否闭环（本实验路线 [0, 23, 0] 为闭环，取 True）
            maxSteeringAngle 前轮转角限幅（rad）
            speedFloor       Stanley 前馈项速度分母下限（m/s）
        """
        self.wp = np.asarray(waypoints, dtype=float)
        self.N = self.wp.shape[1]
        self.k = float(k)
        self.cyclic = bool(cyclic)
        self.maxSteeringAngle = float(maxSteeringAngle)
        self.speedFloor = float(max(abs(speedFloor), 1e-3))
        self.wpi = 0
        self.p_ref = np.zeros(2)
        self.th_ref = 0.0

    def reset(self) -> None:
        """复位路径索引与参考点记录。"""
        self.wpi = 0
        self.p_ref = np.zeros(2)
        self.th_ref = 0.0

    def update(self, p, th: float, speed: float,
               lateralOffset: float = 0.0) -> float:
        """
        计算前轮转角指令。

        参数：
            p             车辆控制点位置（EKF 估计位置 + 前视偏移），np.array([x, y])
            th            车辆航向（rad）
            speed         当前车速（m/s）
            lateralOffset 绕行横向偏移（m，沿路径左法向为正）
        返回：
            前轮转角指令（rad，已限幅）
        """
        p = np.asarray(p, dtype=float).reshape(2)
        wp1 = self.wp[:, int(np.mod(self.wpi, self.N - 1))]
        wp2 = self.wp[:, int(np.mod(self.wpi + 1, self.N - 1))]
        segment = wp2 - wp1
        segmentLength = float(np.linalg.norm(segment))
        if segmentLength < 1e-9:
            return 0.0
        direction = segment / segmentLength
        tangent = float(np.arctan2(direction[1], direction[0]))

        # 当前控制点在路径段上的投影
        projection = float(np.dot(p - wp1, direction))
        if projection >= segmentLength:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1
        referencePoint = wp1 + direction * projection

        # 绕行：把参考点沿路径左法向平移（左法向 = 路径切线逆时针旋转 90°）
        if abs(lateralOffset) > 1e-6:
            leftNormal = np.array([-direction[1], direction[0]])
            referencePoint = referencePoint + leftNormal * float(lateralOffset)

        crossTrack = referencePoint - p
        crossTrackAngle = wrap_to_pi(np.arctan2(crossTrack[1], crossTrack[0]) - tangent)
        crossTrackError = float(np.linalg.norm(crossTrack) * np.sign(crossTrackAngle))
        headingError = wrap_to_pi(tangent - th)

        self.p_ref = referencePoint
        self.th_ref = tangent

        # Stanley 控制律：前馈（航向误差）+ 反馈（横向误差随速度衰减）
        speedDenominator = max(abs(float(speed)), self.speedFloor)
        delta = wrap_to_pi(
            headingError + np.arctan2(self.k * crossTrackError, speedDenominator))
        return float(np.clip(delta, -self.maxSteeringAngle, self.maxSteeringAngle))

#endregion

