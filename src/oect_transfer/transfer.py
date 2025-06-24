# transfer.py
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from typing import List, Optional


@dataclass
class Sequence:
    raw: NDArray[np.float64]
    forward: NDArray[np.float64]
    reverse: NDArray[np.float64]


@dataclass
class Point:
    raw: float
    where: str
    forward: float
    reverse: float


class Transfer:
    def __init__(
        self, x: NDArray[np.float64], y: NDArray[np.float64], device_type: str = "N"
    ) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Validate input arrays
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays.")
        if x.shape[0] != y.shape[0]:
            print(x.shape[0], y.shape[0])
            print(x, y)
            raise ValueError("x and y must have the same length.")
        if x.size == 0 or y.size == 0:
            raise ValueError("x and y must not be empty.")
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError("x and y must not contain NaN values.")
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            raise ValueError("x and y must not contain infinite values.")

        # 修复转折点查找逻辑，添加错误处理
        self.tp_idx = (len(x)-1) // 2

        self.Vg = Sequence(raw=x, forward=x[: self.tp_idx + 1], reverse=x[self.tp_idx:])
        self.I = Sequence(raw=y, forward=y[: self.tp_idx + 1], reverse=y[self.tp_idx:])

        self.gm = self._compute_gm()
        self.absgm_max = self._compute_absgm_max()
        self.absI_max = self._compute_absI_max()
        self.absI_min = self._compute_absI_min()
        self.gm_max = self._compute_gm_max()
        self.I_max = self._compute_I_max()
        self.I_min = self._compute_I_min()
        self.Von = self._compute_Von(device_type=device_type)

    def _compute_gm(self) -> Sequence:
        """
        计算跨导 gm = dy/dx，用 safe_diff 方法处理
        :return: Sequence 包含 raw / forward / reverse 的 gm
        """
        return Sequence(
            raw=self.safe_diff(self.I.raw, self.Vg.raw),
            forward=self.safe_diff(self.I.forward, self.Vg.forward),
            reverse=self.safe_diff(self.I.reverse, self.Vg.reverse),
        )

    def _compute_absgm_max(self) -> Point:
        """
        计算gm的绝对值的最大点
        :return: Point 包含 raw / forward / reverse 的 absgm_max
        """
        if len(self.gm.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        absgm_max_index = np.abs(self.gm.raw).argmax()

        if absgm_max_index < self.tp_idx:
            absgm_max_where = "forward"
        elif absgm_max_index == self.tp_idx:
            absgm_max_where = "turning_point"
        else:
            absgm_max_where = "reverse"
            
        return Point(
            raw=float(np.abs(self.gm.raw).max()),
            where=absgm_max_where,
            forward=float(np.abs(self.gm.forward).max()) if len(self.gm.forward) > 0 else 0.0,
            reverse=float(np.abs(self.gm.reverse).max()) if len(self.gm.reverse) > 0 else 0.0,
        )
    
    def _compute_gm_max(self) -> Point:
        """
        计算最大跨导点
        :return: Point 包含 raw / forward / reverse 的 gm_max
        """
        if len(self.gm.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        gm_max_index = self.gm.raw.argmax()

        if gm_max_index < self.tp_idx:
            gm_max_where = "forward"
        elif gm_max_index == self.tp_idx:
            gm_max_where = "turning_point"
        else:
            gm_max_where = "reverse"
            
        return Point(
            raw=float(self.gm.raw.max()),
            where=gm_max_where,
            forward=float(self.gm.forward.max()) if len(self.gm.forward) > 0 else 0.0,
            reverse=float(self.gm.reverse.max()) if len(self.gm.reverse) > 0 else 0.0,
        )
    
    def _compute_gm_min(self) -> Point:
        """
        计算最小跨导点
        :return: Point 包含 raw / forward / reverse 的 gm_min
        """
        if len(self.gm.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        gm_min_index = self.gm.raw.argmin()

        if gm_min_index < self.tp_idx:
            gm_min_where = "forward"
        elif gm_min_index == self.tp_idx:
            gm_min_where = "turning_point"
        else:
            gm_min_where = "reverse"
            
        return Point(
            raw=float(self.gm.raw.min()),
            where=gm_min_where,
            forward=float(self.gm.forward.min()) if len(self.gm.forward) > 0 else 0.0,
            reverse=float(self.gm.reverse.min()) if len(self.gm.reverse) > 0 else 0.0,
        )

    def _compute_absI_max(self) -> Point:
        """
        计算电流绝对值的最大点 absI_max
        :return: Point 包含 raw / forward / reverse 的 absI_max
        """
        if len(self.I.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        absI_max_index = np.abs(self.I.raw).argmax()

        if absI_max_index < self.tp_idx:
            absI_max_where = "forward"
        elif absI_max_index == self.tp_idx:
            absI_max_where = "turning_point"
        else:
            absI_max_where = "reverse"
            
        return Point(
            raw=float(np.abs(self.I.raw).max()),
            where=absI_max_where,
            forward=float(np.abs(self.I.forward).max()) if len(self.I.forward) > 0 else 0.0,
            reverse=float(np.abs(self.I.reverse).max()) if len(self.I.reverse) > 0 else 0.0,
        )
    
    def _compute_I_max(self) -> Point:
        """
        计算最大电流点
        :return: Point 包含 raw / forward / reverse 的 I_max
        """
        if len(self.I.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        I_max_index = self.I.raw.argmax()

        if I_max_index < self.tp_idx:
            I_max_where = "forward"
        elif I_max_index == self.tp_idx:
            I_max_where = "turning_point"
        else:
            I_max_where = "reverse"
            
        return Point(
            raw=float(self.I.raw.max()),
            where=I_max_where,
            forward=float(self.I.forward.max()) if len(self.I.forward) > 0 else 0.0,
            reverse=float(self.I.reverse.max()) if len(self.I.reverse) > 0 else 0.0,
        )

    def _compute_absI_min(self) -> Point:
        """
        计算电流绝对值的最小点 absI_min
        :return: Point 包含 raw / forward / reverse 的 absI_min
        """
        if len(self.I.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        absI_min_index = np.abs(self.I.raw).argmin()

        if absI_min_index < self.tp_idx:
            absI_min_where = "forward"
        elif absI_min_index == self.tp_idx:
            absI_min_where = "turning_point"
        else:
            absI_min_where = "reverse"
            
        return Point(
            raw=float(np.abs(self.I.raw).min()),
            where=absI_min_where,
            forward=float(np.abs(self.I.forward).min()) if len(self.I.forward) > 0 else 0.0,
            reverse=float(np.abs(self.I.reverse).min()) if len(self.I.reverse) > 0 else 0.0,
        )
    
    def _compute_I_min(self) -> Point:
        """
        计算最小电流点
        :return: Point 包含 raw / forward / reverse 的 I_min
        """
        if len(self.I.raw) == 0:
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
            
        I_min_index = self.I.raw.argmin()

        if I_min_index < self.tp_idx:
            I_min_where = "forward"
        elif I_min_index == self.tp_idx:
            I_min_where = "turning_point"
        else:
            I_min_where = "reverse"
            
        return Point(
            raw=float(self.I.raw.min()),
            where=I_min_where,
            forward=float(self.I.forward.min()) if len(self.I.forward) > 0 else 0.0,
            reverse=float(self.I.reverse.min()) if len(self.I.reverse) > 0 else 0.0,
        )
    
    def _compute_Von(self, device_type: str = "N") -> Point:
        """
        计算Von (阈值电压)
        对于N型器件，使用对数斜率最大法
        对于P型器件，使用对数斜率最小法

        :param device_type: 器件类型，"N"表示N型，"P"表示P型
        :return: Point 包含 raw / forward / reverse 的 Von 值
        """
        try:
            if len(self.I.raw) == 0 or len(self.Vg.raw) == 0:
                return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)
                
            # 计算raw的Von
            log_Id_raw = np.log10(np.clip(np.abs(self.I.raw), 1e-12, None))
            dlogId_dVg_raw = self.safe_diff(log_Id_raw, self.Vg.raw)

            if len(dlogId_dVg_raw) == 0:
                return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)

            # 根据器件类型选择最大或最小斜率点
            if device_type.upper() == "N":
                idx_raw = dlogId_dVg_raw.argmax()  # N型选择最大斜率点
            else:  # 默认为P型
                idx_raw = dlogId_dVg_raw.argmin()  # P型选择最小斜率点

            Von_raw = self.Vg.raw[idx_raw] if idx_raw < len(self.Vg.raw) else 0.0

            # 计算forward的Von
            Von_forward = 0.0
            if len(self.I.forward) > 0 and len(self.Vg.forward) > 0:
                log_Id_forward = np.log10(np.clip(np.abs(self.I.forward), 1e-12, None))
                dlogId_dVg_forward = self.safe_diff(log_Id_forward, self.Vg.forward)

                if len(dlogId_dVg_forward) > 0:
                    if device_type.upper() == "N":
                        idx_forward = dlogId_dVg_forward.argmax()
                    else:
                        idx_forward = dlogId_dVg_forward.argmin()

                    Von_forward = float(self.Vg.forward[idx_forward]) if idx_forward < len(self.Vg.forward) else 0.0

            # 计算reverse的Von
            Von_reverse = 0.0
            if len(self.I.reverse) > 0 and len(self.Vg.reverse) > 0:
                log_Id_reverse = np.log10(np.clip(np.abs(self.I.reverse), 1e-12, None))
                dlogId_dVg_reverse = self.safe_diff(log_Id_reverse, self.Vg.reverse)

                if len(dlogId_dVg_reverse) > 0:
                    if device_type.upper() == "N":
                        idx_reverse = dlogId_dVg_reverse.argmax()
                    else:
                        idx_reverse = dlogId_dVg_reverse.argmin()

                    Von_reverse = float(self.Vg.reverse[idx_reverse]) if idx_reverse < len(self.Vg.reverse) else 0.0

            # 确定Von在哪个序列中
            if idx_raw < self.tp_idx:
                Von_where = "forward"
            elif idx_raw == self.tp_idx:
                Von_where = "turning_point"
            else:
                Von_where = "reverse"

            return Point(
                raw=float(Von_raw),
                where=Von_where,
                forward=Von_forward,
                reverse=Von_reverse,
            )
        except Exception as e:
            print(f"警告：Von计算失败: {e}")
            return Point(raw=0.0, where="unknown", forward=0.0, reverse=0.0)

    @staticmethod
    def safe_diff(
        f: NDArray[np.float64], x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        计算稳定差分导数：前向 + 后向 + 中心差分组合，避免除以0或nan，转折点处做前向差分和后向差分的平均值
        支持任意长度数组
        """
        f = np.asarray(f, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n = len(f)

        if n < 2:
            return np.zeros_like(f)

        df = np.zeros_like(f, dtype=np.float64)

        for i in range(n):
            if i == 0:
                if n > 1:
                    dx = x[1] - x[0]
                    dx = dx if abs(dx) > 1e-12 else 1e-12
                    df[i] = (f[1] - f[0]) / dx
            elif i == n - 1:
                dx = x[-1] - x[-2]
                dx = dx if abs(dx) > 1e-12 else 1e-12
                df[i] = (f[-1] - f[-2]) / dx
            else:
                dx1 = x[i] - x[i - 1]
                dx2 = x[i + 1] - x[i]
                dx1 = dx1 if abs(dx1) > 1e-12 else 1e-12
                dx2 = dx2 if abs(dx2) > 1e-12 else 1e-12
                df1 = (f[i] - f[i - 1]) / dx1
                df2 = (f[i + 1] - f[i]) / dx2
                df[i] = (df1 + df2) / 2

        return df