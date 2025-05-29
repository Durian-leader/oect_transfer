# transfer.py
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from typing import List

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
    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64], device_type='N'):
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
        


        max_idx = x.argmax()

        self.Vg = Sequence(
            raw=x,
            forward=x[:max_idx + 1],
            reverse=x[max_idx:]
        )
        self.I = Sequence(
            raw=y,
            forward=y[:max_idx + 1],
            reverse=y[max_idx:]
        )

        self.gm = self._compute_gm()
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
            reverse=self.safe_diff(self.I.reverse, self.Vg.reverse)
        )
    
    def _compute_gm_max(self) -> Point:
        """
        计算最大跨导点
        :return: Point 包含 raw / forward / reverse 的 gm_max
        """
        gm_max_index = self.gm.raw.argmax()
        index = self.Vg.raw.argmax()

        if gm_max_index < index:
            gm_max_where = "forward"
        elif gm_max_index == index:
            gm_max_where = "turning_point"
        else:
            gm_max_where = "reverse"
        return Point(
            raw=self.gm.raw.max(),
            where=gm_max_where,
            forward=self.gm.forward.max(),
            reverse=self.gm.reverse.max()
        )
    
    def _compute_I_max(self) -> Point:
        """
        计算最大跨导点 Id_max
        :return: Point 包含 raw / forward / reverse 的 Id_max
        """
        I_max_index = abs(self.I.raw).argmax()
        index = self.Vg.raw.argmax()

        if I_max_index < index:
            I_max_where = "forward"
        elif I_max_index == index:
            I_max_where = "turning_point"
        else:
            I_max_where = "reverse"
        return Point(
            raw=abs(self.I.raw).max(),
            where=I_max_where,
            forward=abs(self.I.forward).max(),
            reverse=abs(self.I.reverse).max()
        )
    
    def _compute_I_min(self) -> Point:
        """
        计算最小跨导点 Id_min
        :return: Point 包含 raw / forward / reverse 的 Id_min
        """
        I_min_index = abs(self.I.raw).argmin()
        index = self.Vg.raw.argmax()

        if I_min_index < index:
            I_min_where = "forward"
        elif I_min_index == index:
            I_min_where = "turning_point"
        else:
            I_min_where = "reverse"
        return Point(
            raw=abs(self.I.raw).min(),
            where=I_min_where,
            forward=abs(self.I.forward).min(),
            reverse=abs(self.I.reverse).min()
        )
        
    def _compute_Von(self, device_type="N") -> Point:
        """
        计算Von (阈值电压)
        对于N型器件，使用对数斜率最大法
        对于P型器件，使用对数斜率最小法
        
        :param device_type: 器件类型，"N"表示N型，"P"表示P型
        :return: Point 包含 raw / forward / reverse 的 Von 值
        """
        # 计算raw的Von
        log_Id_raw = np.log10(np.clip(abs(self.I.raw), 1e-12, None))
        dlogId_dVg_raw = self.safe_diff(log_Id_raw, self.Vg.raw)
        
        # 根据器件类型选择最大或最小斜率点
        if device_type.upper() == "N":
            idx_raw = dlogId_dVg_raw.argmax()  # N型选择最大斜率点
        else:  # 默认为P型
            idx_raw = dlogId_dVg_raw.argmin()  # P型选择最小斜率点
        
        Von_raw = self.Vg.raw[idx_raw]
        
        # 计算forward的Von
        log_Id_forward = np.log10(np.clip(abs(self.I.forward), 1e-12, None))
        dlogId_dVg_forward = self.safe_diff(log_Id_forward, self.Vg.forward)
        
        if device_type.upper() == "N":
            idx_forward = dlogId_dVg_forward.argmax()
        else:
            idx_forward = dlogId_dVg_forward.argmin()
        
        Von_forward = self.Vg.forward[idx_forward] if len(self.Vg.forward) > 0 else None
        
        # 计算reverse的Von
        log_Id_reverse = np.log10(np.clip(abs(self.I.reverse), 1e-12, None))
        dlogId_dVg_reverse = self.safe_diff(log_Id_reverse, self.Vg.reverse)
        
        if device_type.upper() == "N":
            idx_reverse = dlogId_dVg_reverse.argmax()
        else:
            idx_reverse = dlogId_dVg_reverse.argmin()
        
        Von_reverse = self.Vg.reverse[idx_reverse] if len(self.Vg.reverse) > 0 else None
        
        # 确定Von在哪个序列中
        index = self.Vg.raw.argmax()
        if idx_raw < index:
            Von_where = "forward"
        elif idx_raw == index:
            Von_where = "turning_point"
        else:
            Von_where = "reverse"
        
        return Point(
            raw=Von_raw,
            where=Von_where,
            forward=Von_forward,
            reverse=Von_reverse
        )

    @staticmethod
    def safe_diff(f: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
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



if __name__ == "__main__":
    x = np.array([0, 1, 2,3,2,1,0], dtype=np.float64)
    y = np.array([0, 1, 4,6,8,23,6], dtype=np.float64)
    transfer = Transfer(x, y)

    print("gm.raw:", transfer.gm.raw)
    print("gm.forward:", transfer.gm.forward)
    print("gm.reverse:", transfer.gm.reverse)
    print("gm_max.raw:", transfer.gm_max.raw)
    print("gm_max.where:", transfer.gm_max.where)
    print("gm_max.forward:", transfer.gm_max.forward)
    print("gm_max.reverse:", transfer.gm_max.reverse)
    print("I_max.raw:", transfer.I_max.raw)
    print("I_max.where:", transfer.I_max.where)
    print("I_max.forward:", transfer.I_max.forward)
    print("I_max.reverse:", transfer.I_max.reverse)
    print("I_min.raw:", transfer.I_min.raw)
    print("I_min.where:", transfer.I_min.where)
    print("I_min.forward:", transfer.I_min.forward)
    print("I_min.reverse:", transfer.I_min.reverse)
    print("Vg.raw:", transfer.Vg.raw)
    print("Vg.forward:", transfer.Vg.forward)
    print("Vg.reverse:", transfer.Vg.reverse)
    print("I.raw:", transfer.I.raw)
    print("I.forward:", transfer.I.forward)
    print("I.reverse:", transfer.I.reverse)
    print("Von (log-slope max Vg):", transfer.Von)