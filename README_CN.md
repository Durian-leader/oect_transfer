以下是你提供的英文文档的中文翻译版本：

---

# OECT 转移曲线分析

[![Python 版本](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![版本](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/your-repo/oect-transfer)

一个用于分析有机电化学晶体管（OECT）转移特性曲线的 Python 包。该库提供全面的工具，用于提取关键器件参数，包括跨导、阈值电压以及器件性能指标。

## 🚀 功能特色

* **跨导分析**：使用稳健的数值微分方法计算跨导（gm）
* **阈值电压提取**：通过对数斜率法自动计算 Von
* **器件类型支持**：支持 N 型和 P 型器件
* **强健的数据处理**：内置噪声数据的验证与错误处理机制
* **正/反向扫描分析**：分别分析正向与反向扫描方向
* **性能指标提取**：提取最大/最小电流点及相关参数

## 📦 安装指南

### 前置条件

* Python 3.7 或更高版本
* NumPy
* 标准库依赖项

### 从源代码安装

```bash
git clone https://github.com/your-repo/oect-transfer.git
cd oect-transfer
pip install -e .
```

### 依赖项

```bash
pip install numpy
```

## 🔧 快速开始

### 基本用法

```python
import numpy as np
from oect_transfer import Transfer

# 示例数据：栅极电压 (Vg) 与漏电流 (Id)
vg = np.linspace(-0.5, 0.5, 100)  # 单位：伏
id = np.exp(vg * 10) * 1e-6       # 单位：安培（示例）

# 创建 Transfer 对象
transfer = Transfer(vg, id, device_type="N")

# 获取计算参数
print(f"最大跨导: {transfer.gm_max.raw:.2e} S")
print(f"阈值电压: {transfer.Von.raw:.3f} V")
print(f"最大电流: {transfer.I_max.raw:.2e} A")
```

### 高级分析

```python
# 分别分析正向和反向扫描
print(f"正向最大跨导: {transfer.gm_max.forward:.2e} S")
print(f"反向最大跨导: {transfer.gm_max.reverse:.2e} S")

# 查看最大跨导出现的位置
print(f"gm_max 出现位置: {transfer.gm_max.where}")  # 'forward'、'reverse' 或 'turning_point'

# 访问原始数据序列
print(f"栅压范围: {transfer.Vg.raw.min():.2f} 到 {transfer.Vg.raw.max():.2f} V")
print(f"电流范围: {transfer.I.raw.min():.2e} 到 {transfer.I.raw.max():.2e} A")
```

### P 型器件分析

```python
# 对于 P 型器件，指定 device_type="P"
transfer_p = Transfer(vg, id, device_type="P")
print(f"P 型器件 Von: {transfer_p.Von.raw:.3f} V")
```

## 📚 API 参考

### 类

#### `Transfer`

用于转移曲线分析的主类。

**构造函数：**

```python
Transfer(x, y, device_type="N")
```

**参数说明：**

* `x`：栅极电压数据（Vg）
* `y`：漏电流数据（Id）
* `device_type`：器件类型，"N" 表示 N 型，"P" 表示 P 型

**属性：**

* `Vg`：栅压序列（包含原始、正向、反向）
* `I`：电流序列（包含原始、正向、反向）
* `gm`：跨导序列
* `gm_max`：最大跨导点
* `I_max`：最大电流点
* `I_min`：最小电流点
* `Von`：阈值电压点

#### `Sequence`

用于保存原始、正向与反向扫描数据的容器。

**属性：**

* `raw`：完整数据
* `forward`：正向扫描数据（到 Vg 最大值）
* `reverse`：反向扫描数据（从 Vg 最大值开始）

#### `Point`

用于保存某个特征点的数据值。

**属性：**

* `raw`：完整数据集对应的值
* `where`：出现位置（"forward"、"reverse" 或 "turning\_point"）
* `forward`：正向扫描对应值
* `reverse`：反向扫描对应值

### 方法

#### `safe_diff(f, x)`

静态方法：进行稳健的数值微分。

**参数：**

* `f`：函数值数组
* `x`：自变量数组

**返回：**

* `NDArray`：微分结果

## 🧪 示例

### 示例 1：分析实验数据

```python
import numpy as np
import matplotlib.pyplot as plt
from oect_transfer import Transfer

# 加载实验数据
vg_data = np.loadtxt('gate_voltage.txt')
id_data = np.loadtxt('drain_current.txt')

# 创建 Transfer 对象
transfer = Transfer(vg_data, id_data, device_type="N")

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 转移曲线
ax1.semilogy(transfer.Vg.raw, np.abs(transfer.I.raw))
ax1.axvline(transfer.Von.raw, color='red', linestyle='--', 
           label=f'Von = {transfer.Von.raw:.3f} V')
ax1.set_xlabel('栅极电压 (V)')
ax1.set_ylabel('|漏电流| (A)')
ax1.legend()
ax1.grid(True)

# 跨导曲线
ax2.plot(transfer.Vg.raw[:-1], transfer.gm.raw)
ax2.axhline(transfer.gm_max.raw, color='red', linestyle='--',
           label=f'gm_max = {transfer.gm_max.raw:.2e} S')
ax2.set_xlabel('栅极电压 (V)')
ax2.set_ylabel('跨导 (S)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### 示例 2：正向与反向扫描比较

```python
# 分析迟滞
forward_von = transfer.Von.forward
reverse_von = transfer.Von.reverse
hysteresis = abs(forward_von - reverse_von)

print(f"正向 Von: {forward_von:.3f} V")
print(f"反向 Von: {reverse_von:.3f} V")
print(f"迟滞宽度: {hysteresis:.3f} V")

# 绘制正反向扫描对比图
plt.figure(figsize=(8, 6))
plt.semilogy(transfer.Vg.forward, np.abs(transfer.I.forward), 
             'b-', label='正向扫描')
plt.semilogy(transfer.Vg.reverse, np.abs(transfer.I.reverse), 
             'r--', label='反向扫描')
plt.axvline(forward_von, color='blue', alpha=0.7, linestyle=':')
plt.axvline(reverse_von, color='red', alpha=0.7, linestyle=':')
plt.xlabel('栅极电压 (V)')
plt.ylabel('|漏电流| (A)')
plt.legend()
plt.grid(True)
plt.title('转移曲线：正向 vs 反向')
plt.show()
```

## ⚠️ 注意事项

### 数据要求

* **输入数组必须是一维**，且长度一致
* **不允许包含 NaN 或无穷值**
* **至少需要两个数据点**以进行分析
* 电压范围应涵盖器件导通区域

### 器件类型选择

* **N 型器件**：使用 `device_type="N"`（默认）

  * Von 使用最大对数斜率计算
  * 适用于增强型 N 通道器件

* **P 型器件**：使用 `device_type="P"`

  * Von 使用最小对数斜率计算
  * 适用于增强型 P 通道器件

### 跨导计算说明

跨导通过稳健的数值微分计算，其特性包括：

* 结合前向、后向和中心差分法
* 对转折点使用平均导数
* 避免除以零的错误处理机制

## 🤝 贡献指南

欢迎贡献！请参考我们的 [贡献指南](CONTRIBUTING.md) 获取更多细节。

### 开发环境配置

```bash
git clone https://github.com/your-repo/oect-transfer.git
cd oect-transfer
pip install -e .[dev]
```

### 运行测试

```bash
pytest tests/
```

## 📄 许可证

本项目基于 MIT 许可证发布，详情见 [LICENSE](LICENSE) 文件。

## 👥 作者信息

* **lidonghao** - *最初开发者* - [lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)

## 🙏 鸣谢

* 感谢 OECT 研究社区的反馈
* 灵感来自有机电子器件标准表征方法

## 📞 技术支持

如有问题或建议：

1. 查看 [Issues 页面](https://github.com/your-repo/oect-transfer/issues)
2. 创建新 issue 并详细描述问题
3. 直接联系维护者：[lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)

## 📈 未来计划

* [ ] 支持输出特性分析
* [ ] 实现迁移率提取方法
* [ ] 添加数据导出功能
* [ ] 开发图形化界面（GUI）
* [ ] 丰富器件参数提取手段

---

**关键词：** OECT、有机电化学晶体管、转移曲线、跨导、阈值电压、器件表征、Python、数据分析
