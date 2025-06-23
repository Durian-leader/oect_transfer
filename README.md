# OECT Transfer Curve Analysis

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/your-repo/oect-transfer)

[简体中文](https://github.com/Durian-leader/oect_transfer/blob/main/README_CN.md)

A Python package for analyzing Organic Electrochemical Transistor (OECT) transfer characteristic curves. This library provides comprehensive tools for extracting key device parameters including transconductance, threshold voltage, and device performance metrics.

## 🚀 Features

- **Transconductance Analysis**: Calculate transconductance (gm) using robust numerical differentiation
- **Threshold Voltage Extraction**: Automatic Von calculation using logarithmic slope method
- **Device Type Support**: Support for both N-type and P-type devices
- **Robust Data Processing**: Built-in validation and error handling for noisy data
- **Forward/Reverse Analysis**: Separate analysis of forward and reverse sweep directions
- **Performance Metrics**: Extract maximum/minimum current points and related parameters

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- Standard library dependencies

### Install from source

```bash
git clone https://github.com/Durian-leader/oect_transfer/oect-transfer.git
cd oect-transfer
pip install -e .
```

### Dependencies

```bash
pip install numpy
```

## 🔧 Quick Start

### Basic Usage

```python
import numpy as np
from oect_transfer import Transfer

# Example data: Gate voltage (Vg) and drain current (Id)
vg = np.linspace(-0.5, 0.5, 100)  # Gate voltage in V
id = np.exp(vg * 10) * 1e-6       # Drain current in A (example)

# Create Transfer object
transfer = Transfer(vg, id, device_type="N")

# Access computed parameters
print(f"Maximum transconductance: {transfer.gm_max.raw:.2e} S")
print(f"Threshold voltage: {transfer.Von.raw:.3f} V")
print(f"Maximum current: {transfer.I_max.raw:.2e} A")
```

### Advanced Analysis

```python
# Analyze forward and reverse sweeps separately
print(f"Forward gm_max: {transfer.gm_max.forward:.2e} S")
print(f"Reverse gm_max: {transfer.gm_max.reverse:.2e} S")

# Check where maximum transconductance occurs
print(f"gm_max location: {transfer.gm_max.where}")  # 'forward', 'reverse', or 'turning_point'

# Access raw data sequences
print(f"Gate voltage range: {transfer.Vg.raw.min():.2f} to {transfer.Vg.raw.max():.2f} V")
print(f"Current range: {transfer.I.raw.min():.2e} to {transfer.I.raw.max():.2e} A")
```

### P-type Device Analysis

```python
# For P-type devices, specify device_type="P"
transfer_p = Transfer(vg, id, device_type="P")
print(f"P-type Von: {transfer_p.Von.raw:.3f} V")
```

## 📚 API Reference

### Classes

#### `Transfer`

Main class for transfer curve analysis.

**Constructor:**
```python
Transfer(x, y, device_type="N")
```

**Parameters:**
- `x` (array-like): Gate voltage data (Vg)
- `y` (array-like): Drain current data (Id)
- `device_type` (str): Device type, "N" for N-type, "P" for P-type

**Attributes:**
- `Vg` (Sequence): Gate voltage data (raw, forward, reverse)
- `I` (Sequence): Drain current data (raw, forward, reverse)
- `gm` (Sequence): Transconductance data
- `gm_max` (Point): Maximum transconductance point
- `I_max` (Point): Maximum current point
- `I_min` (Point): Minimum current point
- `Von` (Point): Threshold voltage point

#### `Sequence`

Data container for raw, forward, and reverse sweep data.

**Attributes:**
- `raw` (NDArray): Complete dataset
- `forward` (NDArray): Forward sweep (up to maximum Vg)
- `reverse` (NDArray): Reverse sweep (from maximum Vg)

#### `Point`

Container for parameter values at specific points.

**Attributes:**
- `raw` (float): Value from complete dataset
- `where` (str): Location ("forward", "reverse", or "turning_point")
- `forward` (float): Value from forward sweep
- `reverse` (float): Value from reverse sweep

### Methods

#### `safe_diff(f, x)`

Static method for robust numerical differentiation.

**Parameters:**
- `f` (array): Function values
- `x` (array): Independent variable values

**Returns:**
- `NDArray`: Computed derivatives

## 🧪 Examples

### Example 1: Analyzing Experimental Data

```python
import numpy as np
import matplotlib.pyplot as plt
from oect_transfer import Transfer

# Load your experimental data
vg_data = np.loadtxt('gate_voltage.txt')
id_data = np.loadtxt('drain_current.txt')

# Create transfer object
transfer = Transfer(vg_data, id_data, device_type="N")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Transfer curve
ax1.semilogy(transfer.Vg.raw, np.abs(transfer.I.raw))
ax1.axvline(transfer.Von.raw, color='red', linestyle='--', 
           label=f'Von = {transfer.Von.raw:.3f} V')
ax1.set_xlabel('Gate Voltage (V)')
ax1.set_ylabel('|Drain Current| (A)')
ax1.legend()
ax1.grid(True)

# Transconductance
ax2.plot(transfer.Vg.raw[:-1], transfer.gm.raw)
ax2.axhline(transfer.gm_max.raw, color='red', linestyle='--',
           label=f'gm_max = {transfer.gm_max.raw:.2e} S')
ax2.set_xlabel('Gate Voltage (V)')
ax2.set_ylabel('Transconductance (S)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Example 2: Comparing Forward and Reverse Sweeps

```python
# Analyze hysteresis
forward_von = transfer.Von.forward
reverse_von = transfer.Von.reverse
hysteresis = abs(forward_von - reverse_von)

print(f"Forward Von: {forward_von:.3f} V")
print(f"Reverse Von: {reverse_von:.3f} V")
print(f"Hysteresis: {hysteresis:.3f} V")

# Plot forward vs reverse
plt.figure(figsize=(8, 6))
plt.semilogy(transfer.Vg.forward, np.abs(transfer.I.forward), 
             'b-', label='Forward')
plt.semilogy(transfer.Vg.reverse, np.abs(transfer.I.reverse), 
             'r--', label='Reverse')
plt.axvline(forward_von, color='blue', alpha=0.7, linestyle=':')
plt.axvline(reverse_von, color='red', alpha=0.7, linestyle=':')
plt.xlabel('Gate Voltage (V)')
plt.ylabel('|Drain Current| (A)')
plt.legend()
plt.grid(True)
plt.title('Transfer Curve: Forward vs Reverse')
plt.show()
```

## ⚠️ Important Notes

### Data Requirements

- **Input arrays must be 1D** and of equal length
- **No NaN or infinite values** are allowed
- **Minimum 2 data points** required for meaningful analysis
- Data should span a reasonable voltage range including the device turn-on region

### Device Type Selection

- **N-type devices**: Use `device_type="N"` (default)
  - Von calculated using maximum logarithmic slope
  - Suitable for enhancement-mode n-channel devices
  
- **P-type devices**: Use `device_type="P"`
  - Von calculated using minimum logarithmic slope
  - Suitable for enhancement-mode p-channel devices

### Transconductance Calculation

The transconductance is calculated using a robust numerical differentiation method that:
- Uses forward, backward, and central difference schemes appropriately
- Handles turning points with averaged derivatives
- Includes safeguards against division by zero

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/Durian-leader/oect_transfer.git
cd oect-transfer
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **lidonghao** - *Initial work* - [lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)

## 🙏 Acknowledgments

- Thanks to the OECT research community for valuable feedback
- Inspired by standard practices in organic electronics characterization

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Durian-leader/oect_transfer/issues) page
2. Create a new issue with detailed description
3. Contact the maintainer at [lidonghao100@outlook.com](mailto:lidonghao100@outlook.com)

## 📈 Roadmap

- [ ] Add support for output characteristic analysis
- [ ] Implement mobility extraction methods
- [ ] Add data export functionality
- [ ] Develop GUI interface
- [ ] Add more device parameter extraction methods

---

**Keywords:** OECT, Organic Electrochemical Transistor, Transfer Curve, Transconductance, Threshold Voltage, Device Characterization, Python, Analysis