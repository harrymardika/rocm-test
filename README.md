# ROCm Deep Learning Framework Benchmark Comparison

A comprehensive benchmarking suite comparing the performance of **PyTorch**, **TensorFlow**, and **JAX** on AMD GPUs using ROCm. This project evaluates the computational performance of these frameworks across different matrix operation sizes and data types.

## 📋 Table of Contents

- [Overview](#overview)
- [Hardware & Software Setup](#hardware--software-setup)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Results & Key Findings](#results--key-findings)
- [Reproduction](#reproduction)
- [Notes & Future Work](#notes--future-work)

## 🎯 Overview

This project evaluates the performance of three major deep learning frameworks on AMD GPU hardware:

- **PyTorch**: Popular deep learning framework with ROCm support
- **TensorFlow**: Google's machine learning framework
- **JAX**: NumPy-like API for high-performance numerical computing

### Key Metrics

The benchmarks measure:

- **Execution Time**: Mean, median, min, max, std deviation
- **Percentiles**: 95th and 99th percentile latencies
- **Throughput**: Theoretical floating-point operations per second (TFLOPS)
- **Consistency**: Standard deviation and variance across runs

### Operations Benchmarked

- **Matrix Multiplication (MatMul)**: Core linear algebra operation tested across different matrix sizes
  - Sizes: 256×256, 512×512, 1024×1024, 2048×2048
  - Data Types: Float32 precision

## 🖥️ Hardware & Software Setup

### Hardware

- **GPU**: AMD Radeon RX 6800S
- **Device Memory**: VRAM as reported by system specs
- **Compute Units**: RDNA-based GPU architecture

### Software Stack

- **ROCm**: AMD's open-source compute platform
- **Framework Versions**:
  - PyTorch: 2.10.0+rocm7.1
  - TensorFlow: TensorFlow with ROCm backend
  - JAX: Latest ROCm-compatible version
  - Flax: Neural network library for JAX
  - Optax: JAX-based optimizers
- **Python**: 3.x
- **Dependencies**: NumPy, Matplotlib for visualization

## 📁 Project Structure

```
rocm-test/
├── README.md                          # This file
├── pytorch.ipynb                      # PyTorch benchmark notebook
├── tensorflow.ipynb                   # TensorFlow benchmark notebook
├── jax.ipynb                          # JAX benchmark notebook
├── comparison.ipynb                   # Comparative analysis and visualization
│
├── result/                            # Benchmark results (JSON format)
│   ├── benchmark_pytorch_results.json
│   ├── benchmark_tensorflow_results.json
│   └── benchmark_jax_results.json
│
└── archieve/                          # Historical benchmarks and earlier versions
    ├── benchmark.ipynb
    ├── benchmark-pytorch.ipynb
    ├── benchmark-tensorflow.ipynb
    ├── comparison.ipynb
    ├── test.ipynb
    ├── torch.ipynb
    ├── tensorflow.ipynb
    ├── yolov8n.pt                    # Model weights (legacy)
    └── images/                        # Previous benchmark reports
        ├── benchmark_comparison_metrics.csv
        ├── BENCHMARK_COMPARISON_REPORT.txt
        └── benchmark_summary.csv
```

### File Descriptions

| File               | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `pytorch.ipynb`    | Benchmarks PyTorch operations on ROCm GPU                  |
| `tensorflow.ipynb` | Benchmarks TensorFlow operations on ROCm GPU               |
| `jax.ipynb`        | Benchmarks JAX operations on ROCm GPU                      |
| `comparison.ipynb` | Loads all results and generates comparative visualizations |
| `result/`          | Contains JSON benchmark results for analysis               |
| `archieve/`        | Previous versions and experimental notebooks               |

## 📦 Requirements

### System Requirements

- Linux OS with ROCm support
- AMD GPU compatible with ROCm (tested on RX 6800S)
- CUDA/ROCm libraries installed
- ~4GB+ GPU memory (for larger matrices)

### Python Packages

```
torch>=2.10.0       # PyTorch with ROCm
tensorflow>=2.13    # TensorFlow with ROCm
jax>=0.4.0         # JAX
jaxlib>=0.4.0      # JAX library
flax>=0.7.0        # Flax (JAX neural networks)
optax>=0.1.4       # JAX optimizers
numpy>=1.20        # Numerical computing
matplotlib>=3.5    # Plotting and visualization
```

## 🚀 Installation

### Step 1: ROCm Setup

Ensure ROCm is installed on your system:

```bash
# Check ROCm installation
rocm-smi

# If not installed, follow AMD ROCm installation:
# https://rocmdocs.amd.com/en/docs-5.6.0/deploy/linux/index.html
```

### Step 2: Clone Repository

```bash
cd /path/to/rocm-test
```

### Step 3: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate
```

### Step 4: Install Python Dependencies

**For PyTorch + ROCm:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**For TensorFlow + ROCm:**

```bash
pip install tensorflow[and-cuda]>=2.13
# Or with ROCm specifically:
pip install tensorflow-rocm
```

**For JAX + ROCm:**

```bash
pip install jax[cuda11_cudnn82]=0.4.4
# Or for ROCm:
pip install jax-rocm
```

**For Other Dependencies:**

```bash
pip install numpy matplotlib flax optax
```

### Step 5: Launch Jupyter

```bash
jupyter notebook
```

## 🔧 Usage

### Running Individual Framework Benchmarks

1. **PyTorch Benchmark**
   - Open `pytorch.ipynb` in Jupyter
   - Run all cells sequentially
   - Results saved to `result/benchmark_pytorch_results.json`

2. **TensorFlow Benchmark**
   - Open `tensorflow.ipynb` in Jupyter
   - Run all cells sequentially
   - Results saved to `result/benchmark_tensorflow_results.json`

3. **JAX Benchmark**
   - Open `jax.ipynb` in Jupyter
   - Run all cells sequentially
   - Results saved to `result/benchmark_jax_results.json`

### Running Comparison Analysis

1. Open `comparison.ipynb`
2. Ensure all three result JSON files are present in the `result/` directory
3. Run all cells to:
   - Load all benchmark results
   - Generate comparative visualizations
   - Compute statistical comparisons
   - Display performance metrics

### Expected Outputs

Each benchmark notebook generates:

- **Console output**: Framework version, device info, ROCm status
- **JSON results**: Detailed timing statistics in `result/` folder
- **Comparison notebook output**: Charts and tables comparing all frameworks

## 📊 Benchmarks

### Matrix Multiplication (MatMul)

The primary benchmark measures matrix multiplication operations, a fundamental operation in deep learning:

```
Operation: C = A @ B
Where A and B are square matrices of size N×N
```

### Tested Configurations

| Size      | PyTorch | TensorFlow | JAX |
| --------- | ------- | ---------- | --- |
| 256×256   | ✓       | ✓          | ✓   |
| 512×512   | ✓       | ✓          | ✓   |
| 1024×1024 | ✓       | ✓          | ✓   |
| 2048×2048 | ✓       | ✓          | ✓   |

### Metrics Captured

For each operation, the following statistics are recorded:

- **mean_ms**: Average execution time (milliseconds)
- **std_ms**: Standard deviation of execution time
- **min_ms**: Minimum execution time
- **max_ms**: Maximum execution time
- **median_ms**: Median execution time
- **p95_ms**: 95th percentile latency
- **p99_ms**: 99th percentile latency
- **num_runs**: Number of benchmark iterations (100-200)
- **tflops**: Floating-point operations per second (for applicable operations)

### Warmup and Synchronization

Each benchmark includes:

- **Warmup runs**: 20 iterations to stabilize GPU state
- **GPU synchronization**: Proper synchronization to measure actual GPU execution time
- **Multiple iterations**: 100-200 runs per operation for statistical significance

## 📈 Results & Key Findings

### Sample Results: PyTorch on RX 6800S

```json
{
  "framework": "PyTorch",
  "version": "2.10.0+rocm7.1",
  "device": "AMD Radeon RX 6800S",
  "benchmarks": {
    "matmul": {
      "float32": {
        "256": {
          "mean_ms": 0.0402,
          "std_ms": 0.0113,
          "tflops": 0.835
        },
        "512": {
          "mean_ms": 0.0921,
          "std_ms": 0.0122,
          "tflops": 2.914
        },
        "1024": {
          "mean_ms": 0.2729,
          "std_ms": 0.0039,
          "tflops": 7.868
        },
        "2048": {
          "mean_ms": 2.3523,
          "std_ms": 0.0606,
          "tflops": 7.303
        }
      }
    }
  }
}
```

### Key Observations

1. **Framework Performance Variations**: Each framework shows different optimization characteristics with ROCm

2. **Matrix Size Effects**:
   - Smaller matrices (256×256) show high variance
   - Larger matrices (2048×2048) show better GPU utilization

3. **Throughput Scaling**:
   - TFLOPS increase with matrix size up to a point
   - Saturation may occur at larger sizes due to memory bandwidth

4. **Consistency**:
   - Standard deviation decreases with larger operations
   - Indicates better GPU scheduling stability for larger workloads

### Visualization

The `comparison.ipynb` notebook generates:

- Performance time comparisons across frameworks
- TFLOPS comparison charts
- Variance and stability analysis
- Framework-specific optimization patterns

## 🔄 Reproduction

### Step-by-Step Reproduction

1. **Verify ROCm Installation**

   ```bash
   rocm-smi --showid --showtemp --showuse
   ```

2. **Update Notebooks** (if needed)
   - Update ROCm, driver, and framework versions
   - Adjust matrix sizes or data types as needed

3. **Run Benchmarks in Order**

   ```
   1. pytorch.ipynb → benchmark_pytorch_results.json
   2. tensorflow.ipynb → benchmark_tensorflow_results.json
   3. jax.ipynb → benchmark_jax_results.json
   4. comparison.ipynb → Visualizations and analysis
   ```

4. **Verify Results**
   - Check JSON files exist in `result/` directory
   - Examine console output for errors
   - Review generated comparison plots

### Customization

To modify benchmarks:

- **Matrix sizes**: Edit size arrays in benchmark notebooks
- **Number of runs**: Change `num_runs` parameter in `benchmark_fn()`
- **Warmup iterations**: Adjust `warmup_runs` parameter
- **Data types**: Add float16, bfloat16 operations
- **Other operations**: Extend with convolution, FFT, other BLAS operations

## 📝 Notes & Future Work

### Current Limitations

1. **Single GPU**: Benchmarks current single GPU only
2. **Limited Operations**: Currently only matrix multiplication tested
3. **Static Batch Sizes**: No variable batch size testing
4. **No Mixed Precision**: Float32 only in current version
5. **Simple Models**: Not testing full neural network training

### Potential Improvements

1. **Extended Operations**:
   - Convolution operations
   - Fast Fourier Transform (FFT)
   - Reduction operations
   - Activation functions

2. **Data Types**:
   - Float16 (half precision)
   - bFloat16 (TensorFlow brain float)
   - Integer operations
   - Mixed precision testing

3. **Real Workloads**:
   - Neural network training (ResNet, BERT)
   - Transformer inference
   - Computer vision models
   - NLP models

4. **Multi-GPU Support**:
   - Distributed benchmarks
   - Communication overhead
   - Scaling efficiency

5. **Advanced Analysis**:
   - Memory bandwidth analysis
   - Kernel launch overhead
   - Python interpreter overhead
   - Compilation time measurement

### Troubleshooting

**Issue**: ROCm not detected

```bash
export HSA_OVERRIDE_GFX_VERSION=906  # For specific GPU versions if needed
rocm-smi
```

**Issue**: Out of memory errors

```python
# Reduce matrix size or batch size
# Clear GPU cache: torch.cuda.empty_cache()
```

**Issue**: Framework-specific errors

- Check framework version compatibility with ROCm version
- Verify correct installation with import tests
- Review framework documentation for ROCm-specific issues

## 📚 References

- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
- [TensorFlow ROCm Support](https://www.tensorflow.org/install/gpu)
- [JAX Installation Guide](https://github.com/google/jax)
- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [Flax Documentation](https://flax.readthedocs.io/)

## 📄 License

This project is provided as-is for benchmarking and research purposes.

## 🤝 Contributing

For improvements or bug reports:

1. Create an issue describing the problem
2. Test on your hardware configuration
3. Submit results with:
   - Hardware details
   - ROCm and framework versions
   - Benchmark results
   - Observations and findings

---

**Last Updated**: March 2026
**Hardware**: AMD Radeon RX 6800S
**Primary Framework Versions**: PyTorch 2.10.0+rocm7.1, TensorFlow 2.x, JAX 0.4.x+
