# ROCm Deep Learning Benchmark Suite

This repository contains a set of Jupyter notebooks that benchmark three deep learning frameworks on an AMD GPU with ROCm:

- PyTorch
- TensorFlow
- JAX

The suite measures low-level compute kernels and higher-level model workloads, then stores the results as JSON for comparison and visualization.

## Overview

The experiment is designed to answer a simple question: how do the major Python deep learning frameworks behave on the same AMD ROCm hardware when they run the same style of workloads?

The notebooks focus on:

- Matrix multiplication throughput and latency
- Batched matrix multiplication
- Element-wise operations and reductions
- Convolution workloads
- Common layer operations such as dense layers, batch normalization, layer normalization, attention, and softmax
- CNN training and inference benchmarks
- Transformer training and inference benchmarks
- Memory transfer and allocation behavior
- JAX-specific JIT compilation overhead

Each framework notebook records timing statistics, throughput metrics, and device information, then writes a result file into `result/`. The comparison notebook loads those result files and generates charts.

## Hardware and Software

### Hardware used in the recorded runs

- GPU: AMD Radeon RX 6800S
- OS: Linux
- ROCm device access verified with `rocm-smi`

### Framework versions captured in the current result files

These are the versions recorded in the JSON output under `result/`:

- PyTorch: `2.10.0+rocm7.1`
- TensorFlow: `2.20.0-dev0+selfbuilt`
- JAX: `0.7.1`

The notebooks also use supporting packages such as NumPy, Matplotlib, Flax, Optax, and Jupyter.

## Repository Layout

```text
rocm-test/
├── pytorch.ipynb
├── tensorflow.ipynb
├── jax.ipynb
├── comparison.ipynb
├── result/
    ├── benchmark_pytorch_results.json
    ├── benchmark_tensorflow_results.json
    └── benchmark_jax_results.json

```

### What each top-level file does

- `pytorch.ipynb`: runs the PyTorch benchmark suite and writes `result/benchmark_pytorch_results.json`
- `tensorflow.ipynb`: runs the TensorFlow benchmark suite and writes `result/benchmark_tensorflow_results.json`
- `jax.ipynb`: runs the JAX benchmark suite and writes `result/benchmark_jax_results.json`
- `comparison.ipynb`: loads all framework results and creates comparison plots

## Benchmark Coverage

The benchmark notebooks are structured into several sections.

### 1. Matrix operations

Benchmarks include:

- Square matrix multiplication at sizes 256, 512, 1024, 2048, 4096, and 8192
- Dtype coverage where supported by the framework and device
- PyTorch: float32, float16, and bfloat16 when available
- TensorFlow: float32, float16, and bfloat16 when available
- JAX: float32, float16, and bfloat16
- Batched matrix multiplication for several batch and shape combinations

### 2. Element-wise and reduction operations

The suite measures:

- Element-wise add
- Element-wise multiply
- exp
- sin
- sigmoid
- tanh
- relu
- gelu
- silu

Reduction benchmarks include:

- sum
- mean
- max
- min
- std
- norm
- argmax
- sort

Tensor sizes used in these tests:

- 1M elements
- 10M elements
- 50M elements
- 100M elements

### 3. Convolution workloads

The convolution benchmarks use representative CNN shapes such as:

- ResNet stem and ResNet blocks
- CIFAR-style convolution
- VGG-style convolution
- High-resolution single-image convolution

Each convolution benchmark records both forward and forward-plus-backward timing.

### 4. Common layer operations

These include:

- Dense or linear layers
- Batch normalization
- Layer normalization
- Multi-head attention
- Softmax

The layer configurations are based on transformer-style and vision-style shapes such as BERT-hidden, BERT-FFN, and LLM-like shapes.

### 5. CNN model training

The notebooks benchmark a ResNet-18-like model across batch sizes 16, 32, and 64.

Recorded metrics include:

- Inference latency
- Training step latency
- Images per second
- Peak GPU memory usage

TensorFlow and PyTorch also include a mixed-precision variant for batch size 32.

### 6. Transformer model training

The notebooks benchmark a transformer classifier across several configurations, including:

- Small transformer
- BERT-base-like
- BERT-long-seq
- BERT-large-like

Recorded metrics include:

- Inference latency
- Training step latency
- Samples per second
- Tokens per second
- Parameter count

### 7. Memory transfer and allocation

These benchmarks measure:

- Host-to-device transfer bandwidth
- Device-to-host transfer bandwidth
- GPU memory allocation latency

### 8. JAX-specific compilation and synchronization behavior

The JAX notebook additionally measures:

- First-call JIT compilation time
- Post-compilation execution time
- Compilation overhead

## Measurement Methodology

All notebooks follow the same basic benchmarking pattern:

1. Warm up the GPU with several small operations.
2. Run the target function multiple times.
3. Synchronize the device before measuring the stop time.
4. Compute summary statistics from the collected timings.

The result JSON files typically include:

- mean_ms
- std_ms
- min_ms
- max_ms
- median_ms
- p95_ms
- p99_ms
- num_runs

Additional metrics may be added depending on the workload:

- tflops
- bandwidth_gbs
- throughput_imgs_per_sec
- throughput_samples_per_sec
- tokens_per_sec
- peak_memory_mb
- param_count
- compilation_overhead_ms

## Result Files

The notebooks write their outputs to `result/`:

- `result/benchmark_pytorch_results.json`
- `result/benchmark_tensorflow_results.json`
- `result/benchmark_jax_results.json`

Each JSON file includes:

- `framework`
- `version`
- `device`
- `timestamp`
- `benchmarks`

The structure under `benchmarks` depends on the notebook and the cells that were executed.

## Comparison Outputs

The comparison notebook reads the result JSON files and can generate these figures when the corresponding benchmark sections are present:

- `comparison_matmul.png`
- `comparison_elementwise.png`
- `comparison_layers.png`
- `comparison_training.png`
- `comparison_memory.png`

It also prints a summary table in the notebook output.

## How to Run

### 1. Confirm ROCm is working

```bash
rocm-smi --showid --showtemp --showuse
```

If `rocm-smi` fails, fix the ROCm driver and device setup first.

### 2. Open the notebooks

Launch Jupyter and open the benchmark notebooks in the repository root.

### 3. Run the framework notebooks

Run the notebooks in this order:

1. `pytorch.ipynb`
2. `tensorflow.ipynb`
3. `jax.ipynb`

Each notebook saves its JSON output into `result/`.

### 4. Run the comparison notebook

After the three JSON files exist, open and run `comparison.ipynb` to generate the comparative charts.

## Environment Setup

The repository does not include a lockfile, so the exact install steps depend on your ROCm stack and the package versions available for your platform.

At minimum, you need:

- Python 3
- Jupyter
- NumPy
- Matplotlib
- PyTorch with ROCm support
- TensorFlow with ROCm support
- JAX with ROCm support
- Flax and Optax for the JAX model benchmarks

If you are recreating the experiment on the same hardware, prefer matching the framework versions recorded in the JSON results.

## Interpreting the Numbers

Use caution when comparing frameworks directly:

- JIT-based frameworks may pay compilation cost on the first run
- Different frameworks have different synchronization behavior
- Memory layout and kernel fusion can change results significantly
- Small benchmarks are often dominated by launch overhead and timing noise
- Larger benchmarks usually give more stable throughput measurements

For the most reliable comparison, compare the same benchmark category, the same shape, and the same dtype.

## Reproducing the Experiment

To reproduce the benchmark set as closely as possible:

1. Use the same Linux + AMD ROCm environment.
2. Verify the GPU is detected by the framework.
3. Run each notebook from a fresh kernel.
4. Keep the warmup and run counts unchanged.
5. Save the generated JSON files in `result/`.
6. Run `comparison.ipynb` after all result files are present.

If you change the GPU, ROCm version, or framework version, record that change alongside the results.

## Troubleshooting

### ROCm is not detected

- Confirm the correct AMD driver and ROCm packages are installed
- Re-run `rocm-smi`
- Check that the framework build is ROCm-enabled

### A benchmark is skipped

Some cells intentionally skip a configuration if the framework or GPU cannot support it, such as unsupported dtypes or out-of-memory cases.

### Out of memory errors

- Reduce matrix size, batch size, or sequence length
- Close other GPU workloads
- Restart the notebook kernel before rerunning

### Comparison notebook shows missing plots

The comparison notebook only draws sections that exist in the JSON files. If a benchmark category was not executed, the corresponding plot may be partially empty or omitted.

## Current Snapshot

The current result files show the following recorded framework/device combinations:

- PyTorch on AMD Radeon RX 6800S
- TensorFlow on GPU device 0
- JAX on `rocm:0`

These values come from the JSON files under `result/` and may change if you rerun the notebooks in a different environment.

## License

No explicit license file is present in this repository. Add one if you plan to share or publish the experiment.
