# Foundation Model Stack

Foundation Model Stack is a collection of components for development, inference, training, and tuning of foundation models leveraging PyTorch native components. For inference optimizations we aim to support PyTorch compile, accelerated transformers, and tensor parallelism. At training time we aim to support FSDP, accelerated transformers, and PyTorch compile. To enable these optimizations, we will provide reimplementations of several popular model architectures starting with Llama and GPT-BigCode. 


# Project Description
This project focuses on modernizing the Tensor-Parallel (TP) implementation in IBM's Foundation Model Stack (FMS) using PyTorch DTensor APIs. The goal is to address scalability and memory bottlenecks, enhance compatibility with modern distributed computation frameworks, and lay the foundation for future scalability by integrating Sequence Parallelism.

The updated implementation supports large-scale language models like LLaMA and Granite, ensuring efficient resource utilization, compatibility with IBM tools, and better performance for training and inference.

### **Outline of the Code Repository**

Below is the outline of the repository, including the updated files and their purposes:

```
├── fms/
│   ├── distributed/
│   │   ├── strategy.py          # Updated to support Tensor Parallelism
│   │   └── ...                  # Other distributed strategy-related files
│   ├── modules/
│   │   ├── attention.py         # Updated to incorporate Tensor Parallel changes
│   │   ├── feedforward.py       # Updated to include Tensor Parallel logic
│   │   └── ...                  # Other module-specific files
│   └── ...                      # Remaining FMS framework files
├── utils/
│   ├── tp_plan.py               # Added for defining Tensor Parallel plans and configurations
│   └── ...                      # Utility scripts for benchmarking, logging, etc.
├── benchmarks/                  # Benchmarking scripts for performance evaluation
│   └── benchmark_inference.py   # Script for running inference benchmarks
└── README.md                    # Project documentation
```

### **Key Updates**
1. **fms/distributed/strategy.py**:
   - Enhanced to support Tensor Parallelism using PyTorch DTensor APIs.
2. **fms/modules/attention.py**:
   - Updated to implement Tensor Parallel changes for attention mechanisms.
3. **fms/modules/feedforward.py**:
   - Modified to support Tensor Parallel logic in feedforward layers.
4. **utils/tp_plan.py**:
   - Newly added to manage Tensor Parallel plans and configurations.
5. **scripts/benchmark_inference.py**:
   - Newly added profiler to track memory usage.

This structure highlights the key files modified or added for the Tensor Parallel modernization.

# Example commands to execute the code
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./scripts/benchmark_inference.py --architecture=llama --variant=2-7b --tokenizer=~/models/tokenizer.model --batch_size=2 --seq_len=512 --skip_correctness_check

# Results (including charts/tables) and your observations

1. Performance Insights
   - Inference Speeds: The modernized implementation has slightly higher communication overhead due to DTensor-based distributed execution but is efficient for larger workloads.
   - Memory Usage: Significant memory savings observed in reserved memory with the compiled benchmarks, making the implementation more resource-efficient.

2. Charts and Tables
   - Inference Speed per Token (LLaMA-7B and Granite-8B):
   DTensor implementation shows comparable performance to the baseline for large workloads.
   - Peak Memory Usage:
   Reserved memory savings in the modernized implementation outperform the baseline.

3. Observations
   - Scalability: The modernized implementation supports hybrid parallelism and lays the groundwork for integrating Sequence Parallelism.
   - Challenges: Communication costs remain a bottleneck, particularly for small sequence lengths.


## Performance Benchmark Results

### Inference Speed for Llama - 7B

| **Sequence Length** | **Benchmarked Repository** | **Uncompiled Single Token Generation** | **Uncompiled End-to-End Sequence Generation** | **Compiled Single Token Generation** | **Compiled End-to-End Sequence Generation** |
|----------------------|----------------------------|-----------------------------------------|-----------------------------------------------|---------------------------------------|---------------------------------------------|
|                      |                            | **Cached**     | **Uncached**     | **Cached**     | **Uncached**     | **Cached**     | **Uncached**     | **Cached**     | **Uncached**     |
| **256**              | IBM FMS                   | 54.26 ms       | 127.07 ms        | 54.21 ms       | 194.23 ms        | 19.48 ms       | 120.00 ms        | 20.12 ms       | 181.42 ms        |
|                      | TP FMS                    | 141.83 ms      | 148.56 ms        | 137.17 ms      | 189.05 ms        | 22.34 ms       | 123.64 ms        | 22.28 ms       | 175.24 ms        |
| **512**              | IBM FMS                   | 54.17 ms       | 245.07 ms        | 54.63 ms       | 318.71 ms        | 19.43 ms       | 230.54 ms        | 20.43 ms       | 298.40 ms        |
|                      | TP FMS                    | 141.09 ms      | 246.47 ms        | 136.77 ms      | 305.35 ms        | 23.03 ms       | 235.35 ms        | 22.68 ms       | 286.20 ms        |
| **1024**             | IBM FMS                   | 54.45 ms       | 518.54 ms        | 56.11 ms       | 613.34 ms        | 20.50 ms       | 464.63 ms        | 21.55 ms       | 539.72 ms        |
|                      | TP FMS                    | 140.56 ms      | 520.62 ms        | 138.98 ms      | 591.14 ms        | 24.00 ms       | 468.72 ms        | 23.57 ms       | 519.03 ms        |

---
### Inference Speed for Granite - 8B

| **Sequence Length** | **Benchmarked Repository** | **Uncompiled Single Token Generation** | **Uncompiled End-to-End Sequence Generation** | **Compiled Single Token Generation** | **Compiled End-to-End Sequence Generation** |
|----------------------|----------------------------|-----------------------------------------|-----------------------------------------------|---------------------------------------|---------------------------------------------|
|                      |                            | **Cached**     | **Uncached**     | **Cached**     | **Uncached**     | **Cached**     | **Uncached**     | **Cached**     | **Uncached**     |
| **256**              | IBM FMS                   | 75.71 ms       | 163.55 ms        | 73.55 ms       | 237.83 ms        | 28.05 ms       | 153.32 ms        | 27.26 ms       | 221.50 ms        |
|                      | TP FMS                    | 167.36 ms      | 173.43 ms        | 165.63 ms      | 237.35 ms        | 28.20 ms       | 154.70 ms        | 28.56 ms       | 220.49 ms        |
| **512**              | IBM FMS                   | 76.02 ms       | 303.16 ms        | 74.60 ms       | 389.98 ms        | 27.79 ms       | 283.00 ms        | 27.69 ms       | 362.11 ms        |
|                      | TP FMS                    | 169.85 ms      | 300.46 ms        | 168.57 ms      | 386.65 ms        | 28.69 ms       | 281.39 ms        | 29.19 ms       | 358.90 ms        |
| **1024**             | IBM FMS                   | 75.93 ms       | 648.20 ms        | 76.03 ms       | 746.85 ms        | 28.04 ms       | 576.15 ms        | 28.56 ms       | 653.90 ms        |
|                      | TP FMS                    | 168.26 ms      | 648.20 ms        | 168.08 ms      | 744.68 ms        | 28.21 ms       | 578.60 ms        | 29.82 ms       | 652.29 ms        |

---

### Memory Performance for Llama - 7B

| **SEQ LEN** | **REPO**   | **Uncompiled Single Token Generation**  | **Uncompiled End-to-End Sequence Generation**  | **Compiled Single Token Generation**  | **Compiled End-to-End Sequence Generation**  |
|-------------|------------|------------------------------------------|------------------------------------------------|---------------------------------------|----------------------------------------------|
|             |            | **Cached Allocated** | **Cached Reserved** | **Uncached Allocated** | **Uncached Reserved** | **Cached Allocated** | **Cached Reserved** | **Uncached Allocated** | **Uncached Reserved** |
| **256**     | IBM FMS    | 3.76 GB             | 4.13 GB             | 3.71 GB                | 4.13 GB                | 3.97 GB             | 4.13 GB             | 3.91 GB                | 6.66 GB                |
|             | TP FMS     | 3.94 GB             | 4.20 GB             | 3.90 GB                | 4.20 GB                | 4.14 GB             | 4.20 GB             | 4.00 GB                | 5.84 GB                |
| **512**     | IBM FMS    | 3.97 GB             | 4.80 GB             | 3.89 GB                | 4.80 GB                | 4.31 GB             | 4.82 GB             | 4.17 GB                | 8.71 GB                |
|             | TP FMS     | 4.17 GB             | 4.80 GB             | 4.09 GB                | 4.80 GB                | 4.50 GB             | 4.80 GB             | 4.24 GB                | 7.51 GB                |
| **1024**    | IBM FMS    | 4.51 GB             | 6.22 GB             | 4.34 GB                | 6.32 GB                | 5.18 GB             | 6.32 GB             | 4.79 GB                | 13.65 GB               |
|             | TP FMS     | 4.70 GB             | 6.23 GB             | 4.55 GB                | 6.22 GB                | 5.37 GB             | 6.25 GB             | 4.78 GB                | 11.41 GB               |

---

### Memory Performance for Granite - 8B

| **SEQ LEN** | **REPO**   | **Uncompiled Single Token Generation**  | **Uncompiled End-to-End Sequence Generation**  | **Compiled Single Token Generation**  | **Compiled End-to-End Sequence Generation**  |
|-------------|------------|------------------------------------------|------------------------------------------------|---------------------------------------|----------------------------------------------|
|             |            | **Cached Allocated** | **Cached Reserved** | **Uncached Allocated** | **Uncached Reserved** | **Cached Allocated** | **Cached Reserved** | **Uncached Allocated** | **Uncached Reserved** |
| **256**     | IBM FMS    | 4.58 GB             | 5.04 GB             | 4.59 GB                | 5.05 GB                | 4.76 GB             | 5.05 GB             | 4.87 GB                | 6.58 GB                |
|             | TP FMS     | 4.57 GB             | 4.99 GB             | 4.59 GB                | 4.99 GB                | 4.75 GB             | 4.99 GB             | 4.86 GB                | 6.22 GB                |
| **512**     | IBM FMS    | 4.68 GB             | 5.66 GB             | 4.69 GB                | 5.66 GB                | 5.00 GB             | 5.67 GB             | 5.10 GB                | 12.05 GB               |
|             | TP FMS     | 4.69 GB             | 5.58 GB             | 4.70 GB                | 5.60 GB                | 5.00 GB             | 5.58 GB             | 5.11 GB                | 14.63 GB               |
| **1024**    | IBM FMS    | 4.94 GB             | 6.61 GB             | 4.95 GB                | 6.61 GB                | 5.56 GB             | 6.61 GB             | 5.62 GB                | 22.58 GB               |
|             | TP FMS     | 4.94 GB             | 6.57 GB             | 4.96 GB                | 6.59 GB                | 5.56 GB             | 6.59 GB             | 5.60 GB                | 22.57 GB               |

---

### Notes:
- **Cached**: Repeated runs leveraging cached data for faster execution.
- **Uncached**: Fresh runs without caching.
- **IBM FMS**: Baseline results from the existing IBM Foundation Model Stack implementation.
- **TP FMS**: Results from the modernized Tensor Parallel implementation.

----------------------------------------------------------------------------------------------
## Models Supported
| Model family | Inference | Tuning and Training |
|--------------| ---------- | ------------------ |
| LLaMA        | :heavy_check_mark: | :heavy_check_mark: |
| GPT-BigCode  | :heavy_check_mark: | :x: |
| RoBERTa      | :heavy_check_mark: | :x: |


## Installation

We recommend running this on Python 3.11 and CUDA 12.1 for best performance, as the CPU overheads of the models are reduced significantly.

### Pypi

```
pip install ibm-fms
```

### Local

Requires [PyTorch >= 2.1](https://pytorch.org/get-started/locally/).

```
pip install -e .
```
or
```
python setup.py install
```


## Inference

#### Approach
Our approach for inference optimization is to use PyTorch compile, accelerated transformers, and tensor parallelism. PyTorch compile compiles the code into optimized kernels, accelerated transformers leverages `scaled_dot_product_attention` (SDPA) for accelerating attention computation while saving memory, and tensor parallelism is necessary for larger models.

To enable the Llama models to compile, we had to reimplement `RoPE` encodings without complex numbers. With this change, Llama model inference is able to leverage model compilation for latency reduction.

#### Inference latency
We measured inference latencies with 1024 token prompt and generation of 256 tokens on AWS P4de instance nodes with 8 80G A100 GPUs and report the median latency in the below table.
| Model | # GPUs | Median latency (ms) |
| ----- | ----------- | ----- |
| 7B | 1 | 14ms |
| 13B | 1 | 22ms |
| 70B | 8 | 30ms |

If you would like to reproduce the latencies, you can run the `scripts/benchmark_inference.py` and the details are described in [inference](./scripts).

For more information on reproducing the benchmarks and running some examples, see [here](scripts/README.md)

## HF Model Support

The support for HF models is provided by our HF model adapter. One can obtain similar latencies as tabulated above with HF models using our HF model adapter:

```python
from fms.models import get_model
from fms.models.hf import to_hf_api
import torch
from transformers import pipeline
# fms model
llama = get_model("llama", "13b")

# huggingface model backed by fms internals
llama_hf = to_hf_api(llama)

# compile the model -- in HF, the decoder only
llama_hf.decoder = torch.compile(llama_hf.decoder)

# generate some text -- the first time will be slow since the model needs to be compiled, but subsequent generations should be faster.
llama_generator = pipeline(task="text-generation", model=llama_hf, tokenizer=tokenizer)
llama_generator("""q: how are you? a: I am good. How about you? q: What is the weather like today? a:""")
```

A detailed example is provided [here](./notebooks/hf_adapted_inference.ipynb).

## Tuning

To fine-tune LLaMA, use the `scripts/train_causal.py` training script. Here's
an example of that command.
```
torchrun --nproc_per_node=2 \
        scripts/train_causal.py \
        --architecture=llama \
        --variant=7b \
        --tokenizer=~/models/tokenizer.model \
        --model_path=~/models/7B/ \
        --report_steps=10 \
        --checkpoint_format=meta \
        --distributed=fsdp
```
See options in the script for other ways to train and tune.

## Structure and contents of this Repository

* `fms/models/` - Pure pytorch implementations of popular model architectures, without requiring any specific common interface beyond `nn.Module`. Each model configuration is registered with `fms.models.register_model()` so that instances can be obtained through `fms.models.get_model('architecture', 'variant', '/path/to/data')`. Each model can also register sources/formats/versions of data to load (e.g. checkpoints provided by meta, HF, or trained from this repo). Users of the repo (e.g. `fms-extras`) can register their own model architectures as well.
* `fms/models/hf/` - Adapters that compose our native PyTorch FMS model architecture implementations in HF-compatible wrapper interfaces. Each FMS model implements an adapter, and adapted instances are obtained via `fms.models.hf.to_hf_api(model)`
* `fms/datasets/` - Code for loading data for pre-training and fine-tuning. Individual datasets are retrieved by `fms.datasets.get_dataset('name', tokenizer, 'optional path or other data reference')`. The expected tokenizer conforms to an `fms.utils.tokenizers.BaseTokenizer` interface.
* `fms/modules/` - Components extending `nn.Module` used in our model architecture implementations. Each Module has a corresponding `TPModule` so that modules can be sharded using a tensor-parallel distribution strategy. FMS modules should all support `torch.compile` without graph breaks.
* `fms/training/` - Pre-training and fine-tuning code.
* `fms/utils/` - Other operators useful in working with LLMs. These include a `generate()` function, `Tensor` subclasses, code for dealing with LLM checkpoints that might be saved/sharded in a variety of formats, tokenization code, and various other useful helper functions.
* `scripts/` - Various scripts for inference, benchmarking, and evaluation, as well as an entry-point for tuning/training.

## Extensions and Use Cases

This library is used by [three](https://github.com/foundation-model-stack/foundation-model-stack/network/dependents) dependent projects at IBM.

* [fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp) - This repo shares training code that has been used to pretrain an fms implementation of LLaMA on IBM internal data.
* [fms-extras](https://github.com/foundation-model-stack/fms-extras) - This repo shares code for additional fms-based models trained by IBM. This repo will also be a home for other extensions, and may also include research or in-developent work intended for eventual upstreaming to fms.
* [TGIS](https://github.com/IBM/text-generation-inference) - This inference server includes support for serving fms models.

## Open Issues

* https://github.com/pytorch/pytorch/issues/107824 prevents training/finetuning from working with `torch.compile`.
* In addition, there are several open issues we are tracking to improve stability and memory footprint of inference
  
## References

* Huggingface TGI: https://github.com/huggingface/text-generation-inference
* IBM TGIS: https://github.com/IBM/text-generation-inference
