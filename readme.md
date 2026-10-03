# Green Lab Project

This project studies the energy consumption and performance of local LLM inference. The broader study compares local and remote inference across prompt lengths, models, quantization levels, and network conditions.

The repository currently contains the local benchmark scripts and results.

## Models tested

- Qwen3-8B Q4_K_M
- Qwen3-8B Q8_0
- Qwen3-16B-A3B Q4_K_M

Each configuration was tested with small, medium, and large prompts, with 10 measured runs per prompt size. The local experiments used a MacBook Air M4 with 24 GB unified memory.

## Files

- `prompts/`: input prompts
- `scripts/run_benchmark.py`: runs inference and collects measurements
- `scripts/consolidate.py`: combines results and generates summaries and plots
- `results/raw/`: measurements and responses from individual runs
- `results/consolidated/`: CSV summaries and plots

## Setup

The benchmark requires Python 3, llama.cpp (`llama-server`), EnergiBridge (`energibridge`), and GGUF model files. Install llama.cpp and EnergiBridge separately and make sure both commands are available on your PATH.

Install the Python packages:

```bash
python -m pip install psutil pandas matplotlib numpy
```

The local runner uses macOS tools for GPU monitoring. Model files must be downloaded separately.

## Run the benchmark

From the repository root:

```bash
python scripts/run_benchmark.py --model models/Qwen3-8B-Q4_K_M.gguf --quantization Q4_K_M --architecture Dense
```

Change the model path and quantization label for other models. Use `--architecture MoE` for Qwen3-16B-A3B.

The runner warms up each prompt size, waits 30 seconds, and then runs the prompts in shuffled order with a 10-second cooldown between requests. The default output limit is 1536 tokens.

Use a different `--results-dir` for new experiments to avoid overwriting existing runs.

## Analyze results

```bash
python scripts/consolidate.py
```

This writes CSV summaries and plots to `results/consolidated/`. Use `--no-plots` for CSV output only. Both scripts support `--help`.

## Measurements

The benchmark records energy, execution time, token counts, throughput, CPU/GPU utilization, and memory usage. Energy per output token uses the actual generated-token count.

EnergiBridge measures system power during client execution, including overhead around the inference request. Warm-up and cooldown are excluded. Prompt and generation energy values are estimates based on phase duration.

The prompts differ in content as well as length. Remote benchmarking and network-condition tests are not included in the current scripts.
