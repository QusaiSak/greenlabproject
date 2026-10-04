# Green Lab Benchmark

This project compares LLM inference energy and performance across prompt lengths, model configurations, quantization levels, and local versus remote execution.

The local testbed is a MacBook Air M4 with 24 GB unified memory. The remote testbed is the `glg3` server with an NVIDIA RTX 3090. The remote runner is orchestrated from the MacBook over SSH.

## Models tested

- Qwen3-8B Q4_K_M
- Qwen3-8B Q8_0
- Qwen3-16B-A3B Q4_K_M

Each configuration was tested with small, medium, and large prompts, with 10 measured runs per prompt size. The local experiments used a MacBook Air M4 with 24 GB unified memory.

## Repository layout

- `prompts/`: input prompts
- `scripts/run_benchmark.py`: local inference benchmark
- `scripts/run_remote_benchmark.py`: MacBook-controlled remote benchmark
- `scripts/consolidate.py`: combines results and generates summaries and plots
- `results/raw/`: measurements and responses from individual runs
- `results/consolidated/`: CSV summaries and plots

## Setup

The benchmark requires Python 3, llama.cpp (`llama-server`), EnergiBridge (`energibridge`), and GGUF model files. Install llama.cpp and EnergiBridge separately and make sure both commands are available on the relevant machine's PATH.

The Python environment needs `pandas`, `matplotlib`, `psutil`, and `numpy` for consolidation, plotting, and monitoring. The repository currently uses the existing `venv` environment rather than a committed `requirements.txt` file.

Use the project interpreter:

```bash
venv/bin/python --version
```

The local runner uses macOS tools for GPU monitoring. Model files must be downloaded separately.

## Run the benchmark

From the repository root:

```bash
venv/bin/python scripts/run_benchmark.py \
	--model models/Qwen3-8B-Q4_K_M.gguf \
	--quantization Q4_K_M \
	--architecture Dense
```

Change the model path and quantization label for other models. Use `--architecture MoE` for Qwen3-16B-A3B.

The runner warms up all prompt sizes in a shuffled order, waits 30 seconds for stabilization, and then executes ten randomized blocks. Each block contains one small, one medium, and one large prompt. There is a 10-second cooldown between measured requests. The randomization seed is `42`, and the default output limit is 1536 tokens.

Use a different `--results-dir` for new experiments to avoid overwriting existing runs.

## Analyze results

```bash
venv/bin/python scripts/consolidate.py
```

This writes CSV summaries and plots to `results/consolidated/`. Use `--no-plots` for CSV output only. Both scripts support `--help`.

The consolidation produces comparisons for:

- Prompt length
- Model configuration
- Quantization
- Architecture
- Energy, execution time, throughput, CPU, GPU, and memory

## Measurements

The benchmark records energy, execution time, token counts, throughput, CPU/GPU utilization, and memory usage. Energy per output token uses the actual generated-token count.

EnergiBridge measures sampled system power during the measured request. Warm-up and cooldown are excluded. Prompt and generation energy values are estimates based on phase duration because the power trace does not expose an exact prompt/generation boundary.

The prompts differ in content as well as length, so prompt-size comparisons are not a perfectly controlled token-only experiment.

## Remote benchmark

The remote benchmark is started from the MacBook. Copy the remote script to the server once:

```bash
scp -O scripts/run_remote_benchmark.py \
	glg3:##/scripts/run_remote_benchmark.py
```

The server-side paths are:

```text
##/models
##/prompts
##/results/raw_remote
##/scripts/run_remote_benchmark.py
```

Run a remote benchmark from the repository root:

```bash
venv/bin/python scripts/run_remote_benchmark.py \
	--model models/Qwen3-8B-Q8_0.gguf \
	--quantization Q8_0 \
	--architecture Dense \
	--network-condition baseline
```

The MacBook starts the remote `llama-server` through SSH, starts the remote agent under EnergiBridge, sends the inference request over HTTP, and retrieves the run files with SCP. Remote results downloaded to the MacBook are stored under `results/raw/remote`.

The remote runner records application-level request/response bytes and a health-request round-trip time. These are not total Ethernet-interface counters and do not include SSH or SCP traffic.

