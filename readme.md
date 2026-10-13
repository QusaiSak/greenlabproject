# Green Lab Project

This project studies the energy consumption and performance of local LLM inference. The broader study compares on device and remote inference across prompt lengths, models, quantization levels, and network conditions.

The repository contains the on device and remote benchmark scripts, the raw results, the analysis script, and a carbon footprint calculation following the Green Software Foundation SCI specification.

## Models tested

- Qwen3-8B Q4_K_M
- Qwen3-8B Q8_0
- Qwen3-16B-A3B Q4_K_M

Each configuration was tested with small, medium, and large prompts, with 10 measured runs per prompt size, in four setups: on device, remote (baseline network), and remote with 100 ms and 500 ms of added one-way network delay.

## Hardware


| Role                                                       | Machine                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| On device / client                                         | MacBook Air 13-inch, Apple M4, 24 GB unified memory, 256 GB SSD          |
| Remote server (`glg3`, S2 Green Lab cluster, VU Amsterdam) | Intel Core i9-11900KF (8C/16T), 64 GB RAM, NVIDIA GeForce RTX 3090 24 GB |




## Files

- `prompts/`: input prompts
- `scripts/run_benchmark.py`: runs inference and collects measurements
- `scripts/Consolidate.R`: combines results and generates summaries and plots
- `scripts/run_remote_benchmark.py`: runs the remote benchmark (when configured)
- `data/carbon_intensity_nl.csv`: Netherlands grid carbon intensity (15-minute values) for the experiment period, used for the SCI calculation
- `results/raw/`: measurements and responses from individual runs
- `results/consolidated/`: CSV summaries and plots



## Setup

Requirements:

- Python 3
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` on your PATH)
- [EnergiBridge](https://github.com/tdurieux/EnergiBridge) (`energibridge` on your PATH)
- R (for the analysis)
- The GGUF model files in `models/` (downloaded separately)

Install the Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install psutil python-dotenv
```

Install the R packages:

```r
install.packages(c("jsonlite", "readr", "dplyr", "purrr", "ggplot2", "tidyr"))
```

Create a `.env` file in the repository root with the server settings:

```
DEFAULT_SERVER_HOST=127.0.0.1
DEFAULT_LLAMA_PORT=8080
# remote runs only
DEFAULT_REMOTE_HOST=...
DEFAULT_REMOTE_MODEL_DIR=...
DEFAULT_REMOTE_PROMPT_DIR=...
DEFAULT_REMOTE_RESULTS_DIR=...
DEFAULT_REMOTE_SCRIPT=...
DEFAULT_AGENT_PORT=...
DEFAULT_NETWORK_DELAY_IP=...
```



## Run the benchmark

All commands run from the repository root.

### On device

Run once per model:

```bash
python scripts/run_benchmark.py --model models/Qwen3-8B-Q4_K_M.gguf --quantization Q4_K_M --architecture Dense --runs 10
python scripts/run_benchmark.py --model models/Qwen3-8B-Q8_0.gguf --quantization Q8_0 --architecture Dense --runs 10
python scripts/run_benchmark.py --model models/kalomaze_Qwen3-16B-A3B-Q4_K_M.gguf --quantization Q4_K_M --architecture MoE --runs 10
```

The runner starts `llama-server`, warms up each prompt size, waits 30 seconds, and then runs the prompts in shuffled order with a 10-second cooldown between requests. The default output limit is 1536 tokens. Results go to `results/raw/`; use `--results-dir` for new experiments to avoid overwriting existing runs.

The on device runner uses macOS tools for GPU monitoring.

### Remote

The remote runner starts the model on the remote host over SSH and measures both sides:

```bash
python scripts/run_remote_benchmark.py --model models/Qwen3-8B-Q4_K_M.gguf --quantization Q4_K_M --architecture Dense --runs 10
```

Results go to `results/raw/remote/`. For the 100 ms and 500 ms delay conditions, see [Remote network delay](#remote-network-delay).

## Analyze results

Run the analysis from the repository root so that its relative input and output paths resolve correctly:

```bash
Rscript scripts/Consolidate.R
```

The script reads `results/raw/` and writes CSV summaries to `results/consolidated/`, with plots in
`results/consolidated/plots/`.


| Output                                         | Content                                                                            |
| ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| `run_table.csv`                                | One row per valid run, including energy, timing, grid intensity and carbon columns |
| `data_quality.csv`                             | Every run with its validity flag                                                   |
| `RQ1_summary.csv` … `RQ4_summary.csv`          | Mean, SD, median and 95% CI per condition                                          |
| `RQ1_tests.csv` … `RQ4_tests.csv`              | Wilcoxon rank-sum tests (Holm-corrected within each RQ) and Cliff's delta          |
| `SCI_summary.csv`, `SCI_summary_by_prompt.csv` | Carbon footprint per setup and model, and for the whole experiment                 |
| `SCI_assumptions.csv`                          | Embodied-carbon parameters for the central, low and high scenarios                 |


Main plots:

- `00_overview_energy_per_token.png`: every setup and model on one page
- `RQ1_*`: on device vs remote (energy per token vs prompt length, total energy, energy breakdown, remote/on device ratio)
- `RQ2_*`: Q4_K_M vs Q8_0 on device (all metrics, relative change)
- `RQ3_*`: dense vs MoE on device, and the speed vs energy trade-off of every configuration
- `RQ4_*`: effect of 100 ms and 500 ms network delay
- `SCI_*`: carbon per output token (operational vs embodied), carbon vs prompt length, and grid intensity during the runs

The script needs R packages only; it reads the cached grid intensity file and downloads it from
Energy-Charts only if `data/carbon_intensity_nl.csv` is missing.

## Remote network delay

The remote runner applies network delay with the remote `agent-delay` wrapper. The wrapper must be
available on the remote host and is invoked as `./agent-delay <target-ip> <agent-port> <delay-ms>`.
For example:

```bash
python scripts/run_remote_benchmark.py \
	--model models/Qwen3-8B-Q4_K_M.gguf \
	--quantization Q4_K_M \
	--architecture Dense \
	--network-delay-ip 145.108.225.3 \
	--network-delay-ms 100
```

Run it once with `--network-delay-ms 100` and once with `--network-delay-ms 500`. The delay is
applied one way only, to the benchmark agent's response port (server to client).

Use `--agent-delay-script` when the wrapper is not in the remote user's home directory. The runner
captures the wrapper PID and sends `kill -TERM <pid>` during cleanup, including when the benchmark is
interrupted. Remote runs store server-side energy in `energy.csv` and orchestrator/client-side energy
in `client_energy.csv`; the latter covers the complete remote HTTP request and response wait.

## Measurements

The benchmark records energy, execution time, token counts, throughput, CPU/GPU utilization, and memory usage. Energy per output token uses the actual generated-token count.

EnergiBridge measures system power during client execution, including overhead around the inference request. Warm-up and cooldown are excluded. Prompt and generation energy values are estimates based on phase duration.

The prompts differ in content as well as length.

## Carbon footprint (SCI)

The carbon footprint follows the [Green Software Foundation SCI specification](https://sci-guide.greensoftware.foundation/):

```
SCI = ((E × I) + M) per R
M   = TE × (TiR / EL) × (RR / ToR)
```


| Term      | Value in this project                                                                                        | Source                                                                                                                            | Type           |
| --------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| E         | Measured energy per run (`total_energy_j`); remote = client Mac + server CPU, DRAM and GPU                   | EnergiBridge                                                                                                                      | Measured       |
| I         | Netherlands grid intensity in the 15-minute slot each run started (about 520–630 g CO2e/kWh during the runs) | [Fraunhofer Energy-Charts](https://api.energy-charts.info/)                                                                       | Published data |
| R         | One output token                                                                                             | llama.cpp token count                                                                                                             | Measured       |
| TE laptop | 90 kg CO2e (120 kg lifecycle × ~75 % non-use share)                                                          | [Apple M4 MacBook Air PER, March 2025](https://www.apple.com/environment/pdf/products/notebooks/M4_MacBook_Air_PER_March2025.pdf) | Published      |
| TE server | 690 kg CO2e (540 kg for CPU, RAM, SSDs and chassis + ~150 kg for the RTX 3090)                               | [Boavizta API](https://api.boavizta.org/); GPU figure assumed                                                                     | Estimate       |
| TiR       | Wall time of each run                                                                                        | Benchmark                                                                                                                         | Measured       |
| EL        | 4 years for both machines                                                                                    | Apple PER; common server practice                                                                                                 | Convention     |
| RR / ToR  | 1 (dedicated laptop; `glg3` is exclusive during booked slots)                                                | Cluster policy                                                                                                                    | —              |


Remote runs carry the embodied share of both the server and the client laptop, because both are in use.

The operational part (E × I) is about 90 % of the total and is based on measurements and time-resolved
grid data. The embodied part is an estimate, so the script also reports a low scenario (server 515 kg,
6-year lifetime) and a high scenario (server 1231 kg, 3-year lifetime). These are shown as whiskers in
`SCI_01_per_token_breakdown.png` and as `sci_low` / `sci_high` columns in `SCI_summary.csv`.

The boundary includes the client laptop and the remote server. It excludes network equipment between
them, warm-up and cooldown periods, and excluded runs. Grid intensity is a grid-average value, not a
measurement of the specific supply to each machine. To change any assumption, edit `SCI_SCENARIOS`
in `scripts/Consolidate.R`.

## AI Use Disclosure

Generative AI tools were used during the development of this project to support software development, debugging, analysis-script development, documentation, and interpretation of experimental results.

AI assistance was used to help refine code, identify and resolve programming errors, improve data-analysis and visualization approaches, and improve the clarity of project documentation. The experimental setup, benchmark execution, measurements, and final interpretation of the results remain the responsibility of the us.

AI-generated suggestions were reviewed and adapted before being incorporated into the project. No experimental measurements were generated by AI; measurements were obtained from the benchmark hardware and software described in this repository.