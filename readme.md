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
| Remote server | Intel Core i9-11900KF (8C/16T), 64 GB RAM, NVIDIA GeForce RTX 3090 24 GB |




## Files

- `prompts/`: input prompts
- `scripts/run_benchmark.py`: runs inference and collects measurements
- `scripts/Consolidate.R`: combines results and generates summaries and plots
- `scripts/run_remote_benchmark.py`: runs the remote benchmark (when configured)
- `data/carbon_intensity_nl.csv`: Netherlands grid carbon intensity (15-minute values) for the experiment period, used for the SCI calculation
- `results/raw/`: measurements and responses from individual runs
- `results/consolidated/`: CSV summaries and plots



## Prerequisites

What you need depends on what you want to do:

| Goal | Needed |
| ---- | ------ |
| Reproduce the analysis, tables and figures from the included raw data | R 4.x and the R packages below |
| Run the on device benchmark | macOS on Apple Silicon (tested on an M4 MacBook Air, 24 GB), Python 3.10+, llama.cpp, EnergiBridge, the GGUF models |
| Run the remote benchmark | Everything above, plus a Linux server with an NVIDIA GPU prepared as in [Remote server setup](#remote-server-setup) |

Tools:

- [R](https://cran.r-project.org/) 4.x (for `scripts/Consolidate.R`)
- Python 3.10+ (for the benchmark runners)
- [llama.cpp](https://github.com/ggml-org/llama.cpp): `llama-server` must be on your `PATH` (macOS: `brew install llama.cpp`)
- [EnergiBridge](https://github.com/tdurieux/EnergiBridge): `energibridge` must be on your `PATH` (build it from source with Rust/Cargo, following its README; on macOS it may need `sudo` to read power counters)
- On the remote server: `nvidia-smi` (comes with the NVIDIA driver), used for GPU usage and power

Model files (GGUF), placed in `models/` with these exact names:

| File | Source |
| ---- | ------ |
| `models/Qwen3-8B-Q4_K_M.gguf` | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| `models/Qwen3-8B-Q8_0.gguf` | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| `models/kalomaze_Qwen3-16B-A3B-Q4_K_M.gguf` | Q4_K_M GGUF of [kalomaze/Qwen3-16B-A3B](https://huggingface.co/kalomaze/Qwen3-16B-A3B) |

The model files are large and are not stored in the repository. For remote runs, the same files must
also be available on the server in `DEFAULT_REMOTE_MODEL_DIR`.

## Setup

1. Clone the repository and go to its root:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the R packages (needed for the analysis):

   ```r
   install.packages(c("jsonlite", "readr", "dplyr", "purrr", "ggplot2", "tidyr", "ARTool"))
   ```

3. Create a Python virtual environment and install the Python packages (needed for the benchmark):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Download the GGUF models into `models/` (see the table above).

5. Create a `.env` file in the repository root with the server settings:

   ```
   DEFAULT_SERVER_HOST=127.0.0.1
   DEFAULT_LLAMA_PORT=8080
   # remote runs only
   DEFAULT_REMOTE_HOST=...
   DEFAULT_REMOTE_MODEL_DIR=...
   DEFAULT_REMOTE_PROMPT_DIR=...
   DEFAULT_REMOTE_RESULTS_DIR=...
   DEFAULT_REMOTE_SCRIPT=...
   DEFAULT_REMOTE_LLAMA_SERVER=...
   DEFAULT_AGENT_PORT=...
   DEFAULT_NETWORK_DELAY_IP=...
   ```

6. Check that the tools are found:

   ```bash
   llama-server --version
   energibridge --help
   Rscript --version
   ```

### Quick start: reproduce the results only

The raw measurements of all 360 runs are included in `results/raw/`, so the analysis can be
reproduced without the benchmark hardware. Only R and the R packages are needed:

```bash
Rscript scripts/Consolidate.R
```

## Remote server setup

The remote runner (`scripts/run_remote_benchmark.py`) is started on the MacBook. Over SSH it starts
`llama-server` on the server, starts a small benchmark agent there (the same script, run with
`--agent`) under EnergiBridge, and forwards the llama-server and agent ports to the MacBook through an
SSH tunnel. The server therefore needs its own copy of the runner, the prompts and the models. The
steps below use `<remote-host>` for the SSH host alias and `~/greenlabbench` as the working directory;
replace `<user>` with your account name on the server.

1. **Passwordless SSH from the MacBook.** The runner calls `ssh` and `scp` without a terminal, so
   key-based login must work without prompts. Add the host to `~/.ssh/config` on the MacBook:

   ```
   Host <remote-host>
       HostName <server address>
       User <user>
       IdentityFile ~/.ssh/id_ed25519
   ```

   Check it with `ssh <remote-host> nvidia-smi`, which must print the GPU without asking for a password.

2. **Create the working directory on the server:**

   ```bash
   ssh <remote-host> 'mkdir -p ~/greenlabbench/scripts ~/greenlabbench/prompts ~/greenlabbench/models ~/greenlabbench/results'
   ```

3. **Copy the runner and the prompts to the server** (from the repository root on the MacBook):

   ```bash
   scp -O scripts/run_remote_benchmark.py requirements.txt <remote-host>:~/greenlabbench/scripts/
   scp -O prompts/prompt_small.txt prompts/prompt_medium.txt prompts/prompt_large.txt <remote-host>:~/greenlabbench/prompts/
   ```

4. **Create the Python environment on the server.** The runner activates the virtual environment
   one level above the script folder, i.e. `~/greenlabbench/venv`:

   ```bash
   ssh <remote-host>
   cd ~/greenlabbench
   python3 -m venv venv
   source venv/bin/activate
   pip install psutil python-dotenv
   ```

5. **Build llama.cpp with CUDA on the server.** The runner starts
   `llama-server` from a CUDA build of llama.cpp. Its absolute path is read from
   `DEFAULT_REMOTE_LLAMA_SERVER` in `.env` (step 9). Building it in your home directory as below gives
   `/home/<user>/llama.cpp/build-cuda/bin/llama-server`:

   ```bash
   cd ~
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   cmake -B build-cuda -DGGML_CUDA=ON
   cmake --build build-cuda --config Release -j
   ```

   This needs the NVIDIA driver, the CUDA toolkit and CMake on the server.

6. **Install EnergiBridge on the server** and make sure `energibridge` is on the `PATH` of
   non-interactive SSH sessions (for example by installing it in `/usr/local/bin`). On Linux it reads
   the CPU and DRAM energy counters (RAPL) and the GPU power through NVML; follow the EnergiBridge
   README to give your user read access to the RAPL counters.

7. **Copy the models to the server** with the same file names as on the MacBook (or download them
   there directly):

   ```bash
   scp -O models/Qwen3-8B-Q4_K_M.gguf models/Qwen3-8B-Q8_0.gguf \
       models/kalomaze_Qwen3-16B-A3B-Q4_K_M.gguf <remote-host>:~/greenlabbench/models/
   ```

   The runner looks for `<DEFAULT_REMOTE_MODEL_DIR>/<local model file name>` and checks the GGUF
   header and file size before starting the server.

8. **Network delay (only for the +100 ms and +500 ms conditions).** The `agent-delay` wrapper must be
   in your home directory on the server (or pass its path with `--agent-delay-script`), and your user
   must be allowed to run `sudo -n /usr/local/bin/agent-delay-clear` without a password. On a shared
   lab server, ask the administrators to provide both.

9. **Fill in the remote settings in `.env` on the MacBook:**

   ```
   DEFAULT_SERVER_HOST=127.0.0.1
   DEFAULT_LLAMA_PORT=8080
   DEFAULT_AGENT_PORT=8090
   DEFAULT_REMOTE_HOST=<remote-host>
   DEFAULT_REMOTE_MODEL_DIR=/home/<user>/greenlabbench/models
   DEFAULT_REMOTE_PROMPT_DIR=/home/<user>/greenlabbench/prompts
   DEFAULT_REMOTE_RESULTS_DIR=/home/<user>/greenlabbench/results
   DEFAULT_REMOTE_SCRIPT=/home/<user>/greenlabbench/scripts/run_remote_benchmark.py
   DEFAULT_REMOTE_LLAMA_SERVER=/home/<user>/llama.cpp/build-cuda/bin/llama-server
   DEFAULT_NETWORK_DELAY_IP=<IP address used by agent-delay>
   ```

   Use absolute paths. The two ports only need to be free on both machines; they are forwarded
   through the SSH tunnel, so no firewall ports have to be opened.

10. **Check the server before a full run:**

    ```bash
    ssh <remote-host> 'ls ~/greenlabbench/prompts ~/greenlabbench/models; \
      ~/llama.cpp/build-cuda/bin/llama-server --version; \
      command -v energibridge; ~/greenlabbench/venv/bin/python -c "import psutil"'
    ```

During a run the server writes its measurements to `~/greenlabbench/results/`; the runner copies
each run's files back to `results/raw/remote/` on the MacBook with `scp`.

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

Prepare the server first (see [Remote server setup](#remote-server-setup)). Results go to `results/raw/remote/`. For the 100 ms and 500 ms delay conditions, see [Remote network delay](#remote-network-delay).

## Analyze results

Run the analysis from the repository root so that its relative input and output paths resolve correctly:

```bash
Rscript scripts/Consolidate.R
```

The script reads `results/raw/`, writes CSV files to `results/consolidated/` and plots to
`results/consolidated/plots/` (the plots folder is emptied first, so no stale figures remain). All
results, tables and figures can be regenerated with this one command.

Statistical procedure: each treatment was run in its own session, so the groups are compared as
independent samples within the same prompt length. Shapiro-Wilk is applied to each group; if both are
normal a Welch t-test with Cohen's d is used, otherwise a Mann-Whitney U test with Cliff's delta.
RQ4 (three delay levels) first uses an omnibus test (Welch ANOVA if all groups are normal, otherwise
Kruskal-Wallis). The RQ1 setup × prompt-length interaction is tested with a mixed ANOVA (ART ANOVA if
the residuals are not normal). All p-values are Holm-corrected within each research question
(RQ2 and RQ3: within each setup).

| Output | Content |
| ------ | ------- |
| `run_table.csv` | One row per valid run, including energy, timing, grid intensity and carbon columns |
| `data_quality.csv` | Every run with its validity flag |
| `outliers.csv` | IQR-rule outliers per group, and runs slower than 2x the group median |
| `RQ1_summary.csv` … `RQ4_summary.csv` | n, mean, SD, variance, median, IQR, min, max and 95% CI per condition |
| `RQ1_tests.csv` … `RQ4_tests.csv` | Independent-sample tests with per-group Shapiro-Wilk p, test used, Holm-adjusted p, Cohen's d and Cliff's delta |
| `RQ1_interaction_anova.csv` | Mixed two-way ANOVA (setup between, prompt length within) per model: residual Shapiro-Wilk p, mixed and ART ANOVA F/p, partial eta², test used, Holm-adjusted p |
| `RQ4_omnibus.csv` | Welch ANOVA or Kruskal-Wallis test across the three network conditions |
| `normality_summary.csv` | Number of cells where both groups pass Shapiro-Wilk, per metric |
| `paper_table_*.csv` | Compact, rounded summary tables (descriptive statistics and tests per RQ, SCI) |
| `SCI_summary.csv`, `SCI_summary_by_prompt.csv` | Carbon footprint per setup and model, by sub-component, and for the whole experiment |
| `SCI_assumptions.csv` | Embodied-carbon parameters and PUE for the central, low and high scenarios |

Plots:

- `paper_*.png`: compact versions of the main figures, drawn at single-column (3.4 in) size
- `RQ1_*` … `RQ4_*`: full-size distribution (violin + box) and Q-Q plots per research question
- `SCI_*`: carbon per output token by sub-component, per-run distribution, and grid intensity during the runs

The script reads the cached grid intensity file `data/carbon_intensity_nl.csv`. If the file is
missing or does not cover every run, the missing period is downloaded from Energy-Charts
(internet access needed only in that case).

## Remote network delay

The remote runner applies network delay with the remote `agent-delay` wrapper. The wrapper must be
available on the remote host and is invoked as `./agent-delay <target-ip> <agent-port> <delay-ms>`.
For example:

```bash
python scripts/run_remote_benchmark.py \
	--model models/Qwen3-8B-Q4_K_M.gguf \
	--quantization Q4_K_M \
	--architecture Dense \
	--network-delay-ip <target-ip> \
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
| I         | Netherlands grid intensity in the 15-minute slot each run started (about 494–639 g CO2e/kWh during the runs) | [Fraunhofer Energy-Charts](https://api.energy-charts.info/)                                                                       | Published data |
| R         | One output token                                                                                             | llama.cpp token count                                                                                                             | Measured       |
| TE laptop | 90 kg CO2e (120 kg lifecycle × ~75 % non-use share)                                                          | [Apple M4 MacBook Air PER, March 2025](https://www.apple.com/environment/pdf/products/notebooks/M4_MacBook_Air_PER_March2025.pdf) | Published      |
| TE server | 690 kg CO2e (540 kg for CPU, RAM, SSDs and chassis + ~150 kg for the RTX 3090)                               | [Boavizta API](https://api.boavizta.org/); GPU figure assumed                                                                     | Estimate       |
| TiR       | Wall time of each run                                                                                        | Benchmark                                                                                                                         | Measured       |
| EL        | 4 years for both machines                                                                                    | Apple PER; common server practice                                                                                                 | Convention     |
| RR / ToR  | 1 (dedicated laptop; the server is used exclusively during booked slots)                                                | Server booking                                                                                                                      | —              |


Remote runs carry the embodied share of both the server and the client laptop, because both are in use.

The operational part (E × I) is about 90 % of the total (server PUE = 1.0, a single on-premise workstation) and is based on measurements and time-resolved
grid data. The embodied part is an estimate, so the script also reports a low scenario (server 515 kg,
6-year lifetime) and a high scenario (server 1231 kg, 3-year lifetime). These are shown as whiskers in
`SCI_01_per_token_breakdown.png` / `paper_SCI_breakdown.png` and as `sci_low` / `sci_high` columns in `SCI_summary.csv`.

The boundary includes the client laptop and the remote server. It excludes network equipment between
them, warm-up and cooldown periods, and excluded runs. Grid intensity is a grid-average value, not a
measurement of the specific supply to each machine. To change any assumption, edit `SCI_SCENARIOS`
in `scripts/Consolidate.R`.

## AI Use Disclosure

Generative AI tools were used during the development of this project to support software development, debugging, analysis-script development, documentation, and interpretation of experimental results.

AI assistance was used to help refine code, identify and resolve programming errors, improve data-analysis and visualization approaches, and improve the clarity of project documentation. The experimental setup, benchmark execution, measurements, and final interpretation of the results remain the responsibility of the us.

AI-generated suggestions were reviewed and adapted before being incorporated into the project. No experimental measurements were generated by AI; measurements were obtained from the benchmark hardware and software described in this repository.