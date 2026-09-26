import argparse
import json
from pathlib import Path

import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def read_metadata(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def find_column(
    df,
    candidates,
):

    columns_lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        candidate_lower = (
            candidate.lower()
        )

        if candidate_lower in columns_lower:

            return columns_lower[
                candidate_lower
            ]

    for column in df.columns:

        column_lower = (
            str(column).lower()
        )

        for candidate in candidates:

            if (
                candidate.lower()
                in column_lower
            ):

                return column

    return None


def mean_column(
    df,
    candidates,
):

    column = find_column(
        df,
        candidates,
    )

    if column is None:

        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.dropna().empty:

        return None

    return float(
        values.mean()
    )


def max_column(
    df,
    candidates,
):

    column = find_column(
        df,
        candidates,
    )

    if column is None:

        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.dropna().empty:

        return None

    return float(
        values.max()
    )


# ============================================================
# ENERGY
# ============================================================

def calculate_energy_from_power(
    df,
):

    power_column = find_column(
        df,
        [
            "SYSTEM_POWER (Watts)",
            "SYSTEM_POWER",
        ],
    )

    delta_column = find_column(
        df,
        [
            "Delta",
        ],
    )

    if (
        power_column is None
        or delta_column is None
    ):

        return None

    power = pd.to_numeric(
        df[power_column],
        errors="coerce",
    )

    delta_ms = pd.to_numeric(
        df[delta_column],
        errors="coerce",
    )

    valid = (
        power.notna()
        & delta_ms.notna()
    )

    if not valid.any():

        return None

    # Energy = Power × time
    #
    # W × milliseconds / 1000 = J

    energy_j = (
        power[valid]
        * delta_ms[valid]
        / 1000.0
    ).sum()

    return float(
        energy_j
    )


# ============================================================
# PROCESS ONE RUN
# ============================================================

def process_run(
    run_dir,
):

    energy_file = (
        run_dir / "energy.csv"
    )

    metadata_file = (
        run_dir / "metadata.json"
    )

    if not energy_file.exists():

        return None

    metadata = read_metadata(
        metadata_file
    )

    try:

        df = pd.read_csv(
            energy_file
        )

    except Exception as error:

        print(
            f"Could not read "
            f"{energy_file}: {error}"
        )

        return None

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    llama_metrics = metadata.get(
        "llama_metrics",
        {},
    )

    system_metrics = metadata.get(
        "system_metrics",
        {},
    )

    configuration = metadata.get(
        "configuration",
        {},
    )

    model = metadata.get(
        "model",
        "",
    )

    quantization = metadata.get(
        "quantization",
        "",
    )

    prompt_size = metadata.get(
        "prompt_size",
        "",
    )

    run_number = metadata.get(
        "run",
        "",
    )

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    energy_j = (
        calculate_energy_from_power(
            df
        )
    )

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    execution_time_s = None

    if "Delta" in df.columns:

        delta = pd.to_numeric(
            df["Delta"],
            errors="coerce",
        )

        execution_time_s = (
            delta.sum()
            / 1000.0
        )

    if execution_time_s is None:

        execution_time_s = metadata.get(
            "wall_clock_time_s"
        )

    if execution_time_s is not None:

        execution_time_s = float(
            execution_time_s
        )

    # --------------------------------------------------------
    # llama.cpp metrics
    # --------------------------------------------------------

    prompt_tokens = (
        llama_metrics.get(
            "prompt_tokens"
        )
    )

    output_tokens = (
        llama_metrics.get(
            "output_tokens"
        )
    )

    prompt_eval_time_ms = (
        llama_metrics.get(
            "prompt_eval_time_ms"
        )
    )

    generation_time_ms = (
        llama_metrics.get(
            "generation_time_ms"
        )
    )

    # --------------------------------------------------------
    # Energy efficiency
    # --------------------------------------------------------

    energy_j_per_token = None

    if (
        energy_j is not None
        and output_tokens is not None
        and output_tokens > 0
    ):

        energy_j_per_token = (
            energy_j
            / output_tokens
        )

    # --------------------------------------------------------
    # Generation throughput
    # --------------------------------------------------------

    generation_time_s = None

    tokens_per_second = None

    if generation_time_ms is not None:

        generation_time_s = (
            generation_time_ms
            / 1000.0
        )

    if (
        output_tokens is not None
        and generation_time_s is not None
        and generation_time_s > 0
    ):

        tokens_per_second = (
            output_tokens
            / generation_time_s
        )

    # --------------------------------------------------------
    # Independent CPU measurements
    # --------------------------------------------------------

    cpu_avg = (
        system_metrics.get(
            "cpu_usage_avg_pct"
        )
    )

    cpu_max = (
        system_metrics.get(
            "cpu_usage_max_pct"
        )
    )

    # --------------------------------------------------------
    # Independent GPU measurements
    # --------------------------------------------------------

    gpu_avg = (
        system_metrics.get(
            "gpu_usage_avg_pct"
        )
    )

    gpu_max = (
        system_metrics.get(
            "gpu_usage_max_pct"
        )
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    memory_avg = (
        system_metrics.get(
            "memory_avg_mb"
        )
    )

    memory_max = (
        system_metrics.get(
            "memory_max_mb"
        )
    )

    memory_avg_pct = (
        system_metrics.get(
            "memory_avg_pct"
        )
    )

    memory_max_pct = (
        system_metrics.get(
            "memory_max_pct"
        )
    )

    # --------------------------------------------------------
    # GPU data from EnergiBridge if available
    # --------------------------------------------------------

    gpu_power = mean_column(
        df,
        [
            "GPU_POWER (Watts)",
            "GPU_POWER",
        ],
    )

    gpu_memory = mean_column(
        df,
        [
            "GPU_MEMORY (Bytes)",
            "GPU_MEMORY",
        ],
    )

    gpu_memory_mb = None

    if gpu_memory is not None:

        gpu_memory_mb = (
            gpu_memory
            / 1024
            / 1024
        )

    # --------------------------------------------------------
    # GPU energy estimate
    # --------------------------------------------------------

    gpu_energy_j = None

    if (
        gpu_power is not None
        and execution_time_s is not None
    ):

        gpu_energy_j = (
            gpu_power
            * execution_time_s
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {

        "model":
            model,

        "quantization":
            quantization,

        "prompt_size":
            prompt_size,

        "run":
            run_number,

        # Prompt/output
        "prompt_tokens":
            prompt_tokens,

        "output_tokens":
            output_tokens,

        # Time
        "execution_time_s":
            execution_time_s,

        "prompt_eval_time_ms":
            prompt_eval_time_ms,

        "generation_time_ms":
            generation_time_ms,

        "generation_time_s":
            generation_time_s,

        "tokens_per_second":
            tokens_per_second,

        # Energy
        "energy_j":
            energy_j,

        "energy_j_per_output_token":
            energy_j_per_token,

        # CPU
        "cpu_usage_avg_pct":
            cpu_avg,

        "cpu_usage_max_pct":
            cpu_max,

        # GPU
        "gpu_usage_avg_pct":
            gpu_avg,

        "gpu_usage_max_pct":
            gpu_max,

        "gpu_power_w":
            gpu_power,

        "gpu_energy_j":
            gpu_energy_j,

        "gpu_memory_mb":
            gpu_memory_mb,

        # RAM
        "memory_avg_mb":
            memory_avg,

        "memory_max_mb":
            memory_max,

        "memory_avg_pct":
            memory_avg_pct,

        "memory_max_pct":
            memory_max_pct,

        # Experiment configuration
        "gpu_layers":
            configuration.get(
                "gpu_layers"
            ),

        "context_size":
            configuration.get(
                "context_size"
            ),

        "batch_size":
            configuration.get(
                "batch_size"
            ),

        "ubatch_size":
            configuration.get(
                "ubatch_size"
            ),

        "requested_output_tokens":
            configuration.get(
                "output_tokens"
            ),

        "temperature":
            configuration.get(
                "temperature"
            ),

        "seed":
            configuration.get(
                "seed"
            ),

        "reasoning_budget":
            configuration.get(
                "reasoning_budget"
            ),

        "run_directory":
            str(run_dir),
    }


# ============================================================
# DATA QUALITY REPORT
#
# The original script silently left cells blank when a metric
# couldn't be found (missing llama_metrics, missing GPU columns,
# a failed run, etc). That made the empty-column bug you hit
# invisible until you inspected the CSV by hand. Print a summary
# so any future breakage - or the two extreme outlier runs in
# your "small" condition - is impossible to miss.
# ============================================================

CORE_METRICS = [
    "prompt_tokens",
    "output_tokens",
    "execution_time_s",
    "prompt_eval_time_ms",
    "generation_time_ms",
    "tokens_per_second",
    "energy_j",
    "energy_j_per_output_token",
    "cpu_usage_avg_pct",
    "gpu_usage_avg_pct",
    "gpu_power_w",
    "memory_avg_mb",
]


def print_data_quality_report(df):

    print()
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    n = len(df)

    for column in CORE_METRICS:

        if column not in df.columns:
            continue

        missing = df[column].isna().sum()

        if missing > 0:
            print(
                f"  {column}: {missing}/{n} runs missing "
                f"({'ALL' if missing == n else 'partial'})"
            )

    if "gpu_power_w" in df.columns and df["gpu_power_w"].isna().all():
        print(
            "\n  Note: gpu_power_w/gpu_energy_j/gpu_memory_mb are "
            "empty for every run because energy.csv has no "
            "GPU_POWER/GPU_MEMORY columns - EnergiBridge isn't "
            "exposing GPU power on this machine. This is a data-"
            "source limitation, not a bug in this script."
        )

    # Flag execution-time outliers per condition (>2x the median of
    # that prompt_size group) - this is exactly what happened with
    # runs 9 and 10 of your "small" condition.
    outlier_rows = []

    for prompt_size, group in df.groupby("prompt_size"):

        median_time = group["execution_time_s"].median()

        if pd.isna(median_time) or median_time == 0:
            continue

        flagged = group[
            group["execution_time_s"] > 2 * median_time
        ]

        for _, row in flagged.iterrows():
            outlier_rows.append(
                (
                    prompt_size,
                    row["run"],
                    row["execution_time_s"],
                    median_time,
                )
            )

    if outlier_rows:

        print(
            "\n  Possible outlier runs (execution_time_s > 2x the "
            "median for that prompt size):"
        )

        for prompt_size, run, exec_time, median_time in outlier_rows:
            print(
                f"    prompt_size={prompt_size} run={run}: "
                f"{exec_time:.1f}s vs median {median_time:.1f}s"
            )

        print(
            "  Consider investigating (thermal throttling, system "
            "sleep, a stalled cooldown) before trusting the "
            "mean/stddev for that condition."
        )

    print("=" * 70)


# ============================================================
# SUMMARY STATISTICS
#
# The benchmark spec asks for mean, standard deviation, and
# variation statistics per prompt-size condition. This was never
# actually computed anywhere in the pipeline - only the raw
# per-run rows were written out. Added here.
# ============================================================

STATS_METRICS = [
    "execution_time_s",
    "prompt_eval_time_ms",
    "generation_time_ms",
    "tokens_per_second",
    "energy_j",
    "energy_j_per_output_token",
    "cpu_usage_avg_pct",
    "cpu_usage_max_pct",
    "gpu_usage_avg_pct",
    "gpu_usage_max_pct",
    "memory_avg_mb",
    "memory_max_mb",
]

PROMPT_ORDER = {"small": 0, "medium": 1, "large": 2}


def compute_summary_stats(df):

    rows = []

    for prompt_size, group in df.groupby("prompt_size"):

        row = {
            "prompt_size": prompt_size,
            "n_runs": len(group),
        }

        for metric in STATS_METRICS:

            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:

                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_cv_pct"] = None

                continue

            mean = float(values.mean())

            std = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )

            cv_pct = (
                (std / mean * 100.0)
                if mean not in (0, None)
                else None
            )

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_cv_pct"] = cv_pct

        rows.append(row)

    stats_df = pd.DataFrame(rows)

    stats_df["_order"] = (
        stats_df["prompt_size"]
        .map(PROMPT_ORDER)
        .fillna(99)
    )

    stats_df = stats_df.sort_values(
        "_order"
    ).drop(columns=["_order"])

    return stats_df


# ============================================================
# GRAPHS
#
# The 7 comparisons requested in the benchmark spec. Each is a
# bar chart of the mean per prompt-size condition, with error
# bars showing +/- 1 standard deviation, ordered small/medium/
# large so prompt length increases left to right.
# ============================================================

GRAPH_SPECS = [
    (
        "energy_vs_prompt_length",
        "energy_j",
        "Total Energy Consumption vs. Prompt Length",
        "Energy (J)",
    ),
    (
        "energy_per_token_vs_prompt_length",
        "energy_j_per_output_token",
        "Energy per Output Token vs. Prompt Length",
        "Energy (J/token)",
    ),
    (
        "execution_time_vs_prompt_length",
        "execution_time_s",
        "Execution Time vs. Prompt Length",
        "Time (s)",
    ),
    (
        "cpu_utilization_vs_prompt_length",
        "cpu_usage_avg_pct",
        "Average CPU Utilization vs. Prompt Length",
        "CPU Usage (%)",
    ),
    (
        "gpu_utilization_vs_prompt_length",
        "gpu_usage_avg_pct",
        "Average GPU Utilization vs. Prompt Length",
        "GPU Usage (%)",
    ),
    (
        "memory_usage_vs_prompt_length",
        "memory_avg_mb",
        "Average Memory Usage vs. Prompt Length",
        "Memory (MB)",
    ),
    (
        "throughput_vs_prompt_length",
        "tokens_per_second",
        "Generation Throughput vs. Prompt Length",
        "Tokens/second",
    ),
]


def generate_plots(stats_df, plots_dir):

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nmatplotlib not installed - skipping graph generation. "
            "Install it with: pip install matplotlib"
        )
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)

    order = [
        p for p in ["small", "medium", "large"]
        if p in stats_df["prompt_size"].values
    ]

    written = []

    for filename, metric, title, ylabel in GRAPH_SPECS:

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in stats_df.columns:
            print(f"  Skipping {filename}: no data for '{metric}'")
            continue

        plot_df = stats_df.set_index("prompt_size").loc[order]

        if plot_df[mean_col].isna().all():
            print(f"  Skipping {filename}: '{metric}' is empty for every run")
            continue

        fig, ax = plt.subplots(figsize=(6, 4.5))

        means = plot_df[mean_col].astype(float)
        stds = plot_df[std_col].astype(float).fillna(0.0)

        ax.bar(
            order,
            means,
            yerr=stds,
            capsize=5,
            color="#4C72B0",
        )

        ax.set_title(title)
        ax.set_xlabel("Prompt size")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()

        out_path = plots_dir / f"{filename}.png"

        fig.savefig(out_path, dpi=150)

        plt.close(fig)

        written.append(out_path)

    return written


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/raw",
    )

    parser.add_argument(
        "--output",
        default=
        "results/consolidated/"
        "local_results.csv",
    )

    parser.add_argument(
        "--stats-output",
        default=
        "results/consolidated/"
        "summary_statistics.csv",
    )

    parser.add_argument(
        "--plots-dir",
        default=
        "results/consolidated/plots",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating the comparison graphs.",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input
    ).expanduser().resolve()

    output_file = Path(
        args.output
    ).expanduser().resolve()

    stats_file = Path(
        args.stats_output
    ).expanduser().resolve()

    plots_dir = Path(
        args.plots_dir
    ).expanduser().resolve()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Searching:\n{input_dir}"
    )

    rows = []

    for metadata_file in sorted(
        input_dir.rglob(
            "metadata.json"
        )
    ):

        run_dir = (
            metadata_file.parent
        )

        row = process_run(
            run_dir
        )

        if row is not None:

            rows.append(row)

    if not rows:

        print(
            "No valid benchmark runs found."
        )

        return

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df["_prompt_order"] = (
        df["prompt_size"]
        .map(PROMPT_ORDER)
        .fillna(99)
    )

    df = df.sort_values(
        [
            "model",
            "quantization",
            "_prompt_order",
            "run",
        ]
    )

    df = df.drop(
        columns=[
            "_prompt_order"
        ]
    )

    # --------------------------------------------------------
    # Save per-run CSV
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "CONSOLIDATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Runs: {len(df)}"
    )

    print(
        f"Per-run output:\n{output_file}"
    )

    # --------------------------------------------------------
    # Summary statistics per prompt-size condition
    # --------------------------------------------------------

    stats_df = compute_summary_stats(df)

    stats_df.to_csv(
        stats_file,
        index=False,
    )

    print(
        f"\nSummary statistics:\n{stats_file}"
    )

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    if not args.no_plots:

        print(
            f"\nGenerating comparison graphs in:\n{plots_dir}"
        )

        written = generate_plots(stats_df, plots_dir)

        for path in written:
            print(f"  {path.name}")

    # --------------------------------------------------------
    # Data quality report
    # --------------------------------------------------------

    print_data_quality_report(df)

    print(
        "\nColumns in per-run CSV:"
    )

    for column in df.columns:

        print(
            f"  {column}"
        )


if __name__ == "__main__":

    main()