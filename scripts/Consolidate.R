# ============================================================
# RQ1 ANALYSIS
# Prompt length vs energy efficiency of LLM inference
#
# Robust to:
#   - multiple models
#   - multiple quantization levels
#   - on-device only (current data)
#   - on-device + remote (future data)
#   - repeated runs
#   - missing optional metrics
#
# Main RQ1 variable:
#   prompt_tokens = actual prompt token count from llama-server
#
# Main outcome:
#   energy_j_per_output_token = total energy / generated output tokens
# ============================================================

required_packages <- c("jsonlite", "readr", "dplyr", "purrr", "ggplot2", "tidyr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Install missing packages first: ", paste(missing_packages, collapse = ", "))
}

library(jsonlite)
library(readr)
library(dplyr)
library(purrr)
library(ggplot2)
library(tidyr)

INPUT_DIR <- "results/raw"
OUTPUT_DIR <- "results/consolidated"
PLOTS_DIR <- file.path(OUTPUT_DIR, "plots")

# Set TRUE only if you want to exclude obviously invalid runs.
# The script never silently removes runs: exclusions are written to a CSV.
APPLY_BASIC_VALIDITY_FILTER <- TRUE

# ============================================================
# Helpers
# ============================================================

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

safe_num <- function(x) {
  if (is.null(x)) return(NA_real_)
  y <- unlist(x, use.names = FALSE)
  if (length(y) == 0) return(NA_real_)
  suppressWarnings(as.numeric(y[1]))
}

safe_chr <- function(x) {
  if (is.null(x)) return(NA_character_)
  y <- unlist(x, use.names = FALSE)
  if (length(y) == 0) return(NA_character_)
  as.character(y[1])
}

get_nested <- function(x, ...) {
  keys <- list(...)
  out <- x
  for (k in keys) {
    if (is.null(out) || is.null(out[[k]])) return(NULL)
    out <- out[[k]]
  }
  out
}

read_json_safe <- function(path) {
  tryCatch(
    fromJSON(path, simplifyVector = FALSE),
    error = function(e) {
      warning("Could not read JSON: ", path, " | ", conditionMessage(e))
      NULL
    }
  )
}

find_col <- function(df, candidates) {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) == 0) return(NULL)
  hit[1]
}

calculate_energy <- function(path) {
  if (!file.exists(path)) return(NA_real_)

  e <- tryCatch(
    read_csv(path, show_col_types = FALSE, progress = FALSE),
    error = function(err) NULL
  )

  if (is.null(e) || nrow(e) == 0) return(NA_real_)

  power_col <- find_col(e, c("SYSTEM_POWER (Watts)", "SYSTEM_POWER", "system_power"))
  delta_col <- find_col(e, c("Delta", "delta", "DELTA"))

  if (is.null(power_col) || is.null(delta_col)) return(NA_real_)

  power <- suppressWarnings(as.numeric(e[[power_col]]))
  delta_ms <- suppressWarnings(as.numeric(e[[delta_col]]))

  valid <- is.finite(power) & is.finite(delta_ms) & delta_ms >= 0
  if (!any(valid)) return(NA_real_)

  # E = P * dt, with power in W and Delta in ms.
  sum(power[valid] * delta_ms[valid] / 1000)
}

# ============================================================
# Read every run
# ============================================================

metadata_files <- list.files(
  INPUT_DIR,
  pattern = "^metadata\\.json$",
  recursive = TRUE,
  full.names = TRUE
)

if (length(metadata_files) == 0) {
  stop("No metadata.json files found under: ", INPUT_DIR)
}

process_run <- function(metadata_path) {
  meta <- read_json_safe(metadata_path)
  if (is.null(meta)) return(NULL)

  run_dir <- dirname(metadata_path)
  lm <- meta$llama_metrics
  cfg <- meta$configuration

  # -------------------------
  # Core identifiers
  # -------------------------
  model <- safe_chr(meta$model)
  quantization <- safe_chr(meta$quantization)
  architecture <- safe_chr(meta$architecture)
  prompt_size <- safe_chr(meta$prompt_size)
  run <- safe_num(meta$run)

  deployment <- safe_chr(get_nested(cfg, "deployment"))
  if (is.na(deployment)) deployment <- safe_chr(meta$deployment)
  if (is.na(deployment)) deployment <- "unknown"

  runtime <- safe_chr(get_nested(cfg, "runtime"))

  # A model label that never merges different quantizations accidentally.
  model_label <- model
  if (!is.na(quantization) && nzchar(quantization) &&
      (is.na(model) || !grepl(quantization, model, fixed = TRUE, ignore.case = TRUE))) {
    model_label <- paste0(model, " (", quantization, ")")
  }
  if (is.na(model_label) || !nzchar(model_label)) model_label <- "unknown model"

  # -------------------------
  # Tokens
  # -------------------------
  prompt_tokens <- safe_num(get_nested(lm, "prompt_tokens"))
  output_tokens <- safe_num(get_nested(lm, "output_tokens"))

  # Fallback to client_result.json when needed.
  client_path <- file.path(run_dir, "client_result.json")
  if ((is.na(prompt_tokens) || is.na(output_tokens)) && file.exists(client_path)) {
    client <- read_json_safe(client_path)
    usage <- get_nested(client, "response", "usage")
    if (is.na(prompt_tokens)) prompt_tokens <- safe_num(get_nested(usage, "prompt_tokens"))
    if (is.na(output_tokens)) output_tokens <- safe_num(get_nested(usage, "completion_tokens"))
  }

  # -------------------------
  # Timing
  # -------------------------
  prompt_eval_time_ms <- safe_num(get_nested(lm, "prompt_eval_time_ms"))
  generation_time_ms <- safe_num(get_nested(lm, "generation_time_ms"))
  generation_time_s <- ifelse(is.finite(generation_time_ms), generation_time_ms / 1000, NA_real_)

  request_wall_time_s <- safe_num(get_nested(lm, "request_wall_time_s"))
  if (!is.finite(request_wall_time_s)) request_wall_time_s <- safe_num(meta$wall_clock_time_s)

  # -------------------------
  # Energy
  # -------------------------
  energy_j <- calculate_energy(file.path(run_dir, "energy.csv"))

  energy_j_per_output_token <- if (
    is.finite(energy_j) && energy_j > 0 && is.finite(output_tokens) && output_tokens > 0
  ) energy_j / output_tokens else NA_real_

  output_tokens_per_joule <- if (
    is.finite(energy_j) && energy_j > 0 && is.finite(output_tokens) && output_tokens > 0
  ) output_tokens / energy_j else NA_real_

  # Generation throughput, using generated tokens / generation time.
  generation_tokens_per_second <- safe_num(get_nested(lm, "generation_tokens_per_second"))
  if (!is.finite(generation_tokens_per_second) &&
      is.finite(output_tokens) && output_tokens > 0 && is.finite(generation_time_s) && generation_time_s > 0) {
    generation_tokens_per_second <- output_tokens / generation_time_s
  }

  tibble(
    run = run,
    deployment = deployment,
    model = model,
    model_label = model_label,
    architecture = architecture,
    quantization = quantization,
    prompt_size = prompt_size,
    prompt_tokens = prompt_tokens,
    output_tokens = output_tokens,
    energy_j = energy_j,
    energy_j_per_output_token = energy_j_per_output_token,
    output_tokens_per_joule = output_tokens_per_joule,
    prompt_eval_time_ms = prompt_eval_time_ms,
    generation_time_s = generation_time_s,
    request_wall_time_s = request_wall_time_s,
    tokens_per_second = generation_tokens_per_second,
    runtime = runtime,
    source_dir = run_dir
  )
}

run_table <- map_dfr(metadata_files, process_run)

if (nrow(run_table) == 0) stop("No readable runs were found.")

# ============================================================
# Data-quality checks
# ============================================================

run_table <- run_table %>%
  mutate(
    valid_prompt = is.finite(prompt_tokens) & prompt_tokens > 0,
    valid_output = is.finite(output_tokens) & output_tokens > 0,
    valid_energy = is.finite(energy_j) & energy_j > 0,
    valid_efficiency = is.finite(energy_j_per_output_token) & energy_j_per_output_token > 0,
    valid_generation_time = is.finite(generation_time_s) & generation_time_s > 0
  ) %>%
  mutate(
    analysis_valid = valid_prompt & valid_output & valid_energy & valid_efficiency
  )

quality_table <- run_table %>%
  transmute(
    source_dir,
    run,
    deployment,
    model_label,
    prompt_size,
    prompt_tokens,
    output_tokens,
    energy_j,
    analysis_valid,
    reason = case_when(
      !valid_prompt ~ "Invalid or missing prompt_tokens",
      !valid_output ~ "Invalid or missing output_tokens",
      !valid_energy ~ "Invalid or missing energy_j",
      !valid_efficiency ~ "Invalid energy efficiency",
      TRUE ~ "valid"
    )
  )

write_csv(quality_table, file.path(OUTPUT_DIR, "data_quality.csv"), na = "")

# Never silently discard invalid runs unless explicitly enabled.
if (APPLY_BASIC_VALIDITY_FILTER) {
  analysis_df <- run_table %>% filter(analysis_valid)
} else {
  analysis_df <- run_table
}

if (nrow(analysis_df) == 0) stop("No valid runs remain for analysis. Check data_quality.csv.")

# ============================================================
# Main per-run CSV
# ============================================================

run_table_out <- analysis_df %>%
  select(
    run,
    deployment,
    model,
    model_label,
    architecture,
    quantization,
    prompt_size,
    prompt_tokens,
    output_tokens,
    energy_j,
    energy_j_per_output_token,
    output_tokens_per_joule,
    prompt_eval_time_ms,
    generation_time_s,
    request_wall_time_s,
    tokens_per_second,
    runtime,
    source_dir
  ) %>%
  arrange(model_label, deployment, prompt_tokens, run)

write_csv(run_table_out, file.path(OUTPUT_DIR, "run_table.csv"), na = "")

# ============================================================
# RQ1 summary
#
# Grouping includes model + quantization + deployment so that
# different models are NEVER incorrectly averaged together.
# ============================================================

summary_group <- c("model_label", "deployment", "prompt_size")

RQ1_summary <- run_table_out %>%
  group_by(across(all_of(summary_group))) %>%
  summarise(
    n_runs = n(),
    mean_prompt_tokens = mean(prompt_tokens, na.rm = TRUE),
    sd_prompt_tokens = sd(prompt_tokens, na.rm = TRUE),
    mean_energy_j = mean(energy_j, na.rm = TRUE),
    sd_energy_j = sd(energy_j, na.rm = TRUE),
    mean_energy_j_per_output_token = mean(energy_j_per_output_token, na.rm = TRUE),
    sd_energy_j_per_output_token = sd(energy_j_per_output_token, na.rm = TRUE),
    mean_output_tokens_per_joule = mean(output_tokens_per_joule, na.rm = TRUE),
    sd_output_tokens_per_joule = sd(output_tokens_per_joule, na.rm = TRUE),
    mean_generation_time_s = mean(generation_time_s, na.rm = TRUE),
    sd_generation_time_s = sd(generation_time_s, na.rm = TRUE),
    mean_tokens_per_second = mean(tokens_per_second, na.rm = TRUE),
    sd_tokens_per_second = sd(tokens_per_second, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(RQ1_summary, file.path(OUTPUT_DIR, "RQ1_summary.csv"), na = "")

# ============================================================
# Optional model/quantization summary
# Useful because the RQ concerns prompt length, while model and
# quantization are experimental factors that should not be hidden.
# ============================================================

model_summary <- run_table_out %>%
  group_by(model_label, deployment) %>%
  summarise(
    n_runs = n(),
    mean_energy_j_per_output_token = mean(energy_j_per_output_token, na.rm = TRUE),
    sd_energy_j_per_output_token = sd(energy_j_per_output_token, na.rm = TRUE),
    mean_output_tokens_per_joule = mean(output_tokens_per_joule, na.rm = TRUE),
    mean_tokens_per_second = mean(tokens_per_second, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(model_summary, file.path(OUTPUT_DIR, "model_summary.csv"), na = "")

# ============================================================
# Plot helpers
# ============================================================

save_plot <- function(plot, filename, width = 9, height = 6) {
  ggsave(
    filename = file.path(PLOTS_DIR, filename),
    plot = plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )
}

plot_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

n_models <- n_distinct(analysis_df$model_label)
n_quant <- n_distinct(na.omit(analysis_df$quantization))
n_deployments <- n_distinct(analysis_df$deployment)

# Use model+quantization as the comparison group. If only one model
# exists, this remains a clean single-series plot.
analysis_df <- analysis_df %>%
  mutate(
    model_quant = ifelse(
      !is.na(quantization) & nzchar(quantization) & !grepl(quantization, model, fixed = TRUE, ignore.case = TRUE),
      paste0(model, " (", quantization, ")"),
      model_label
    )
  )

# ============================================================
# 1. Main relationship: prompt length vs total energy
# ============================================================

p1 <- ggplot(analysis_df, aes(x = prompt_tokens, y = energy_j)) +
  geom_point(aes(color = model_quant), size = 2.7, alpha = 0.8) +
  geom_smooth(aes(color = model_quant), method = "lm", se = FALSE, linewidth = 0.9) +
  labs(
    title = "Prompt Length vs Total Energy",
    subtitle = "Each point represents one inference run",
    x = "Prompt length (tokens)",
    y = "Total energy (J)",
    color = "Model / quantization"
  ) +
  plot_theme

save_plot(p1, "01_prompt_length_vs_total_energy.png")

# ============================================================
# 2. PRIMARY RQ1 PLOT: prompt length vs energy efficiency
# ============================================================

p2 <- ggplot(analysis_df, aes(x = prompt_tokens, y = energy_j_per_output_token)) +
  geom_point(aes(color = model_quant), size = 2.9, alpha = 0.85) +
  geom_smooth(aes(color = model_quant), method = "lm", se = TRUE, linewidth = 0.9) +
  labs(
    title = "Prompt Length vs Energy Efficiency",
    subtitle = "Primary RQ1 outcome: lower J/token means better energy efficiency",
    x = "Prompt length (tokens)",
    y = "Energy per output token (J/token)",
    color = "Model / quantization"
  ) +
  plot_theme

save_plot(p2, "02_PRIMARY_prompt_length_vs_energy_efficiency.png")

# ============================================================
# 3. Same efficiency result as tokens/J
#    Useful because higher = better, which is intuitive.
# ============================================================

p3 <- ggplot(analysis_df, aes(x = prompt_tokens, y = output_tokens_per_joule)) +
  geom_point(aes(color = model_quant), size = 2.9, alpha = 0.85) +
  geom_smooth(aes(color = model_quant), method = "lm", se = TRUE, linewidth = 0.9) +
  labs(
    title = "Prompt Length vs Output Efficiency",
    subtitle = "Higher output tokens per joule indicates better energy efficiency",
    x = "Prompt length (tokens)",
    y = "Output tokens per joule (tokens/J)",
    color = "Model / quantization"
  ) +
  plot_theme

save_plot(p3, "03_prompt_length_vs_tokens_per_joule.png")

# ============================================================
# 4. Supporting performance plot
# ============================================================

p4 <- ggplot(analysis_df, aes(x = prompt_tokens, y = generation_time_s)) +
  geom_point(aes(color = model_quant), size = 2.7, alpha = 0.8) +
  geom_smooth(aes(color = model_quant), method = "lm", se = FALSE, linewidth = 0.9) +
  labs(
    title = "Prompt Length vs Generation Time",
    subtitle = "Supporting metric for interpreting energy differences",
    x = "Prompt length (tokens)",
    y = "Generation time (s)",
    color = "Model / quantization"
  ) +
  plot_theme

save_plot(p4, "04_prompt_length_vs_generation_time.png")

# ============================================================
# 5. Supporting throughput plot
# ============================================================

p5 <- ggplot(analysis_df, aes(x = prompt_tokens, y = tokens_per_second)) +
  geom_point(aes(color = model_quant), size = 2.7, alpha = 0.8) +
  geom_smooth(aes(color = model_quant), method = "lm", se = FALSE, linewidth = 0.9) +
  labs(
    title = "Prompt Length vs Generation Throughput",
    subtitle = "Supporting metric: generated tokens per second",
    x = "Prompt length (tokens)",
    y = "Generation throughput (tokens/s)",
    color = "Model / quantization"
  ) +
  plot_theme

save_plot(p5, "05_prompt_length_vs_throughput.png")

# ============================================================
# 6. Condition-level plot
# Useful when small/medium/large prompts were deliberately tested.
# Error bars show +/- 1 SD across repeated runs.
# ============================================================

condition_df <- analysis_df %>%
  group_by(model_quant, deployment, prompt_size) %>%
  summarise(
    mean_prompt_tokens = mean(prompt_tokens, na.rm = TRUE),
    mean_energy_efficiency = mean(energy_j_per_output_token, na.rm = TRUE),
    sd_energy_efficiency = sd(energy_j_per_output_token, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    sd_energy_efficiency = ifelse(is.na(sd_energy_efficiency), 0, sd_energy_efficiency)
  )

if (n_distinct(condition_df$prompt_size) > 1) {
  p6 <- ggplot(
    condition_df,
    aes(x = mean_prompt_tokens, y = mean_energy_efficiency, color = model_quant, group = model_quant)
  ) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(
        ymin = pmax(0, mean_energy_efficiency - sd_energy_efficiency),
        ymax = mean_energy_efficiency + sd_energy_efficiency
      ),
      width = 0
    ) +
    labs(
      title = "Energy Efficiency Across Prompt-Length Conditions",
      subtitle = "Mean ± 1 SD across repeated runs",
      x = "Mean prompt length (tokens)",
      y = "Energy per output token (J/token)",
      color = "Model / quantization"
    ) +
    plot_theme

  save_plot(p6, "06_prompt_condition_energy_efficiency.png")
}

# ============================================================
# 7. Deployment comparison — ONLY after remote data exist.
# This block creates no fake remote comparison for the current
# on-device-only dataset.
# ============================================================

if (n_deployments >= 2) {
  deployment_df <- analysis_df %>%
    group_by(deployment, model_quant, prompt_size) %>%
    summarise(
      mean_prompt_tokens = mean(prompt_tokens, na.rm = TRUE),
      mean_energy_efficiency = mean(energy_j_per_output_token, na.rm = TRUE),
      sd_energy_efficiency = sd(energy_j_per_output_token, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(sd_energy_efficiency = ifelse(is.na(sd_energy_efficiency), 0, sd_energy_efficiency))

  p7 <- ggplot(
    deployment_df,
    aes(x = mean_prompt_tokens, y = mean_energy_efficiency, color = deployment, group = deployment)
  ) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(
        ymin = pmax(0, mean_energy_efficiency - sd_energy_efficiency),
        ymax = mean_energy_efficiency + sd_energy_efficiency
      ),
      width = 0
    ) +
    facet_wrap(~ model_quant) +
    labs(
      title = "On-Device vs Remote Energy Efficiency",
      subtitle = "Generated only when both deployment types are present",
      x = "Mean prompt length (tokens)",
      y = "Energy per output token (J/token)",
      color = "Deployment"
    ) +
    plot_theme

  save_plot(p7, "07_deployment_energy_efficiency.png", width = 10, height = 6.5)
}

# ============================================================
# 8. Model comparison summary — only if multiple model groups exist.
# This is not the primary RQ plot, but prevents the second model
# from being hidden in the analysis.
# ============================================================

if (n_models > 1) {
  model_plot_df <- analysis_df %>%
    group_by(model_quant, prompt_size) %>%
    summarise(
      mean_prompt_tokens = mean(prompt_tokens, na.rm = TRUE),
      mean_energy_efficiency = mean(energy_j_per_output_token, na.rm = TRUE),
      sd_energy_efficiency = sd(energy_j_per_output_token, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(sd_energy_efficiency = ifelse(is.na(sd_energy_efficiency), 0, sd_energy_efficiency))

  p8 <- ggplot(
    model_plot_df,
    aes(x = mean_prompt_tokens, y = mean_energy_efficiency, group = model_quant, color = model_quant)
  ) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(
        ymin = pmax(0, mean_energy_efficiency - sd_energy_efficiency),
        ymax = mean_energy_efficiency + sd_energy_efficiency
      ),
      width = 0
    ) +
    labs(
      title = "Energy Efficiency by Model Across Prompt Lengths",
      subtitle = "Model/quantization differences are shown without combining models",
      x = "Mean prompt length (tokens)",
      y = "Energy per output token (J/token)",
      color = "Model / quantization"
    ) +
    plot_theme

  save_plot(p8, "08_model_energy_efficiency_comparison.png", width = 10, height = 6.5)
}

# ============================================================
# Final console report
# ============================================================

cat("\n===============================================\n")
cat("RQ1 ANALYSIS COMPLETE\n")
cat("===============================================\n")
cat("Raw metadata files found: ", length(metadata_files), "\n", sep = "")
cat("Runs available for analysis: ", nrow(analysis_df), "\n", sep = "")
cat("Models: ", paste(unique(analysis_df$model_label), collapse = ", "), "\n", sep = "")
cat("Deployments: ", paste(unique(analysis_df$deployment), collapse = ", "), "\n", sep = "")
cat("Prompt sizes: ", paste(unique(analysis_df$prompt_size), collapse = ", "), "\n", sep = "")
cat("Output directory: ", normalizePath(OUTPUT_DIR, mustWork = FALSE), "\n", sep = "")
cat("Plots directory: ", normalizePath(PLOTS_DIR, mustWork = FALSE), "\n", sep = "")
cat("\nMain RQ1 plot: 02_PRIMARY_prompt_length_vs_energy_efficiency.png\n")
if (n_deployments < 2) {
  cat("Remote comparison: NOT generated because fewer than two deployment types are present.\n")
}
