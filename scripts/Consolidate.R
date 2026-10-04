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
      (is.na(model) || !grepl(tolower(quantization), tolower(model), fixed = TRUE))) {
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

save_plot <- function(plot, filename, width = 10, height = 7) {
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
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11),
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.1, "lines")
  )

# ============================================================
# LOCAL-ONLY PLOTS
# ============================================================
# The current stage of the project has only on-device data.
# Remote results are intentionally excluded from these figures.
# They can be added later as a separate deployment comparison.
# ============================================================

local_df <- analysis_df %>%
  filter(tolower(deployment) %in% c("on_device", "on-device", "on device", "local"))

if (nrow(local_df) == 0) {
  stop("No on-device/local runs found. Check the deployment field in run_table.csv.")
}

# Clean model/quantization labels.
# model_quant = unique comparison group.
# model_family = model name with a trailing quantization suffix removed,
# when the metadata model field already contains the quantization name.
local_df <- local_df %>%
  mutate(
    model_quant = ifelse(
      !is.na(quantization) & nzchar(quantization) &
        !grepl(quantization, model, ignore.case = TRUE, fixed = FALSE),
      paste0(model, " (", quantization, ")"),
      model_label
    ),
    model_quant = ifelse(
      is.na(model_quant) | !nzchar(model_quant), "Unknown model", model_quant
    ),
    model_family = model,
    prompt_condition = factor(
      prompt_size,
      levels = c("small", "medium", "large"),
      labels = c("Short", "Medium", "Long")
    )
  )

# If prompt_size contains another naming convention, keep it instead
# of producing missing facet labels.
if (all(is.na(local_df$prompt_condition))) {
  local_df$prompt_condition <- factor(local_df$prompt_size)
}

model_levels <- local_df %>%
  distinct(model_quant) %>%
  arrange(model_quant) %>%
  pull(model_quant)
local_df$model_quant <- factor(local_df$model_quant, levels = model_levels)

prompt_levels <- levels(droplevels(local_df$prompt_condition))

# ------------------------------------------------------------
# Condition summaries
# ------------------------------------------------------------
# Mean +/- 95% CI is used for the trend plots.
# Boxplots show the complete run-to-run distribution.
# ------------------------------------------------------------

condition_df <- local_df %>%
  group_by(model_quant, prompt_condition) %>%
  summarise(
    n = n(),
    mean_prompt_tokens = mean(prompt_tokens, na.rm = TRUE),
    mean_energy_j = mean(energy_j, na.rm = TRUE),
    mean_efficiency = mean(energy_j_per_output_token, na.rm = TRUE),
    sd_efficiency = sd(energy_j_per_output_token, na.rm = TRUE),
    mean_tokens_per_joule = mean(output_tokens_per_joule, na.rm = TRUE),
    mean_generation_time_s = mean(generation_time_s, na.rm = TRUE),
    mean_throughput = mean(tokens_per_second, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    sd_efficiency = ifelse(is.na(sd_efficiency), 0, sd_efficiency),
    se_efficiency = sd_efficiency / sqrt(pmax(n, 1)),
    ci95_efficiency = ifelse(
      n > 1,
      qt(0.975, df = n - 1) * se_efficiency,
      0
    ),
    lower95_efficiency = pmax(0, mean_efficiency - ci95_efficiency),
    upper95_efficiency = mean_efficiency + ci95_efficiency
  )

# ============================================================
# FIGURE 1 — PRIMARY: ENERGY EFFICIENCY BOX PLOTS
# ============================================================
# This is the clearest figure for the current RQ1 stage.
# Each panel = one prompt-length condition.
# Each box = one model/quantization configuration.
# Dots = individual runs.
# Diamond = mean.
# The reader can therefore see BOTH:
#   1. prompt-length effect
#   2. model/quantization effect at the same prompt length
# ============================================================

p1 <- ggplot(
  local_df,
  aes(x = model_quant, y = energy_j_per_output_token, fill = model_quant)
) +
  geom_boxplot(
    width = 0.62,
    alpha = 0.55,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    aes(color = model_quant),
    width = 0.13,
    height = 0,
    size = 2.0,
    alpha = 0.65
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 3.2,
    fill = "white",
    color = "black",
    stroke = 0.8
  ) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_x") +
  labs(
    title = "Energy Efficiency by Prompt Length and Model",
    subtitle = "Box = distribution of runs; dots = individual runs; diamond = mean",
    x = "Model / quantization",
    y = "Energy per output token (J/token)",
    fill = "Model / quantization",
    color = "Model / quantization"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    legend.position = "none"
  )

save_plot(p1, "01_PRIMARY_energy_efficiency_boxplots.png", width = 12, height = 6.8)

# ============================================================
# FIGURE 2 — MODEL EFFECT AT EACH PROMPT LENGTH
# ============================================================
# Same data, but arranged with MODEL on the x-axis and PROMPT
# LENGTH as facets. This is the strongest visual for the question:
# "Which model is more energy efficient at the same prompt length?"
# ============================================================

p2 <- ggplot(
  local_df,
  aes(x = model_quant, y = energy_j_per_output_token, fill = model_quant)
) +
  geom_boxplot(
    width = 0.62,
    alpha = 0.60,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    aes(color = model_quant),
    width = 0.13,
    height = 0,
    size = 2.1,
    alpha = 0.60
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 3.2,
    fill = "white",
    color = "black"
  ) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_x") +
  labs(
    title = "Model and Quantization Effect on Energy Efficiency",
    subtitle = "Direct comparison of configurations under the same prompt-length condition",
    x = "Model / quantization",
    y = "Energy per output token (J/token)"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

save_plot(p2, "02_MODEL_effect_energy_efficiency.png", width = 12, height = 6.8)

# ============================================================
# FIGURE 3 — TOTAL ENERGY BOX PLOTS
# ============================================================
# Supporting result. Unlike J/token, total energy also reflects the
# amount of generated output, so it is not the primary efficiency metric.
# ============================================================

p3 <- ggplot(
  local_df,
  aes(x = model_quant, y = energy_j, fill = model_quant)
) +
  geom_boxplot(
    width = 0.62,
    alpha = 0.55,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    aes(color = model_quant),
    width = 0.13,
    height = 0,
    size = 2.0,
    alpha = 0.65
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 3.0,
    fill = "white",
    color = "black"
  ) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_y") +
  labs(
    title = "Total Energy by Prompt Length and Model",
    subtitle = "Supporting measure; individual runs are retained",
    x = "Model / quantization",
    y = "Total energy (J)"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

save_plot(p3, "03_total_energy_boxplots.png", width = 12, height = 6.8)

# ============================================================
# FIGURE 4 — OUTPUT EFFICIENCY BOX PLOTS
# ============================================================
# Equivalent to J/token but with the intuitive direction:
# higher tokens/J = better.
# ============================================================

p4 <- ggplot(
  local_df,
  aes(x = model_quant, y = output_tokens_per_joule, fill = model_quant)
) +
  geom_boxplot(
    width = 0.62,
    alpha = 0.55,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    aes(color = model_quant),
    width = 0.13,
    height = 0,
    size = 2.0,
    alpha = 0.65
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 3.0,
    fill = "white",
    color = "black"
  ) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_y") +
  labs(
    title = "Output Efficiency by Prompt Length and Model",
    subtitle = "Higher tokens per joule indicates better energy efficiency",
    x = "Model / quantization",
    y = "Output tokens per joule (tokens/J)"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

save_plot(p4, "04_output_efficiency_boxplots.png", width = 12, height = 6.8)

# ============================================================
# FIGURE 5 — GENERATION TIME BOX PLOTS
# ============================================================

p5 <- ggplot(
  local_df,
  aes(x = model_quant, y = generation_time_s, fill = model_quant)
) +
  geom_boxplot(
    width = 0.62,
    alpha = 0.55,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    aes(color = model_quant),
    width = 0.13,
    height = 0,
    size = 2.0,
    alpha = 0.65
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 3.0,
    fill = "white",
    color = "black"
  ) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_y") +
  labs(
    title = "Generation Time by Prompt Length and Model",
    subtitle = "Supporting performance measure for interpreting energy differences",
    x = "Model / quantization",
    y = "Generation time (s)"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

save_plot(p5, "05_generation_time_boxplots.png", width = 12, height = 6.8)

# ============================================================
# FIGURE 6 — GENERATION THROUGHPUT BOX PLOTS
# ============================================================

p6 <- ggplot(
  local_df,
  aes(x = model_quant, y = tokens_per_second, fill = model_quant)
) +
  geom_boxplot(
    width = 0.62,
    alpha = 0.55,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_jitter(
    aes(color = model_quant),
    width = 0.13,
    height = 0,
    size = 2.0,
    alpha = 0.65
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 3.0,
    fill = "white",
    color = "black"
  ) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_y") +
  labs(
    title = "Generation Throughput by Prompt Length and Model",
    subtitle = "Supporting measure; higher tokens/s indicates faster generation",
    x = "Model / quantization",
    y = "Generation throughput (tokens/s)"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

save_plot(p6, "06_generation_throughput_boxplots.png", width = 12, height = 6.8)

# ============================================================
# FIGURE 7 — PROMPT-LENGTH TREND WITH MEAN +/- 95% CI
# ============================================================
# This is the complementary figure to the boxplots.
# It makes the prompt-length trend easy to see for every model.
# No regression line is used because there are only a few deliberate
# prompt-length conditions; connecting the experimental means is clearer.
# ============================================================

p7 <- ggplot(
  condition_df,
  aes(
    x = mean_prompt_tokens,
    y = mean_efficiency,
    color = model_quant,
    group = model_quant
  )
) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 3.4) +
  geom_errorbar(
    aes(ymin = lower95_efficiency, ymax = upper95_efficiency),
    width = 0,
    linewidth = 0.7
  ) +
  labs(
    title = "Energy Efficiency Trend Across Prompt Lengths",
    subtitle = "Mean ± 95% CI; each line represents one model/quantization configuration",
    x = "Mean prompt length (tokens)",
    y = "Energy per output token (J/token)",
    color = "Model / quantization"
  ) +
  plot_theme

save_plot(p7, "07_prompt_length_energy_efficiency_trend.png", width = 10.5, height = 6.8)

# ============================================================
# FIGURE 8 — MODEL COMPARISON AT EACH CONDITION (POINT-RANGE)
# ============================================================
# A compact "candle-like" summary: mean point + 95% CI for each
# model at each prompt condition. This is easier to read in a paper
# than a very dense boxplot when the number of repeated runs grows.
# ============================================================

p8 <- ggplot(
  condition_df,
  aes(x = model_quant, y = mean_efficiency, color = model_quant)
) +
  geom_errorbar(
    aes(ymin = lower95_efficiency, ymax = upper95_efficiency),
    width = 0.16,
    linewidth = 0.8
  ) +
  geom_point(size = 3.3) +
  facet_wrap(~ prompt_condition, nrow = 1, scales = "free_x") +
  labs(
    title = "Model Energy Efficiency at Each Prompt Length",
    subtitle = "Mean energy per output token with 95% confidence intervals",
    x = "Model / quantization",
    y = "Energy per output token (J/token)"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

save_plot(p8, "08_model_comparison_point_range.png", width = 12, height = 6.8)

# ============================================================
# NO REMOTE PLOTS AT THIS STAGE
# ============================================================
# Remote data can remain in the consolidated CSVs, but are not used
# in the current figures. A separate deployment comparison can be
# added once remote experiments are completed.
# ============================================================

write_csv(
  condition_df,
  file.path(OUTPUT_DIR, "RQ1_on_device_condition_summary.csv"),
  na = ""
)

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
