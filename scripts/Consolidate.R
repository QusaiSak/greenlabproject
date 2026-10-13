required_packages <- c("jsonlite", "readr", "dplyr", "purrr", "ggplot2", "tidyr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Install missing packages first: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(readr)
  library(dplyr)
  library(purrr)
  library(ggplot2)
  library(tidyr)
})

INPUT_DIR <- "results/raw"
CARBON_INTENSITY_FILE <- "data/carbon_intensity_nl.csv"
OUTPUT_DIR <- "results/consolidated"
PLOTS_DIR <- file.path(OUTPUT_DIR, "plots")

dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Helpers
# ============================================================

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

first_finite <- function(...) {
  for (v in list(...)) if (length(v) == 1 && is.finite(v)) return(v)
  NA_real_
}

read_json_safe <- function(path) {
  tryCatch(fromJSON(path, simplifyVector = FALSE), error = function(e) {
    warning("Could not read JSON: ", path, " | ", conditionMessage(e))
    NULL
  })
}

read_energy_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  e <- tryCatch(read_csv(path, show_col_types = FALSE, progress = FALSE), error = function(err) NULL)
  if (is.null(e) || nrow(e) == 0) return(NULL)
  e
}

num_col <- function(e, col) suppressWarnings(as.numeric(e[[col]]))

# Sum of positive increments of a cumulative counter (robust to counter resets).
counter_delta <- function(x) {
  d <- diff(x[is.finite(x)])
  sum(d[d > 0])
}

# Mac (EnergiBridge) energy: integrate SYSTEM_POWER (W) over Delta (ms).
mac_energy_j <- function(path) {
  e <- read_energy_csv(path)
  if (is.null(e) || !all(c("SYSTEM_POWER (Watts)", "Delta") %in% names(e))) return(NA_real_)
  p <- num_col(e, "SYSTEM_POWER (Watts)")
  dt <- num_col(e, "Delta")
  ok <- is.finite(p) & is.finite(dt) & dt >= 0
  if (!any(ok)) return(NA_real_)
  sum(p[ok] * dt[ok] / 1000)
}

# Remote server (Linux + NVIDIA) energy, split into components.
server_energy_parts <- function(path) {
  out <- c(cpu = NA_real_, dram = NA_real_, gpu = NA_real_)
  e <- read_energy_csv(path)
  if (is.null(e)) return(out)
  if ("PACKAGE_ENERGY (J)" %in% names(e)) out["cpu"] <- counter_delta(num_col(e, "PACKAGE_ENERGY (J)"))
  if ("DRAM_ENERGY (J)" %in% names(e)) out["dram"] <- counter_delta(num_col(e, "DRAM_ENERGY (J)"))
  if (all(c("GPU0_POWER (mWatts)", "Delta") %in% names(e))) {
    p <- num_col(e, "GPU0_POWER (mWatts)")
    dt <- num_col(e, "Delta")
    ok <- is.finite(p) & is.finite(dt) & dt >= 0
    out["gpu"] <- sum(p[ok] * dt[ok]) / 1e6
  }
  out
}

# Run start time (Unix seconds) from the first EnergiBridge sample.
energy_start_s <- function(path) {
  e <- read_energy_csv(path)
  if (is.null(e) || !"Time" %in% names(e)) return(NA_real_)
  t <- num_col(e, "Time")
  if (!any(is.finite(t))) return(NA_real_)
  min(t, na.rm = TRUE) / 1000
}

# ============================================================
# Read every run
# ============================================================

metadata_files <- list.files(INPUT_DIR, pattern = "^metadata\\.json$", recursive = TRUE, full.names = TRUE)
if (length(metadata_files) == 0) stop("No metadata.json files found under: ", INPUT_DIR)

process_run <- function(metadata_path) {
  meta <- read_json_safe(metadata_path)
  if (is.null(meta)) return(NULL)

  run_dir <- dirname(metadata_path)
  lm <- meta$llama_metrics
  cfg <- meta$configuration
  sm <- meta$system_metrics
  net <- meta$network

  deployment <- safe_chr(cfg$deployment)
  if (is.na(deployment)) deployment <- safe_chr(meta$deployment)
  deployment <- ifelse(grepl("remote", tolower(deployment)), "remote", "on-device")

  output_tokens <- first_finite(safe_num(lm$output_tokens), safe_num(meta$actual_output_tokens))
  prompt_tokens <- safe_num(lm$prompt_tokens)

  wall_time_s <- first_finite(safe_num(lm$request_wall_time_s), safe_num(meta$wall_clock_time_s))

  run_start_unix_s <- if (deployment == "on-device") {
    energy_start_s(file.path(run_dir, "energy.csv"))
  } else {
    first_finite(energy_start_s(file.path(run_dir, "client_energy.csv")), energy_start_s(file.path(run_dir, "energy.csv")))
  }

  if (deployment == "on-device") {
    client_energy_j <- mac_energy_j(file.path(run_dir, "energy.csv"))
    server <- c(cpu = NA_real_, dram = NA_real_, gpu = NA_real_)
    server_energy_j <- NA_real_
    total_energy_j <- client_energy_j
    network_condition <- "local"
    delay_ms <- NA_real_
  } else {
    client_energy_j <- first_finite(mac_energy_j(file.path(run_dir, "client_energy.csv")), safe_num(meta$client_energy_j))
    server <- server_energy_parts(file.path(run_dir, "energy.csv"))
    server_energy_j <- if (all(is.na(server))) NA_real_ else sum(server, na.rm = TRUE)
    total_energy_j <- client_energy_j + server_energy_j
    network_condition <- safe_chr(net$condition)
    delay_ms <- first_finite(safe_num(net$configured_delay_ms), 0)
  }

  tibble(
    deployment = deployment,
    network_condition = network_condition,
    configured_delay_ms = delay_ms,
    network_rtt_ms = safe_num(net$network_rtt_ms),
    network_total_bytes = safe_num(net$total_bytes),
    model = safe_chr(meta$model),
    architecture = safe_chr(meta$architecture),
    quantization = safe_chr(meta$quantization),
    prompt_size = safe_chr(meta$prompt_size),
    run = safe_num(meta$run),
    schedule_index = safe_num(meta$schedule_index),
    run_start_unix_s = run_start_unix_s,
    prompt_tokens = prompt_tokens,
    output_tokens = output_tokens,
    finish_reason = safe_chr(lm$finish_reason),
    wall_time_s = wall_time_s,
    prompt_eval_time_s = safe_num(lm$prompt_eval_time_ms) / 1000,
    generation_time_s = safe_num(lm$generation_time_ms) / 1000,
    tokens_per_second = first_finite(
      safe_num(lm$generation_tokens_per_second),
      if (is.finite(output_tokens) && is.finite(wall_time_s)) output_tokens / wall_time_s else NA_real_
    ),
    client_energy_j = client_energy_j,
    server_cpu_energy_j = unname(server["cpu"]),
    server_dram_energy_j = unname(server["dram"]),
    server_gpu_energy_j = unname(server["gpu"]),
    server_energy_j = server_energy_j,
    total_energy_j = total_energy_j,
    cpu_usage_avg_pct = safe_num(sm$cpu_usage_avg_pct),
    gpu_usage_avg_pct = safe_num(sm$gpu_usage_avg_pct),
    memory_avg_mb = safe_num(sm$memory_avg_mb),
    memory_max_mb = safe_num(sm$memory_max_mb),
    gpu_memory_mb = safe_num(sm$gpu_memory_mb),
    source_dir = run_dir
  )
}

runs <- map_dfr(metadata_files, process_run)
if (nrow(runs) == 0) stop("No readable runs were found.")

# ============================================================
# Derived variables + labels
# ============================================================

runs <- runs %>%
  mutate(
    energy_j_per_output_token = ifelse(output_tokens > 0, total_energy_j / output_tokens, NA_real_),
    client_j_per_output_token = ifelse(output_tokens > 0, client_energy_j / output_tokens, NA_real_),
    server_j_per_output_token = ifelse(output_tokens > 0, server_energy_j / output_tokens, NA_real_),
    avg_power_w = total_energy_j / wall_time_s,
    prompt_size = factor(prompt_size, levels = c("small", "medium", "large"), labels = c("Short", "Medium", "Long")),
    model_family = case_when(
      grepl("16B-A3B", model) ~ "Qwen3-16B-A3B (MoE)",
      grepl("Qwen3-8B", model) ~ "Qwen3-8B (Dense)",
      TRUE ~ model
    ),
    config = paste0(model_family, " ", quantization),
    setup = case_when(
      deployment == "on-device" ~ "On-device",
      configured_delay_ms > 0 ~ paste0("Remote +", configured_delay_ms, " ms"),
      TRUE ~ "Remote (baseline)"
    ),
    setup = factor(setup, levels = unique(c("On-device", "Remote (baseline)",
                                            paste0("Remote +", sort(unique(configured_delay_ms[configured_delay_ms > 0])), " ms"),
                                            sort(unique(setup))))),
    valid = is.finite(prompt_tokens) & prompt_tokens > 0 &
      is.finite(output_tokens) & output_tokens > 0 &
      is.finite(total_energy_j) & total_energy_j > 0
  )

config_levels <- c("Qwen3-8B (Dense) Q4_K_M", "Qwen3-8B (Dense) Q8_0", "Qwen3-16B-A3B (MoE) Q4_K_M")
runs$config <- factor(runs$config, levels = unique(c(config_levels, sort(unique(runs$config)))))
runs <- runs %>%
  mutate(config_short = case_when(
    grepl("16B-A3B", model) ~ paste("MoE", sub("_.*", "", quantization)),
    TRUE ~ paste("8B", sub("_.*", "", quantization))
  ))

# ============================================================
# Carbon footprint (Green Software Foundation SCI specification)
#   SCI = ((E * I) + M) per R
#   E = measured energy (kWh), I = grid carbon intensity at run time (gCO2e/kWh),
#   M = embodied emissions = TE * (TiR / EL) * (RR / ToR), R = one output token.
# ============================================================

HOURS_PER_YEAR <- 365 * 24

# Embodied footprints (kg CO2e, production + transport + end-of-life, use phase excluded).
# Laptop: Apple Product Environmental Report, MacBook Air M4 (March 2025): 13-inch 256 GB = 120 kg
#   lifecycle, of which ~75 % is non-use (production 71 %, transport 3 %, EoL <1 %).
# Server (glg3): Boavizta API estimate for i9-11900KF, 2x32 GB RAM, 2 TB + 1 TB SSD = 540 kg
#   (range 415-931), plus an assumed ~150 kg for the RTX 3090 (no vendor figure published).
# RR/ToR = 1 for both: the laptop was dedicated and glg3 is exclusive during booked slots.
SCI_SCENARIOS <- list(
  central = list(laptop_te_kg = 120 * 0.75, server_te_kg = 540 + 150, lifetime_years = 4),
  low     = list(laptop_te_kg = 120 * 0.75, server_te_kg = 415 + 100, lifetime_years = 6),
  high    = list(laptop_te_kg = 120 * 0.75, server_te_kg = 931 + 300, lifetime_years = 3)
)

# Embodied emission rate (g CO2e per hour of use) for a deployment.
embodied_g_per_h <- function(deployment, sc) {
  te_kg <- sc$laptop_te_kg + ifelse(deployment == "remote", sc$server_te_kg, 0)
  te_kg * 1000 / (sc$lifetime_years * HOURS_PER_YEAR)
}

# Netherlands grid carbon intensity (15-min resolution, Fraunhofer Energy-Charts).
# Cached in data/; downloaded only when the cache file is missing.
load_carbon_intensity <- function(start_s, end_s) {
  if (!file.exists(CARBON_INTENSITY_FILE)) {
    fmt <- function(t) format(as.POSIXct(t, origin = "1970-01-01", tz = "UTC"), "%Y-%m-%dT%H:%MZ", tz = "UTC")
    url <- sprintf("https://api.energy-charts.info/co2eq?country=nl&start=%s&end=%s",
                   fmt(start_s - 86400), fmt(end_s + 86400))
    j <- fromJSON(url)
    ci <- tibble(unix_s = as.numeric(j$unix_seconds), g_co2eq_per_kwh = as.numeric(j$co2eq)) %>%
      filter(is.finite(g_co2eq_per_kwh))
    dir.create(dirname(CARBON_INTENSITY_FILE), recursive = TRUE, showWarnings = FALSE)
    write_csv(ci, CARBON_INTENSITY_FILE)
  }
  read_csv(CARBON_INTENSITY_FILE, show_col_types = FALSE) %>% arrange(unix_s)
}

carbon_intensity <- load_carbon_intensity(min(runs$run_start_unix_s, na.rm = TRUE),
                                          max(runs$run_start_unix_s, na.rm = TRUE))

# Each run gets the intensity of the 15-minute slot it started in.
ci_index <- findInterval(runs$run_start_unix_s, carbon_intensity$unix_s)
ci_index[ci_index == 0 | !is.finite(runs$run_start_unix_s)] <- NA
runs$grid_g_co2eq_per_kwh <- carbon_intensity$g_co2eq_per_kwh[ci_index]
if (any(is.na(runs$grid_g_co2eq_per_kwh))) {
  warning(sum(is.na(runs$grid_g_co2eq_per_kwh)), " run(s) have no carbon intensity; delete ",
          CARBON_INTENSITY_FILE, " to re-download for the full experiment period.")
}

runs <- runs %>%
  mutate(
    carbon_operational_g = total_energy_j / 3.6e6 * grid_g_co2eq_per_kwh,
    carbon_embodied_g = wall_time_s / 3600 * embodied_g_per_h(deployment, SCI_SCENARIOS$central),
    carbon_embodied_low_g = wall_time_s / 3600 * embodied_g_per_h(deployment, SCI_SCENARIOS$low),
    carbon_embodied_high_g = wall_time_s / 3600 * embodied_g_per_h(deployment, SCI_SCENARIOS$high),
    carbon_total_g = carbon_operational_g + carbon_embodied_g,
    sci_mg_per_output_token = ifelse(output_tokens > 0, 1000 * carbon_total_g / output_tokens, NA_real_)
  )

write_csv(
  runs %>% transmute(source_dir, setup, config, prompt_size, run, output_tokens, finish_reason, total_energy_j, valid),
  file.path(OUTPUT_DIR, "data_quality.csv"), na = ""
)

df <- runs %>% filter(valid) %>% droplevels()
write_csv(df %>% arrange(setup, config, prompt_size, run), file.path(OUTPUT_DIR, "run_table.csv"), na = "")

local_df <- df %>% filter(deployment == "on-device")
remote_df <- df %>% filter(deployment == "remote")
remote_base <- remote_df %>% filter(setup == "Remote (baseline)")

# ============================================================
# Summary + stats helpers
# ============================================================

mean_ci <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  m <- mean(x)
  h <- if (n > 1) qt(0.975, n - 1) * sd(x) / sqrt(n) else 0
  tibble(n = n, mean = m, sd = if (n > 1) sd(x) else NA_real_, median = median(x), lo = m - h, hi = m + h)
}

summarise_metrics <- function(data, groups, metrics) {
  data %>%
    select(all_of(c(groups, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(across(all_of(c(groups, "metric")))) %>%
    summarise(mean_ci(value), .groups = "drop")
}

# Cliff's delta: effect size for two independent samples (|d| >= 0.474 = large).
cliffs_delta <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (!length(a) || !length(b)) return(NA_real_)
  mean(sign(outer(a, b, "-")))
}

# Wilcoxon rank-sum test of every level of `factor_col` against the first (reference) level, per stratum.
pairwise_tests <- function(data, factor_col, strata, metrics, rq) {
  lv <- levels(droplevels(factor(data[[factor_col]])))
  if (length(lv) < 2) return(tibble())
  map_dfr(lv[-1], function(other) pairwise_test_two(data, factor_col, c(lv[1], other), strata, metrics)) %>%
    mutate(rq = rq, p_holm = p.adjust(wilcox_p, "holm"), .before = 1)
}

pairwise_test_two <- function(data, factor_col, lv, strata, metrics) {
  data %>%
    filter(.data[[factor_col]] %in% lv) %>%
    select(all_of(c(factor_col, strata, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(across(all_of(c(strata, "metric")))) %>%
    group_modify(function(g, key) {
      a <- g$value[g[[factor_col]] == lv[1]]
      b <- g$value[g[[factor_col]] == lv[2]]
      a <- a[is.finite(a)]; b <- b[is.finite(b)]
      if (length(a) < 2 || length(b) < 2) return(tibble())
      tibble(
        group_a = lv[1], group_b = lv[2],
        median_a = median(a), median_b = median(b),
        pct_change_b_vs_a = 100 * (median(b) - median(a)) / median(a),
        wilcox_p = suppressWarnings(wilcox.test(a, b, exact = FALSE)$p.value),
        cliffs_delta_b_vs_a = -cliffs_delta(a, b)
      )
    }) %>%
    ungroup()
}

# ============================================================
# Plot helpers
# ============================================================

save_plot <- function(plot, filename, width = 11, height = 6.5) {
  ggsave(file.path(PLOTS_DIR, filename), plot, width = width, height = height, dpi = 300, bg = "white")
}

theme_set(
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 10.5, colour = "grey30"),
      axis.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold", size = 10.5),
      strip.background = element_rect(fill = "grey95", colour = NA),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.spacing = unit(1.1, "lines")
    )
)

PAL_SETUP <- c("On-device" = "#2a6fdb", "Remote (baseline)" = "#e07b39", "Remote +100 ms" = "#b23a48",
               "Remote +500 ms" = "#6a1b4d")
PAL_CONFIG <- c("Qwen3-8B (Dense) Q4_K_M" = "#2a9d8f", "Qwen3-8B (Dense) Q8_0" = "#264653", "Qwen3-16B-A3B (MoE) Q4_K_M" = "#e9c46a")
fill_scale <- function(pal) scale_fill_manual(values = pal, drop = TRUE)
colour_scale <- function(pal) scale_colour_manual(values = pal, drop = TRUE)

# Box + jittered runs + mean diamond, dodged by a grouping variable.
dodged_box <- function(data, x, y, group, pal) {
  pd <- position_dodge(width = 0.8)
  ggplot(data, aes(x = .data[[x]], y = .data[[y]], fill = .data[[group]])) +
    geom_boxplot(width = 0.7, alpha = 0.55, outlier.shape = NA, position = pd, linewidth = 0.4) +
    geom_point(aes(colour = .data[[group]]), position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8, seed = 1),
               size = 1.4, alpha = 0.6, show.legend = FALSE) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 2.4, fill = "white", colour = "black", position = pd,
                 show.legend = FALSE) +
    fill_scale(pal) + colour_scale(pal)
}

# Multi-metric small multiples: one panel per metric, free y-scale.
metric_labels <- c(
  total_energy_j = "Total energy (J)",
  energy_j_per_output_token = "Energy per output token (J/token)",
  wall_time_s = "Execution time (s)",
  tokens_per_second = "Generation throughput (tokens/s)",
  avg_power_w = "Average power (W)",
  cpu_usage_avg_pct = "CPU usage, mean (%)",
  gpu_usage_avg_pct = "GPU usage, mean (%)",
  memory_avg_mb = "Memory used, mean (MB)",
  client_energy_j = "Client (Mac) energy (J)",
  server_energy_j = "Server energy (J)",
  client_j_per_output_token = "Client energy per token (J/token)",
  server_j_per_output_token = "Server energy per token (J/token)",
  network_rtt_ms = "Measured RTT (ms)",
  sci_mg_per_output_token = "SCI (mg CO2e/token)"
)

metric_panel <- function(data, metrics, x, group, pal, title, subtitle) {
  long <- data %>%
    select(all_of(c(x, group, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    filter(is.finite(value)) %>%
    mutate(metric = factor(metric, levels = metrics, labels = metric_labels[metrics]))
  dodged_box(long, x, "value", group, pal) +
    facet_wrap(~ metric, scales = "free_y", ncol = 3) +
    labs(title = title, subtitle = subtitle, x = "Prompt length", y = NULL, fill = NULL)
}

BOX_NOTE <- "Box = IQR of runs, dots = individual runs, diamond = mean"

# ============================================================
# Overview — every setup and model on one page
# ============================================================

overview <- df %>%
  group_by(setup, config, prompt_size) %>%
  summarise(mean_ci(energy_j_per_output_token), .groups = "drop")

p <- ggplot(overview, aes(x = mean, y = config, colour = setup)) +
  geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.8, position = position_dodge(width = 0.6)) +
  geom_point(size = 3, position = position_dodge(width = 0.6)) +
  facet_wrap(~ prompt_size, nrow = 1) +
  colour_scale(PAL_SETUP) +
  scale_y_discrete(limits = rev) +
  expand_limits(x = 0) +
  labs(title = "Overview · Energy per output token for every configuration",
       subtitle = "Mean ± 95% CI over runs. Further left = more energy-efficient. Remote = client + server energy",
       x = "Energy per output token (J/token)", y = NULL, colour = NULL)
save_plot(p, "00_overview_energy_per_token.png", 13, 5)

# ============================================================
# RQ1 — On-device vs remote across prompt lengths
# (remote = baseline network condition only, so delay does not confound)
# ============================================================

rq1_df <- df %>% filter(setup %in% c("On-device", "Remote (baseline)")) %>% droplevels()

if (n_distinct(rq1_df$setup) == 2) {
  # Fig 1: J/token vs actual prompt tokens, mean +/- 95% CI per condition.
  rq1_trend <- rq1_df %>%
    group_by(setup, config, prompt_size) %>%
    summarise(prompt_tokens = mean(prompt_tokens), mean_ci(energy_j_per_output_token), .groups = "drop")

  p <- ggplot(rq1_trend, aes(prompt_tokens, mean, colour = setup, group = setup)) +
    geom_line(linewidth = 0.9) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0, linewidth = 0.7) +
    geom_point(size = 2.8) +
    facet_wrap(~ config, nrow = 1) +
    colour_scale(PAL_SETUP) +
    expand_limits(y = 0) +
    labs(
      title = "RQ1 · Energy per output token vs prompt length",
      subtitle = "Mean ± 95% CI over runs. Remote = client (Mac) + server energy, baseline network",
      x = "Prompt length (input tokens)", y = "Energy per output token (J/token)", colour = NULL
    )
  save_plot(p, "RQ1_01_energy_per_token_vs_prompt_tokens.png", 13, 5.5)

  # Fig 2: same for total energy, distribution view.
  p <- dodged_box(rq1_df, "prompt_size", "total_energy_j", "setup", PAL_SETUP) +
    facet_wrap(~ config, nrow = 1) +
    expand_limits(y = 0) +
    labs(title = "RQ1 · Total energy per request", subtitle = BOX_NOTE,
         x = "Prompt length", y = "Total energy (J)", fill = NULL)
  save_plot(p, "RQ1_02_total_energy_boxplots.png", 13, 5.5)

  # Fig 3: where the energy goes — stacked components.
  rq1_parts <- rq1_df %>%
    transmute(setup, config, prompt_size,
              `Client / on-device Mac` = client_energy_j,
              `Server CPU package` = server_cpu_energy_j,
              `Server DRAM` = server_dram_energy_j,
              `Server GPU` = server_gpu_energy_j) %>%
    pivot_longer(-c(setup, config, prompt_size), names_to = "component", values_to = "j") %>%
    group_by(setup, config, prompt_size, component) %>%
    summarise(j = mean(j, na.rm = TRUE), .groups = "drop") %>%
    filter(is.finite(j)) %>%
    mutate(component = factor(component, levels = c("Server GPU", "Server DRAM", "Server CPU package", "Client / on-device Mac")))

  p <- ggplot(rq1_parts, aes(x = setup, y = j, fill = component)) +
    geom_col(width = 0.7) +
    facet_grid(config ~ prompt_size, scales = "free_y") +
    scale_fill_manual(values = c("Client / on-device Mac" = "#2a6fdb", "Server CPU package" = "#f4a261",
                                 "Server DRAM" = "#e9c46a", "Server GPU" = "#e76f51")) +
    labs(title = "RQ1 · Energy breakdown per request (mean)",
         subtitle = "On-device = Mac only; remote = Mac waiting for reply + server CPU, DRAM and GPU",
         x = NULL, y = "Mean energy (J)", fill = NULL)
  save_plot(p, "RQ1_03_energy_breakdown.png", 11, 9)

  # Fig 4: ratio remote / on-device (how many times more energy remote costs end-to-end).
  rq1_ratio <- rq1_df %>%
    group_by(setup, config, prompt_size) %>%
    summarise(jpt = mean(energy_j_per_output_token), .groups = "drop") %>%
    pivot_wider(names_from = setup, values_from = jpt) %>%
    mutate(ratio = `Remote (baseline)` / `On-device`)

  p <- ggplot(rq1_ratio, aes(prompt_size, ratio, fill = config)) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    geom_text(aes(label = sprintf("%.1f×", ratio)), position = position_dodge(0.8), vjust = -0.4, size = 3.3) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    fill_scale(PAL_CONFIG) +
    expand_limits(y = 0) +
    labs(title = "RQ1 · Remote energy relative to on-device (per output token)",
         subtitle = "Bar = how many times more energy per generated token remote uses (client + server). Dashed line = equal",
         x = "Prompt length", y = "Remote / on-device (J/token)", fill = NULL)
  save_plot(p, "RQ1_04_remote_vs_on_device_ratio.png", 10, 6)

  write_csv(summarise_metrics(rq1_df, c("setup", "config", "prompt_size"),
                              c("prompt_tokens", "total_energy_j", "energy_j_per_output_token",
                                "client_energy_j", "server_energy_j", "wall_time_s")),
            file.path(OUTPUT_DIR, "RQ1_summary.csv"), na = "")

  rq1_tests <- pairwise_tests(rq1_df, "setup", c("config", "prompt_size"),
                              c("total_energy_j", "energy_j_per_output_token", "wall_time_s"), "RQ1")
  write_csv(rq1_tests, file.path(OUTPUT_DIR, "RQ1_tests.csv"), na = "")
}

# ============================================================
# RQ2 — Quantization (Qwen3-8B Q4_K_M vs Q8_0, on-device)
# ============================================================

RESOURCE_METRICS <- c("total_energy_j", "energy_j_per_output_token", "avg_power_w",
                      "wall_time_s", "tokens_per_second", "cpu_usage_avg_pct",
                      "gpu_usage_avg_pct", "memory_avg_mb")

rq2_df <- local_df %>% filter(model_family == "Qwen3-8B (Dense)") %>% droplevels()

if (n_distinct(rq2_df$quantization) >= 2) {
  rq2_df$quantization <- factor(rq2_df$quantization, levels = c("Q4_K_M", "Q8_0"))
  pal_q <- c(Q4_K_M = "#2a9d8f", Q8_0 = "#264653")

  p <- metric_panel(rq2_df, RESOURCE_METRICS, "prompt_size", "quantization", pal_q,
                    "RQ2 · Effect of quantization on on-device Qwen3-8B", BOX_NOTE)
  save_plot(p, "RQ2_01_quantization_all_metrics.png", 13, 10)

  # Relative change Q8_0 vs Q4_K_M (median-based), one bar per metric x prompt.
  rq2_tests <- pairwise_tests(rq2_df, "quantization", "prompt_size", RESOURCE_METRICS, "RQ2")
  write_csv(rq2_tests, file.path(OUTPUT_DIR, "RQ2_tests.csv"), na = "")

  p <- ggplot(rq2_tests %>% mutate(metric = factor(metric, levels = rev(RESOURCE_METRICS), labels = metric_labels[rev(RESOURCE_METRICS)])),
              aes(pct_change_b_vs_a, metric, fill = prompt_size)) +
    geom_col(position = position_dodge(0.8), width = 0.75) +
    geom_vline(xintercept = 0) +
    scale_fill_manual(values = c(Short = "#a8dadc", Medium = "#457b9d", Long = "#1d3557")) +
    labs(title = "RQ2 · Q8_0 relative to Q4_K_M",
         subtitle = "Change in median; positive = Q8_0 higher. Significance in RQ2_tests.csv",
         x = "Change vs Q4_K_M (%)", y = NULL, fill = "Prompt")
  save_plot(p, "RQ2_02_quantization_relative_change.png", 10, 6)

  write_csv(summarise_metrics(rq2_df, c("quantization", "prompt_size"), RESOURCE_METRICS),
            file.path(OUTPUT_DIR, "RQ2_summary.csv"), na = "")
}

# ============================================================
# RQ3 — Model choice on-device
# Fair comparison at equal quantization (Q4_K_M): Dense 8B vs MoE 16B-A3B.
# The Q8_0 variant is still shown in the overview figure for context.
# ============================================================

rq3_df <- local_df %>% filter(quantization == "Q4_K_M") %>% droplevels()

if (n_distinct(rq3_df$model_family) >= 2) {
  pal_m <- c("Qwen3-8B (Dense)" = "#2a9d8f", "Qwen3-16B-A3B (MoE)" = "#e9c46a")
  rq3_df$model_family <- factor(rq3_df$model_family, levels = names(pal_m))

  p <- metric_panel(rq3_df, RESOURCE_METRICS, "prompt_size", "model_family", pal_m,
                    "RQ3 · Model comparison on-device (both Q4_K_M)", BOX_NOTE)
  save_plot(p, "RQ3_01_models_all_metrics.png", 13, 10)

  rq3_tests <- pairwise_tests(rq3_df, "model_family", "prompt_size", RESOURCE_METRICS, "RQ3")
  write_csv(rq3_tests, file.path(OUTPUT_DIR, "RQ3_tests.csv"), na = "")

  # Trade-off: speed vs energy efficiency for every on-device config.
  # Trade-off: speed vs energy efficiency for every configuration and setup.
  tradeoff <- df %>%
    group_by(setup, config, config_short) %>%
    summarise(tps = mean(tokens_per_second, na.rm = TRUE),
              jpt = mean(energy_j_per_output_token, na.rm = TRUE), .groups = "drop")

  p <- ggplot(tradeoff, aes(tps, jpt, colour = config, shape = setup)) +
    geom_point(size = 4, alpha = 0.9) +
    geom_text(aes(label = config_short), vjust = -1.1, size = 3.2, show.legend = FALSE) +
    colour_scale(PAL_CONFIG) +
    scale_shape_manual(values = c("On-device" = 16, "Remote (baseline)" = 17, "Remote +100 ms" = 15, "Remote +500 ms" = 18)) +
    scale_x_log10() +
    expand_limits(y = 0) +
    labs(title = "Speed vs energy trade-off of every configuration",
         subtitle = "Mean over all prompt lengths and runs. Bottom-right is best: fast and low energy per token (log x-axis)",
         x = "Generation throughput (tokens/s, log scale)", y = "Energy per output token (J/token)",
         colour = NULL, shape = NULL)
  save_plot(p, "RQ3_02_speed_energy_tradeoff.png", 10.5, 6.5)

  write_csv(summarise_metrics(local_df, c("config", "prompt_size"), RESOURCE_METRICS),
            file.path(OUTPUT_DIR, "RQ3_summary.csv"), na = "")
}

# ============================================================
# RQ4 — Network delay on remote generation
# ============================================================

if (n_distinct(remote_df$setup) >= 2) {
  pal_net <- PAL_SETUP[levels(droplevels(remote_df$setup))]
  NET_METRICS <- c("energy_j_per_output_token", "client_j_per_output_token", "server_j_per_output_token",
                   "total_energy_j", "wall_time_s", "network_rtt_ms")

  p <- metric_panel(remote_df, NET_METRICS, "prompt_size", "setup", pal_net,
                    "RQ4 · Effect of added network delay on remote generation",
                    paste(BOX_NOTE, "· all remote models pooled; see RQ4_02 per model"))
  save_plot(p, "RQ4_01_delay_all_metrics.png", 13, 8)

  p <- dodged_box(remote_df, "prompt_size", "wall_time_s", "setup", pal_net) +
    facet_wrap(~ config, nrow = 1, scales = "free_y") +
    labs(title = "RQ4 · Execution time per model and delay", subtitle = BOX_NOTE,
         x = "Prompt length", y = "Request wall time (s)", fill = NULL)
  save_plot(p, "RQ4_02_delay_time_per_model.png", 13, 5.5)

  p <- dodged_box(remote_df, "prompt_size", "client_j_per_output_token", "setup", pal_net) +
    facet_wrap(~ config, nrow = 1, scales = "free_y") +
    labs(title = "RQ4 · Client-side energy per output token",
         subtitle = paste("The client waits longer under delay, so delay cost lands mainly here.", BOX_NOTE),
         x = "Prompt length", y = "Client energy per token (J/token)", fill = NULL)
  save_plot(p, "RQ4_03_delay_client_energy_per_model.png", 13, 5.5)

  rq4_tests <- pairwise_tests(remote_df, "setup", c("config", "prompt_size"), NET_METRICS, "RQ4")
  write_csv(rq4_tests, file.path(OUTPUT_DIR, "RQ4_tests.csv"), na = "")
  write_csv(summarise_metrics(remote_df, c("setup", "config", "prompt_size"), NET_METRICS),
            file.path(OUTPUT_DIR, "RQ4_summary.csv"), na = "")
}

# ============================================================
# Carbon footprint (SCI) — summaries and figures
# ============================================================

sci_summary <- function(data, groups) {
  data %>%
    group_by(across(all_of(groups))) %>%
    summarise(
      n_runs = n(),
      output_tokens = sum(output_tokens),
      energy_kwh = sum(total_energy_j) / 3.6e6,
      runtime_h = sum(wall_time_s) / 3600,
      grid_g_per_kwh_weighted = sum(carbon_operational_g) / energy_kwh,
      operational_g = sum(carbon_operational_g),
      embodied_g = sum(carbon_embodied_g),
      embodied_low_g = sum(carbon_embodied_low_g),
      embodied_high_g = sum(carbon_embodied_high_g),
      total_g = operational_g + embodied_g,
      embodied_share_pct = 100 * embodied_g / total_g,
      sci_mg_per_token = 1000 * total_g / output_tokens,
      sci_low_mg_per_token = 1000 * (operational_g + embodied_low_g) / output_tokens,
      sci_high_mg_per_token = 1000 * (operational_g + embodied_high_g) / output_tokens,
      .groups = "drop"
    )
}

sci_by_config <- sci_summary(df, c("setup", "config"))
sci_total <- sci_summary(df %>% mutate(setup = "All runs", config = "All configurations"), c("setup", "config"))
write_csv(bind_rows(sci_by_config %>% mutate(across(c(setup, config), as.character)), sci_total),
          file.path(OUTPUT_DIR, "SCI_summary.csv"), na = "")
write_csv(sci_summary(df, c("setup", "config", "prompt_size")),
          file.path(OUTPUT_DIR, "SCI_summary_by_prompt.csv"), na = "")
write_csv(
  bind_rows(lapply(names(SCI_SCENARIOS), function(nm) as_tibble(SCI_SCENARIOS[[nm]]) %>% mutate(scenario = nm, .before = 1))),
  file.path(OUTPUT_DIR, "SCI_assumptions.csv"), na = ""
)

# Fig SCI 1: SCI per token split into operational and embodied, with sensitivity range.
sci_parts <- sci_by_config %>%
  transmute(setup, config,
            `Operational (E × I)` = 1000 * operational_g / output_tokens,
            `Embodied (M)` = 1000 * embodied_g / output_tokens) %>%
  pivot_longer(-c(setup, config), names_to = "part", values_to = "mg") %>%
  mutate(part = factor(part, levels = c("Embodied (M)", "Operational (E × I)")))

p <- ggplot(sci_parts, aes(x = config, y = mg)) +
  geom_col(aes(fill = part), width = 0.65) +
  geom_errorbar(data = sci_by_config, aes(y = sci_mg_per_token, ymin = sci_low_mg_per_token, ymax = sci_high_mg_per_token),
                width = 0.2, linewidth = 0.6) +
  geom_text(data = sci_by_config, aes(y = sci_high_mg_per_token, label = sprintf("%.2f", sci_mg_per_token)),
            vjust = -0.6, size = 3.4) +
  facet_wrap(~ setup, nrow = 1) +
  scale_fill_manual(values = c("Operational (E × I)" = "#457b9d", "Embodied (M)" = "#a8dadc")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Carbon intensity of generation (SCI per output token)",
       subtitle = "SCI = (E × I + M) / tokens. Whiskers = embodied sensitivity range (low/high hardware footprint and lifetime)",
       x = NULL, y = "SCI (mg CO2e per output token)", fill = NULL) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
save_plot(p, "SCI_01_per_token_breakdown.png", 13, 6.5)

# Fig SCI 2: SCI per token vs prompt length (per-run distribution).
sci_trend <- df %>%
  group_by(setup, config, prompt_size) %>%
  summarise(mean_ci(sci_mg_per_output_token), .groups = "drop")

p <- ggplot(sci_trend, aes(prompt_size, mean, colour = setup, group = setup)) +
  geom_line(linewidth = 0.9) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.08, linewidth = 0.7) +
  geom_point(size = 2.8) +
  facet_wrap(~ config, nrow = 1) +
  colour_scale(PAL_SETUP) +
  expand_limits(y = 0) +
  labs(title = "SCI per output token across prompt lengths",
       subtitle = "Mean ± 95% CI over runs, central embodied scenario",
       x = "Prompt length", y = "SCI (mg CO2e per output token)", colour = NULL)
save_plot(p, "SCI_02_per_token_vs_prompt.png", 13, 5.5)

# Fig SCI 3: grid carbon intensity during the experiment, with run windows.
to_time <- function(t) as.POSIXct(t, origin = "1970-01-01", tz = "UTC")
windows <- df %>%
  group_by(setup, config_short) %>%
  summarise(start = to_time(min(run_start_unix_s, na.rm = TRUE)),
            end = to_time(max(run_start_unix_s + wall_time_s, na.rm = TRUE)), .groups = "drop") %>%
  mutate(day = format(start, "%d %b %Y"))
ci_plot <- carbon_intensity %>%
  mutate(time = to_time(unix_s), day = format(time, "%d %b %Y")) %>%
  inner_join(windows %>% group_by(day) %>% summarise(from = min(start) - 3600, to = max(end) + 3600), by = "day") %>%
  filter(time >= from, time <= to)

p <- ggplot() +
  geom_rect(data = windows, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf, fill = setup), alpha = 0.2) +
  geom_text(data = windows, aes(x = start + (end - start) / 2, y = Inf, label = config_short),
            vjust = 1.5, size = 3) +
  geom_step(data = ci_plot, aes(time, g_co2eq_per_kwh), linewidth = 0.8) +
  facet_wrap(~ day, scales = "free_x") +
  fill_scale(PAL_SETUP) +
  scale_x_datetime(date_labels = "%H:%M") +
  labs(title = "Netherlands grid carbon intensity during the experiment",
       subtitle = "15-minute values (Fraunhofer Energy-Charts). Shaded = when each configuration ran (UTC)",
       x = "Time (UTC)", y = "Grid intensity (g CO2e/kWh)", fill = NULL)
save_plot(p, "SCI_03_grid_intensity_timeline.png", 13, 5)

# ============================================================
# Final console report
# ============================================================

cat("\n===============================================\n")
cat("ANALYSIS COMPLETE\n")
cat("===============================================\n")
cat("Metadata files found: ", length(metadata_files), "\n", sep = "")
cat("Valid runs: ", nrow(df), " (invalid: ", sum(!runs$valid), ", see data_quality.csv)\n", sep = "")
print(count(df, setup, config, prompt_size), n = Inf)
cat(sprintf("Carbon footprint: %.1f g CO2e total (%.1f operational + %.1f embodied), see SCI_summary.csv\n",
            sci_total$total_g, sci_total$operational_g, sci_total$embodied_g))
cat("Plots: ", normalizePath(PLOTS_DIR, mustWork = FALSE), "\n", sep = "")
