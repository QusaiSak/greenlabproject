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
# Start from an empty plots folder so no stale figures from older versions remain.
unlink(list.files(PLOTS_DIR, pattern = "\\.png$", full.names = TRUE))

# Figures used in the paper (results.tex) are copied here.
REPORT_FIG_DIR <- "reportTemplate/figures"
dir.create(REPORT_FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Hardware the CPU / GPU metrics come from.
#   On-device: MacBook Air M4 (Mac16,12), Apple M4 SoC = 10-core CPU (4P + 6E) + 10-core integrated GPU.
#     EnergiBridge reports only whole-system power on macOS, so Mac energy is not split into CPU / GPU.
#   Remote (glg3): Intel Core i9-11900KF CPU (RAPL package + DRAM) + NVIDIA GeForce RTX 3090 GPU (NVML power).
HARDWARE <- tibble(
  deployment = c("on-device", "remote"),
  cpu_model = c("Apple M4 (10-core CPU)", "Intel Core i9-11900KF"),
  gpu_model = c("Apple M4 (10-core integrated GPU)", "NVIDIA GeForce RTX 3090")
)
HW_NOTE <- paste0("Hardware - on-device: MacBook Air, Apple M4 CPU + integrated GPU | ",
                  "remote: Intel Core i9-11900KF CPU + NVIDIA GeForce RTX 3090 GPU")

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
  left_join(HARDWARE, by = "deployment") %>%
  relocate(cpu_model, gpu_model, .after = deployment) %>%
  mutate(
    energy_j_per_output_token = ifelse(output_tokens > 0, total_energy_j / output_tokens, NA_real_),
    client_j_per_output_token = ifelse(output_tokens > 0, client_energy_j / output_tokens, NA_real_),
    server_j_per_output_token = ifelse(output_tokens > 0, server_energy_j / output_tokens, NA_real_),
    server_cpu_j_per_output_token = ifelse(output_tokens > 0, server_cpu_energy_j / output_tokens, NA_real_),
    server_gpu_j_per_output_token = ifelse(output_tokens > 0, server_gpu_energy_j / output_tokens, NA_real_),
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
    grepl("16B-A3B", model) ~ paste("16B-A3B", sub("_.*", "", quantization)),
    TRUE ~ paste("8B", sub("_.*", "", quantization))
  ),
  config_short = factor(config_short, levels = unique(c("8B Q4", "8B Q8", "16B-A3B Q4", sort(unique(config_short))))))

# ============================================================
# Carbon footprint (Green Software Foundation SCI specification)
#   SCI = ((E * I) + M) per R
#   E = measured energy (kWh), I = grid carbon intensity at run time (gCO2e/kWh),
#   M = embodied emissions = TE * (TiR / EL) * (RR / ToR), R = one output token.
# Location-based: 15-minute Netherlands grid mix, no market instruments (RECs, offsets).
# Boundary (included): client Mac whole-system energy (EnergiBridge), server CPU package + DRAM (RAPL)
#   + GPU (NVML), embodied emissions of the Mac and of the server.
# Boundary (excluded): server PSU / motherboard / fans / storage power not covered by RAPL + NVML,
#   network equipment and transmission, model download and storage, idle time between runs.
# Each sub-component is computed per run and summed, with the same R (output token) and time window.
# ============================================================

HOURS_PER_YEAR <- 365 * 24

# Server PUE. glg3 is a single on-premise workstation in a regular room (no data-centre cooling
# or UPS overhead), so PUE = 1.0. The Mac is a personal device and has no PUE.
SERVER_PUE <- 1.0

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

# Embodied emission rate (g CO2e per hour of use) of one device.
embodied_g_per_h <- function(te_kg, sc) te_kg * 1000 / (sc$lifetime_years * HOURS_PER_YEAR)

# Netherlands grid carbon intensity (15-min resolution, Fraunhofer Energy-Charts).
# Cached in data/; (re-)downloaded when the cache does not cover every run.
load_carbon_intensity <- function(start_s, end_s) {
  cached <- if (file.exists(CARBON_INTENSITY_FILE)) read_csv(CARBON_INTENSITY_FILE, show_col_types = FALSE) else
    tibble(unix_s = numeric(), g_co2eq_per_kwh = numeric())
  covered <- nrow(cached) > 0 && min(cached$unix_s) <= start_s && max(cached$unix_s) >= end_s
  if (!covered) {
    fmt <- function(t) format(as.POSIXct(t, origin = "1970-01-01", tz = "UTC"), "%Y-%m-%dT%H:%MZ", tz = "UTC")
    url <- sprintf("https://api.energy-charts.info/co2eq?country=nl&start=%s&end=%s",
                   fmt(start_s - 86400), fmt(end_s + 86400))
    j <- fromJSON(url)
    ci <- tibble(unix_s = as.numeric(j$unix_seconds), g_co2eq_per_kwh = as.numeric(j$co2eq)) %>%
      filter(is.finite(g_co2eq_per_kwh))
    cached <- bind_rows(cached, ci) %>% distinct(unix_s, .keep_all = TRUE)
    dir.create(dirname(CARBON_INTENSITY_FILE), recursive = TRUE, showWarnings = FALSE)
    write_csv(cached %>% arrange(unix_s), CARBON_INTENSITY_FILE)
  }
  cached %>% arrange(unix_s)
}

carbon_intensity <- load_carbon_intensity(min(runs$run_start_unix_s, na.rm = TRUE),
                                          max(runs$run_start_unix_s + runs$wall_time_s, na.rm = TRUE))

# Each run gets the intensity of the 15-minute slot it started in (no value = NA, never the last slot).
ci_index <- findInterval(runs$run_start_unix_s, carbon_intensity$unix_s)
ci_index[ci_index == 0 | !is.finite(runs$run_start_unix_s)] <- NA
stale <- !is.na(ci_index) & runs$run_start_unix_s - carbon_intensity$unix_s[ci_index] > 900
ci_index[stale] <- NA
runs$grid_g_co2eq_per_kwh <- carbon_intensity$g_co2eq_per_kwh[ci_index]
if (any(is.na(runs$grid_g_co2eq_per_kwh))) {
  warning(sum(is.na(runs$grid_g_co2eq_per_kwh)), " run(s) have no carbon intensity; delete ",
          CARBON_INTENSITY_FILE, " to re-download for the full experiment period.")
}

z0 <- function(x) ifelse(is.finite(x), x, 0)
emb_g <- function(wall_s, te_kg, sc) wall_s / 3600 * embodied_g_per_h(te_kg, sc)
is_remote <- runs$deployment == "remote"

runs <- runs %>%
  mutate(
    carbon_client_op_g = client_energy_j / 3.6e6 * grid_g_co2eq_per_kwh,
    carbon_server_op_g = z0(server_energy_j) * SERVER_PUE / 3.6e6 * grid_g_co2eq_per_kwh,
    carbon_operational_g = carbon_client_op_g + carbon_server_op_g,
    carbon_emb_laptop_g = emb_g(wall_time_s, SCI_SCENARIOS$central$laptop_te_kg, SCI_SCENARIOS$central),
    carbon_emb_server_g = ifelse(is_remote, emb_g(wall_time_s, SCI_SCENARIOS$central$server_te_kg, SCI_SCENARIOS$central), 0),
    carbon_embodied_g = carbon_emb_laptop_g + carbon_emb_server_g,
    carbon_embodied_low_g = emb_g(wall_time_s, SCI_SCENARIOS$low$laptop_te_kg + ifelse(is_remote, SCI_SCENARIOS$low$server_te_kg, 0), SCI_SCENARIOS$low),
    carbon_embodied_high_g = emb_g(wall_time_s, SCI_SCENARIOS$high$laptop_te_kg + ifelse(is_remote, SCI_SCENARIOS$high$server_te_kg, 0), SCI_SCENARIOS$high),
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

# Descriptive statistics of one sample (mean +/- 95% CI in lo / hi).
mean_ci <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n == 0) return(tibble(n = 0L, mean = NA_real_, sd = NA_real_, var = NA_real_, median = NA_real_,
                            q1 = NA_real_, q3 = NA_real_, min = NA_real_, max = NA_real_, lo = NA_real_, hi = NA_real_))
  m <- mean(x)
  s <- if (n > 1) sd(x) else NA_real_
  h <- if (n > 1) qt(0.975, n - 1) * s / sqrt(n) else 0
  tibble(n = n, mean = m, sd = s, var = s^2, median = median(x),
         q1 = unname(quantile(x, 0.25)), q3 = unname(quantile(x, 0.75)),
         min = min(x), max = max(x), lo = m - h, hi = m + h)
}

summarise_metrics <- function(data, groups, metrics) {
  data %>%
    select(all_of(c(groups, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(across(all_of(c(groups, "metric")))) %>%
    summarise(mean_ci(value), .groups = "drop")
}

# Cliff's delta of a vs b (|d| >= 0.474 = large).
cliffs_delta <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (!length(a) || !length(b)) return(NA_real_)
  mean(sign(outer(a, b, "-")))
}

# Paired comparison of two levels of `factor_col`, per stratum. Runs are paired by repetition number.
# Shapiro-Wilk on the paired differences decides the test: paired t-test (+ Cohen's d) when normal,
# Wilcoxon signed-rank (+ Cliff's delta) otherwise. Both effect sizes are always reported.
pairwise_test_two <- function(data, factor_col, lv, strata, metrics) {
  data %>%
    filter(.data[[factor_col]] %in% lv) %>%
    select(all_of(c(factor_col, strata, "run", metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(across(all_of(c(strata, "metric")))) %>%
    group_modify(function(g, key) {
      a <- g[g[[factor_col]] == lv[1], c("run", "value")]
      b <- g[g[[factor_col]] == lv[2], c("run", "value")]
      p <- inner_join(a, b, by = "run", suffix = c("_a", "_b")) %>% filter(is.finite(value_a), is.finite(value_b))
      if (nrow(p) < 3) return(tibble())
      d <- p$value_b - p$value_a
      sw <- if (length(unique(d)) >= 3) shapiro.test(d)$p.value else NA_real_
      normal <- is.finite(sw) && sw > 0.05
      p_t <- if (sd(d) > 0) t.test(p$value_b, p$value_a, paired = TRUE)$p.value else 1
      p_w <- if (any(d != 0)) suppressWarnings(wilcox.test(p$value_b, p$value_a, paired = TRUE, exact = FALSE)$p.value) else 1
      tibble(
        group_a = lv[1], group_b = lv[2], n_pairs = nrow(p),
        median_a = median(p$value_a), median_b = median(p$value_b),
        pct_change_b_vs_a = 100 * (median(p$value_b) - median(p$value_a)) / median(p$value_a),
        shapiro_p_diff = sw, normal_diff = normal,
        test = ifelse(normal, "paired t", "Wilcoxon signed-rank"),
        p_value = ifelse(normal, p_t, p_w),
        cohens_d = ifelse(sd(d) > 0, mean(d) / sd(d), NA_real_),
        cliffs_delta_b_vs_a = cliffs_delta(p$value_b, p$value_a)
      )
    }) %>%
    ungroup()
}

# Every level of `factor_col` against the first (reference) level; Holm correction over the whole RQ.
pairwise_tests <- function(data, factor_col, strata, metrics, rq) {
  lv <- levels(droplevels(factor(data[[factor_col]])))
  if (length(lv) < 2) return(tibble())
  map_dfr(lv[-1], function(other) pairwise_test_two(data, factor_col, c(lv[1], other), strata, metrics)) %>%
    mutate(rq = rq, p_holm = p.adjust(p_value, "holm"), .before = 1)
}

# Friedman test (paired, k levels) per stratum; blocks = repetition number.
friedman_tests <- function(data, factor_col, strata, metrics, rq) {
  data %>%
    select(all_of(c(factor_col, strata, "run", metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(across(all_of(c(strata, "metric")))) %>%
    group_modify(function(g, key) {
      w <- g %>% transmute(run, lvl = as.character(.data[[factor_col]]), value) %>%
        pivot_wider(names_from = lvl, values_from = value) %>% na.omit()
      if (nrow(w) < 3) return(tibble())
      ft <- friedman.test(as.matrix(w[, -1]))
      tibble(n_blocks = nrow(w), chi2 = unname(ft$statistic), df = unname(ft$parameter), p_value = ft$p.value)
    }) %>%
    ungroup() %>%
    mutate(rq = rq, p_holm = p.adjust(p_value, "holm"), .before = 1)
}

# Share of paired-difference cells that pass Shapiro-Wilk, per metric.
normality_summary <- function(tests) {
  # RQ2 / RQ3 are tested per setup (on-device and remote); RQ1 / RQ4 have no setup column (NA).
  tests %>% group_by(rq, across(any_of("setup")), group_b, metric) %>%
    summarise(cells = n(), normal = sum(normal_diff), non_normal = cells - normal, .groups = "drop")
}

# ============================================================
# Plot helpers
# ============================================================

save_plot <- function(plot, filename, width = 11, height = 6.5) {
  plot <- plot + labs(caption = HW_NOTE) + theme(plot.caption = element_text(size = 9, colour = "grey40", hjust = 0))
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
    stat_summary(aes(group = .data[[group]]), fun = mean, geom = "point", shape = 23, size = 2.4, fill = "white",
                 colour = "black", position = pd,
                 show.legend = FALSE) +
    fill_scale(pal) + colour_scale(pal)
}

metric_labels <- c(
  total_energy_j = "Total energy (J)",
  energy_j_per_output_token = "Energy per output token (J/token)",
  wall_time_s = "Execution time (s)",
  tokens_per_second = "Generation throughput (tokens/s)",
  avg_power_w = "Average power (W)",
  cpu_usage_avg_pct = "CPU usage, mean (%)",
  gpu_usage_avg_pct = "GPU usage, mean (%)",
  memory_avg_mb = "Memory used, mean (MB)",
  output_tokens = "Output tokens",
  client_energy_j = "Client (Mac) energy (J)",
  server_energy_j = "Server energy (J)",
  client_j_per_output_token = "Client energy per token (J/token)",
  server_j_per_output_token = "Server energy per token (J/token)",
  server_cpu_energy_j = "Server CPU energy, i9-11900KF (J)",
  server_gpu_energy_j = "Server GPU energy, RTX 3090 (J)",
  server_cpu_j_per_output_token = "Server CPU (i9-11900KF) energy per token (J/token)",
  server_gpu_j_per_output_token = "Server GPU (RTX 3090) energy per token (J/token)",
  network_rtt_ms = "Measured RTT (ms)",
  sci_mg_per_output_token = "SCI (mg CO2e/token)"
)

BOX_NOTE <- "Box = IQR of runs, dots = individual runs, diamond = mean (value printed above each box)"
VIOLIN_NOTE <- "Violin = density of the 10 runs, box = median and IQR, dots = individual runs"

# Mean value printed above each box so exact numbers are readable.
mean_label <- function(group) {
  stat_summary(aes(group = .data[[group]]),
               fun.data = function(v) data.frame(y = max(v, na.rm = TRUE), label = format(signif(mean(v, na.rm = TRUE), 3))),
               geom = "text", position = position_dodge(width = 0.8), vjust = -0.7, size = 2.9, colour = "grey15")
}

# One image per metric: panels = levels of `by`, one row.
metric_files <- function(data, metrics, group, pal, by, prefix, title_prefix) {
  for (i in seq_along(metrics)) {
    m <- metrics[i]
    d <- data %>% filter(is.finite(.data[[m]]))
    if (!nrow(d)) next
    p <- dodged_box(d, "prompt_size", m, group, pal) +
      mean_label(group) +
      facet_wrap(vars(.data[[by]]), nrow = 1, scales = "free_y") +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
      labs(title = paste(title_prefix, "·", metric_labels[[m]]), subtitle = BOX_NOTE,
           x = "Prompt length", y = metric_labels[[m]], fill = NULL)
    save_plot(p, sprintf("%s_%02d_%s.png", prefix, i, m), 13, 5.5)
  }
}

# Violin + box + runs for several metrics in one image. Rows = metrics; columns = levels of `by` (optional).
# free_panels = TRUE gives every (metric, by) panel its own y-axis, for columns measured on different hardware.
dist_figure <- function(data, metrics, group, pal, title, filename, by = NULL, ncol = 2, width = 9, height = 10,
                        free_panels = FALSE, note = NULL) {
  long <- data %>%
    select(all_of(c("prompt_size", group, by, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    filter(is.finite(value)) %>%
    mutate(metric = factor(metric, levels = metrics, labels = metric_labels[metrics]))
  pd <- position_dodge(width = 0.8)
  # Metric names are drawn on the left as y-axis titles (strip switched to the outside of the axis).
  facet <- if (is.null(by)) facet_wrap(~ metric, scales = "free_y", ncol = ncol, strip.position = "left",
                                       labeller = label_wrap_gen(30)) else if (free_panels)
    facet_wrap(vars(metric, .data[[by]]), scales = "free_y", ncol = n_distinct(long[[by]]), strip.position = "left",
               labeller = labeller(metric = label_wrap_gen(24), .multi_line = TRUE)) else
    facet_grid(rows = vars(metric), cols = vars(.data[[by]]), scales = "free_y", switch = "y",
               labeller = labeller(metric = label_wrap_gen(22), .default = label_wrap_gen(28)))
  p <- ggplot(long, aes(prompt_size, value, fill = .data[[group]])) +
    geom_violin(position = pd, width = 0.75, alpha = 0.35, scale = "width", linewidth = 0.3, trim = TRUE) +
    geom_boxplot(position = pd, width = 0.22, alpha = 0.85, outlier.shape = NA, linewidth = 0.35) +
    geom_point(aes(colour = .data[[group]]), position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.8, seed = 1),
               size = 0.8, alpha = 0.6, show.legend = FALSE) +
    facet + fill_scale(pal) + colour_scale(pal) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.12))) +
    labs(title = title, subtitle = paste(c(VIOLIN_NOTE, note), collapse = "\n"), x = "Prompt length", y = NULL, fill = NULL) +
    theme(strip.placement = "outside",
          strip.text.y.left = element_text(angle = 90, face = "bold", size = 10),
          strip.background.y = element_blank())
  save_plot(p, filename, width, height)
}

# Normal Q-Q plots of the standardised paired differences (level - reference) for one metric.
qq_figure <- function(data, factor_col, ref, others, metric, title, filename, width = 9, height = 6.5) {
  d <- map_dfr(others, function(o) {
    data %>% filter(.data[[factor_col]] %in% c(ref, o)) %>%
      transmute(config, prompt_size, run, lvl = ifelse(.data[[factor_col]] == ref, "a", "b"), value = .data[[metric]]) %>%
      pivot_wider(names_from = lvl, values_from = value) %>%
      mutate(diff = b - a, comparison = paste(o, "-", ref))
  }) %>%
    filter(is.finite(diff)) %>%
    group_by(config, prompt_size, comparison) %>%
    mutate(z = if (sd(diff) > 0) (diff - mean(diff)) / sd(diff) else 0) %>%
    ungroup()
  p <- ggplot(d, aes(sample = z, colour = comparison)) +
    stat_qq(size = 1.2, alpha = 0.8) +
    stat_qq_line(linewidth = 0.4) +
    facet_grid(rows = vars(config), cols = vars(prompt_size), labeller = labeller(config = label_wrap_gen(14))) +
    labs(title = title, subtitle = paste("Standardised paired differences of", metric_labels[[metric]],
                                         "; points on the line = normal"),
         x = "Theoretical quantiles", y = "Sample quantiles", colour = NULL)
  if (length(others) == 1) p <- p + scale_colour_manual(values = "#457b9d", guide = "none")
  save_plot(p, filename, width, height)
  # Paper version at column width (rows = configurations / setups).
  save_paper(p + labs(colour = NULL) + guides(colour = guide_legend(nrow = 2)),
             paste0("paper_", filename), height = 0.6 + 1.0 * n_distinct(d$config) + 0.25 * (length(others) > 1))
}

# ------------------------------------------------------------
# Paper figures: drawn at single-column size (3.4 in wide, 8 pt text) so they are readable in the
# two-column report without scaling. Titles and the hardware note go into the LaTeX caption instead.
# Files are saved as plots/paper_*.png and copied to reportTemplate/figures.
# ------------------------------------------------------------

PAPER_W <- 3.4
PAPER_LABELS <- c(energy_j_per_output_token = "Energy per token (J)", total_energy_j = "Total energy (J)",
                  wall_time_s = "Execution time (s)", memory_avg_mb = "Memory (MB)",
                  cpu_usage_avg_pct = "CPU usage (%)", gpu_usage_avg_pct = "GPU usage (%)",
                  client_j_per_output_token = "Client energy per token (J)", network_rtt_ms = "Measured RTT (ms)",
                  sci_mg_per_output_token = "SCI (mg CO2e per token)")

paper_theme <- theme_minimal(base_size = 8) +
  theme(plot.title = element_blank(), plot.subtitle = element_blank(), plot.caption = element_blank(),
        axis.title = element_text(face = "bold", size = 7.5), axis.text = element_text(size = 6.5, colour = "grey20"),
        strip.text = element_text(face = "bold", size = 6.3, margin = margin(2, 1, 2, 1)), strip.background = element_rect(fill = "grey93", colour = NA),
        legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 6.5),
        legend.key.size = unit(0.32, "cm"), legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(-6, 0, 0, 0),
        panel.grid.minor = element_blank(), panel.spacing = unit(0.45, "lines"), plot.margin = margin(3, 4, 2, 2))

save_paper <- function(p, filename, height, width = PAPER_W) {
  ggsave(file.path(PLOTS_DIR, filename), p + paper_theme, width = width, height = height, dpi = 400, bg = "white")
}

# Same design as the full-size plots: violin (density) + box (median, IQR) + every run as a point.
paper_box <- function(data, y, group, pal) {
  pd <- position_dodge(width = 0.8)
  ggplot(data, aes(prompt_size, .data[[y]], fill = .data[[group]])) +
    geom_violin(position = pd, width = 0.75, alpha = 0.35, scale = "width", linewidth = 0.25, trim = TRUE) +
    geom_boxplot(position = pd, width = 0.22, alpha = 0.85, outlier.shape = NA, linewidth = 0.3) +
    geom_point(aes(colour = .data[[group]]), position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.8, seed = 1),
               size = 0.45, alpha = 0.7, show.legend = FALSE) +
    fill_scale(pal) + colour_scale(pal) +
    labs(x = "Prompt length", fill = NULL)
}

# Several metrics in one column-width figure. Without `by`: one panel per metric (ncol columns).
# With `by`: one row per level of `by`, one column per metric; every panel has its own y-axis.
paper_metrics <- function(data, metrics, group, pal, filename, height, by = NULL, ncol = 2) {
  long <- data %>%
    select(all_of(c("prompt_size", group, by, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    filter(is.finite(value)) %>%
    mutate(metric = factor(metric, levels = metrics, labels = PAPER_LABELS[metrics]))
  facet <- if (is.null(by)) facet_wrap(~ metric, ncol = ncol, scales = "free_y") else
    facet_wrap(vars(.data[[by]], metric), ncol = length(metrics), scales = "free_y",
               labeller = labeller(.multi_line = TRUE))
  save_paper(paper_box(long, "value", group, pal) + facet + labs(y = NULL), filename, height)
}

# One metric, one panel per level of `by` stacked vertically.
paper_single <- function(data, metric, group, pal, filename, by = "config", height = 4.0, legend_rows = 1) {
  p <- paper_box(data %>% filter(is.finite(.data[[metric]])), metric, group, pal) +
    facet_wrap(vars(.data[[by]]), ncol = 1, scales = "free_y") +
    labs(y = PAPER_LABELS[[metric]]) +
    guides(fill = guide_legend(nrow = legend_rows))
  save_paper(p, filename, height)
}

# ============================================================
# RQ1 — On-device vs remote across prompt lengths
# (remote = baseline network condition only, so delay does not confound)
# ============================================================

rq1_df <- df %>% filter(setup %in% c("On-device", "Remote (baseline)")) %>% droplevels()
RQ1_METRICS <- c("total_energy_j", "energy_j_per_output_token", "wall_time_s", "output_tokens")

if (n_distinct(rq1_df$setup) == 2) {
  dist_figure(rq1_df, c("energy_j_per_output_token", "total_energy_j", "wall_time_s"), "setup", PAL_SETUP,
              "RQ1 · On-device vs remote (client + server)", "RQ1_01_distributions.png", by = "config",
              width = 12, height = 10)
  for (m in c("energy_j_per_output_token", "total_energy_j", "wall_time_s")) {
    paper_single(rq1_df, m, "setup", PAL_SETUP, sprintf("paper_RQ1_%s.png", m))
  }
  qq_figure(rq1_df, "setup", "On-device", "Remote (baseline)", "energy_j_per_output_token",
            "RQ1 · Normality of paired differences", "RQ1_02_qq_energy_per_token.png")

  write_csv(summarise_metrics(rq1_df, c("setup", "config", "prompt_size"),
                              c("prompt_tokens", RQ1_METRICS, "client_energy_j", "server_energy_j",
                                "server_cpu_energy_j", "server_dram_energy_j", "server_gpu_energy_j",
                                "cpu_usage_avg_pct", "gpu_usage_avg_pct")) %>%
              left_join(HARDWARE %>% mutate(setup = ifelse(deployment == "remote", "Remote (baseline)", "On-device")) %>%
                          select(-deployment), by = "setup"),
            file.path(OUTPUT_DIR, "RQ1_summary.csv"), na = "")

  rq1_tests <- pairwise_tests(rq1_df, "setup", c("config", "prompt_size"), RQ1_METRICS, "RQ1")
  write_csv(rq1_tests, file.path(OUTPUT_DIR, "RQ1_tests.csv"), na = "")
}

# ============================================================
# RQ2 — Quantization (Qwen3-8B Q4_K_M vs Q8_0), on-device and remote (baseline network)
# ============================================================

RESOURCE_METRICS <- c("total_energy_j", "energy_j_per_output_token", "wall_time_s", "cpu_usage_avg_pct",
                      "gpu_usage_avg_pct", "memory_avg_mb", "avg_power_w", "tokens_per_second")
PLOT_METRICS <- c("energy_j_per_output_token", "total_energy_j", "wall_time_s",
                  "memory_avg_mb", "cpu_usage_avg_pct", "gpu_usage_avg_pct")

# Tests run separately per setup; Holm correction is applied within each setup, so the
# on-device family (the RQ as defined) is not changed by adding the remote comparison.
tests_per_setup <- function(data, factor_col, metrics, rq) {
  map_dfr(levels(droplevels(data$setup)), function(st)
    pairwise_tests(data %>% filter(setup == st), factor_col, c("setup", "prompt_size"), metrics, rq))
}

rq2_df <- rq1_df %>% filter(model_family == "Qwen3-8B (Dense)") %>% droplevels()

if (n_distinct(rq2_df$quantization) >= 2) {
  rq2_df$quantization <- factor(rq2_df$quantization, levels = c("Q4_K_M", "Q8_0"))
  pal_q <- c(Q4_K_M = "#2a9d8f", Q8_0 = "#264653")

  dist_figure(rq2_df, PLOT_METRICS, "quantization", pal_q, "RQ2 · Qwen3-8B Q4_K_M vs Q8_0, on-device and remote",
              "RQ2_01_distributions.png", by = "setup", width = 10, height = 15,
              note = paste("Each panel has its own y-axis. Remote CPU / GPU / memory are measured on the server",
                          "\n(i9-11900KF, RTX 3090; memory = server RAM, model weights are in GPU VRAM)."), free_panels = TRUE)
  for (st in levels(droplevels(rq2_df$setup))) {
    paper_metrics(rq2_df %>% filter(setup == st), PLOT_METRICS, "quantization", pal_q,
                  sprintf("paper_RQ2_%s.png", ifelse(st == "On-device", "on_device", "remote")), height = 4.4)
  }
  qq_figure(rq2_df %>% mutate(config = setup), "quantization", "Q4_K_M", "Q8_0", "energy_j_per_output_token",
            "RQ2 · Normality of paired differences (Q8_0 - Q4_K_M)", "RQ2_02_qq_energy_per_token.png", height = 5.5)

  rq2_tests <- tests_per_setup(rq2_df, "quantization", RESOURCE_METRICS, "RQ2")
  write_csv(rq2_tests, file.path(OUTPUT_DIR, "RQ2_tests.csv"), na = "")
  write_csv(summarise_metrics(rq2_df, c("setup", "quantization", "prompt_size"), RESOURCE_METRICS),
            file.path(OUTPUT_DIR, "RQ2_summary.csv"), na = "")
}

# ============================================================
# RQ3 — Model choice at equal quantization (Q4_K_M): Dense 8B vs MoE 16B-A3B, on-device and remote
# ============================================================

rq3_df <- rq1_df %>% filter(quantization == "Q4_K_M") %>% droplevels()

if (n_distinct(rq3_df$model_family) >= 2) {
  pal_m <- c("Qwen3-8B (Dense)" = "#2a9d8f", "Qwen3-16B-A3B (MoE)" = "#e9c46a")
  rq3_df$model_family <- factor(rq3_df$model_family, levels = names(pal_m))

  dist_figure(rq3_df, PLOT_METRICS, "model_family", pal_m, "RQ3 · Dense vs MoE (Q4_K_M), on-device and remote",
              "RQ3_01_distributions.png", by = "setup", width = 10, height = 15,
              note = paste("Each panel has its own y-axis. Remote CPU / GPU / memory are measured on the server",
                          "\n(i9-11900KF, RTX 3090; memory = server RAM, model weights are in GPU VRAM)."), free_panels = TRUE)
  for (st in levels(droplevels(rq3_df$setup))) {
    paper_metrics(rq3_df %>% filter(setup == st), PLOT_METRICS, "model_family", pal_m,
                  sprintf("paper_RQ3_%s.png", ifelse(st == "On-device", "on_device", "remote")), height = 4.4)
  }
  qq_figure(rq3_df %>% mutate(config = setup), "model_family", "Qwen3-8B (Dense)", "Qwen3-16B-A3B (MoE)",
            "energy_j_per_output_token", "RQ3 · Normality of paired differences (MoE - Dense)",
            "RQ3_02_qq_energy_per_token.png", height = 5.5)

  rq3_tests <- tests_per_setup(rq3_df, "model_family", RESOURCE_METRICS, "RQ3")
  write_csv(rq3_tests, file.path(OUTPUT_DIR, "RQ3_tests.csv"), na = "")
  write_csv(summarise_metrics(rq3_df, c("setup", "model_family", "prompt_size"), RESOURCE_METRICS),
            file.path(OUTPUT_DIR, "RQ3_summary.csv"), na = "")
}

# ============================================================
# RQ4 — Network delay on remote generation
# ============================================================

if (n_distinct(remote_df$setup) >= 2) {
  pal_net <- PAL_SETUP[levels(droplevels(remote_df$setup))]
  NET_METRICS <- c("energy_j_per_output_token", "total_energy_j", "wall_time_s", "client_j_per_output_token",
                   "server_j_per_output_token", "network_rtt_ms")

  dist_figure(remote_df, c("energy_j_per_output_token", "wall_time_s", "client_j_per_output_token", "network_rtt_ms"),
              "setup", pal_net, "RQ4 · Remote generation under added network delay", "RQ4_01_distributions.png",
              by = "config", width = 12, height = 11)
  delay_lv <- levels(droplevels(remote_df$setup))
  paper_metrics(remote_df, c("energy_j_per_output_token", "wall_time_s"), "setup", pal_net,
                "paper_RQ4_energy_time.png", height = 4.6, by = "config")
  paper_metrics(remote_df, c("client_j_per_output_token", "network_rtt_ms"), "setup", pal_net,
                "paper_RQ4_client_rtt.png", height = 4.6, by = "config")
  qq_figure(remote_df, "setup", delay_lv[1], delay_lv[-1], "energy_j_per_output_token",
            "RQ4 · Normality of paired differences vs baseline", "RQ4_02_qq_energy_per_token.png")

  rq4_friedman <- friedman_tests(remote_df, "setup", c("config", "prompt_size"), NET_METRICS, "RQ4")
  write_csv(rq4_friedman, file.path(OUTPUT_DIR, "RQ4_friedman.csv"), na = "")
  rq4_tests <- pairwise_tests(remote_df, "setup", c("config", "prompt_size"), NET_METRICS, "RQ4")
  write_csv(rq4_tests, file.path(OUTPUT_DIR, "RQ4_tests.csv"), na = "")
  write_csv(summarise_metrics(remote_df, c("setup", "config", "prompt_size"), NET_METRICS),
            file.path(OUTPUT_DIR, "RQ4_summary.csv"), na = "")
}

# Outliers (planning: IQR rule per metric, and execution time > 2x the group median = throttling).
iqr_out <- function(x) { q <- quantile(x, c(0.25, 0.75), na.rm = TRUE); h <- 1.5 * diff(q); sum(x < q[1] - h | x > q[2] + h, na.rm = TRUE) }
outliers <- df %>%
  group_by(setup, config, prompt_size) %>%
  summarise(n = n(),
            iqr_energy_per_token = iqr_out(energy_j_per_output_token),
            iqr_total_energy = iqr_out(total_energy_j),
            iqr_wall_time = iqr_out(wall_time_s),
            time_over_2x_median = sum(wall_time_s > 2 * median(wall_time_s)), .groups = "drop")
write_csv(outliers, file.path(OUTPUT_DIR, "outliers.csv"), na = "")

all_tests <- bind_rows(mget(c("rq1_tests", "rq2_tests", "rq3_tests", "rq4_tests"), ifnotfound = list(NULL)))
write_csv(normality_summary(all_tests), file.path(OUTPUT_DIR, "normality_summary.csv"), na = "")

# ============================================================
# Carbon footprint (SCI) — summaries and figures
# ============================================================

# Sub-components are summed per run first, then aggregated: R (output token) and window stay the same.
sci_summary <- function(data, groups) {
  data %>%
    group_by(across(all_of(groups))) %>%
    summarise(
      n_runs = n(),
      output_tokens = sum(output_tokens),
      energy_kwh = sum(total_energy_j) / 3.6e6,
      client_kwh = sum(client_energy_j) / 3.6e6,
      server_kwh = sum(z0(server_energy_j)) * SERVER_PUE / 3.6e6,
      runtime_h = sum(wall_time_s) / 3600,
      grid_g_per_kwh_weighted = sum(carbon_operational_g) / energy_kwh,
      grid_g_per_kwh_min = min(grid_g_co2eq_per_kwh),
      grid_g_per_kwh_max = max(grid_g_co2eq_per_kwh),
      client_op_g = sum(carbon_client_op_g),
      server_op_g = sum(carbon_server_op_g),
      operational_g = sum(carbon_operational_g),
      laptop_emb_g = sum(carbon_emb_laptop_g),
      server_emb_g = sum(carbon_emb_server_g),
      embodied_g = sum(carbon_embodied_g),
      embodied_low_g = sum(carbon_embodied_low_g),
      embodied_high_g = sum(carbon_embodied_high_g),
      total_g = operational_g + embodied_g,
      embodied_share_pct = 100 * embodied_g / total_g,
      sci_client_op_mg = 1000 * client_op_g / output_tokens,
      sci_server_op_mg = 1000 * server_op_g / output_tokens,
      sci_laptop_emb_mg = 1000 * laptop_emb_g / output_tokens,
      sci_server_emb_mg = 1000 * server_emb_g / output_tokens,
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
  bind_rows(lapply(names(SCI_SCENARIOS), function(nm) as_tibble(SCI_SCENARIOS[[nm]]) %>% mutate(scenario = nm, .before = 1))) %>%
    mutate(server_pue = SERVER_PUE),
  file.path(OUTPUT_DIR, "SCI_assumptions.csv"), na = ""
)

# Fig SCI 1: SCI per token split into its four sub-components (composition, so stacked bars),
# whiskers = low / high embodied scenario.
SCI_PARTS <- c("Client operational" = "#2a6fdb", "Server operational" = "#e07b39",
               "Laptop embodied" = "#a8dadc", "Server embodied" = "#f4c095")
sci_parts <- sci_by_config %>%
  transmute(setup, config,
            `Client operational` = sci_client_op_mg, `Server operational` = sci_server_op_mg,
            `Laptop embodied` = sci_laptop_emb_mg, `Server embodied` = sci_server_emb_mg) %>%
  pivot_longer(-c(setup, config), names_to = "part", values_to = "mg") %>%
  filter(mg > 0) %>%
  mutate(part = factor(part, levels = rev(names(SCI_PARTS))))

p <- ggplot(sci_parts, aes(x = config, y = mg)) +
  geom_col(aes(fill = part), width = 0.65) +
  geom_errorbar(data = sci_by_config, aes(y = sci_mg_per_token, ymin = sci_low_mg_per_token, ymax = sci_high_mg_per_token),
                width = 0.2, linewidth = 0.6) +
  geom_text(data = sci_by_config, aes(y = sci_high_mg_per_token, label = sprintf("%.2f", sci_mg_per_token)),
            vjust = -0.6, size = 3.4) +
  facet_wrap(~ setup, nrow = 1) +
  scale_fill_manual(values = SCI_PARTS, breaks = names(SCI_PARTS)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "SCI per output token by sub-component",
       subtitle = "SCI = (E × I + M) / R, R = one output token. Whiskers = low / high embodied scenario",
       x = NULL, y = "SCI (mg CO2e per output token)", fill = NULL) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
save_plot(p, "SCI_01_per_token_breakdown.png", 13, 6.5)

# Fig SCI 2: per-run SCI distribution.
dist_figure(df, "sci_mg_per_output_token", "setup", PAL_SETUP, "SCI per output token, per run",
            "SCI_02_per_token_distribution.png", by = "config", width = 12, height = 4.8)

# Paper versions: horizontal stacked bars (no rotated labels) and per-run boxes, at column width.
sci_parts_p <- sci_parts %>% mutate(config_short = factor(config, levels = rev(levels(df$config))))
sci_cfg_p <- sci_by_config %>% mutate(config_short = factor(config, levels = levels(sci_parts_p$config_short)))
p <- ggplot(sci_parts_p, aes(x = mg, y = config_short)) +
  geom_col(aes(fill = part), width = 0.65) +
  geom_errorbar(data = sci_cfg_p, aes(x = sci_mg_per_token, xmin = sci_low_mg_per_token, xmax = sci_high_mg_per_token),
                width = 0.25, linewidth = 0.35, orientation = "y") +
  geom_text(data = sci_cfg_p, aes(x = sci_high_mg_per_token, label = sprintf("%.2f", sci_mg_per_token)),
            hjust = -0.25, size = 2.2) +
  facet_wrap(~ setup, ncol = 1) +
  scale_fill_manual(values = SCI_PARTS, breaks = names(SCI_PARTS)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "SCI (mg CO2e per output token)", y = NULL, fill = NULL) +
  guides(fill = guide_legend(nrow = 2))
save_paper(p, "paper_SCI_breakdown.png", height = 4.2)
paper_single(df, "sci_mg_per_output_token", "setup", PAL_SETUP, "paper_SCI_distribution.png", height = 4.4, legend_rows = 2)

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
# Paper tables: one CSV per table in results.tex (results/consolidated/paper_table_*.csv),
# rounded as printed in the paper. Figures used in the paper are copied to reportTemplate/figures.
# ============================================================

CFG_ORDER <- c("Qwen3-8B (Dense) Q4_K_M", "Qwen3-8B (Dense) Q8_0", "Qwen3-16B-A3B (MoE) Q4_K_M")
PROMPT_ORDER <- c("Short", "Medium", "Long")
SETUP_ORDER <- c("On-device", "Remote (baseline)", "Remote +100 ms", "Remote +500 ms")
PAPER_METRICS <- c("energy_j_per_output_token", "total_energy_j", "wall_time_s", "memory_avg_mb",
                   "cpu_usage_avg_pct", "gpu_usage_avg_pct")
DIGITS <- c(energy_j_per_output_token = 3, total_energy_j = 0, wall_time_s = 1, memory_avg_mb = 0,
            cpu_usage_avg_pct = 2, gpu_usage_avg_pct = 2, output_tokens = 0, client_j_per_output_token = 3)

paper_csv <- function(d, name) write_csv(d, file.path(OUTPUT_DIR, paste0("paper_table_", name, ".csv")), na = "")
sort_paper <- function(d) {
  d %>% mutate(.c = if ("config" %in% names(d)) match(as.character(config), CFG_ORDER) else 0,
               .p = match(as.character(prompt_size), PROMPT_ORDER),
               .s = if ("setup" %in% names(d)) match(as.character(setup), SETUP_ORDER) else 0) %>%
    arrange(.c, .p, .s) %>% select(-.c, -.p, -.s)
}
round_desc <- function(d) {
  d %>% mutate(dg = DIGITS[metric],
               across(c(mean, median, sd, min, max), ~ round(.x, dg)),
               var = signif(var, 4)) %>% select(-dg)
}
# Test used, Shapiro p of the paired differences, Holm p and the matching effect size.
paper_tests <- function(t, keys) {
  t %>% transmute(across(all_of(keys)), metric, comparison = paste(group_b, "vs", group_a), test,
                  shapiro_p = round(shapiro_p_diff, 3), p_holm = signif(p_holm, 3),
                  effect = ifelse(test == "paired t", "Cohen's d", "Cliff's delta"),
                  effect_size = round(ifelse(test == "paired t", cohens_d, cliffs_delta_b_vs_a), 2))
}

if (exists("rq1_tests")) {
  s1 <- read_csv(file.path(OUTPUT_DIR, "RQ1_summary.csv"), show_col_types = FALSE)
  paper_csv(s1 %>% filter(metric == "energy_j_per_output_token") %>%
              select(config, prompt_size, setup, metric, mean, median, sd, var, min, max) %>%
              round_desc() %>% sort_paper(), "rq1_descriptives")
  paper_csv(s1 %>% filter(metric %in% c("total_energy_j", "wall_time_s", "output_tokens")) %>%
              select(config, prompt_size, setup, metric, mean, sd) %>%
              mutate(dg = ifelse(metric == "wall_time_s", 1, 0), mean = round(mean, dg), sd = round(sd, dg)) %>%
              select(-dg) %>% pivot_wider(names_from = metric, values_from = c(mean, sd)) %>% sort_paper(),
            "rq1_energy_time_tokens")
  paper_csv(paper_tests(rq1_tests, c("config", "prompt_size")) %>% sort_paper(), "rq1_tests")
}

for (rq in c("rq2", "rq3")) {
  if (!exists(paste0(rq, "_tests"))) next
  grp <- if (rq == "rq2") "quantization" else "model_family"
  s <- read_csv(file.path(OUTPUT_DIR, paste0(toupper(rq), "_summary.csv")), show_col_types = FALSE)
  paper_csv(s %>% filter(metric %in% PAPER_METRICS) %>%
              select(setup, metric, prompt_size, all_of(grp), mean, median, sd, var, min, max) %>%
              round_desc() %>% mutate(.m = match(metric, PAPER_METRICS)) %>% sort_paper() %>%
              arrange(.m) %>% select(-.m), paste0(rq, "_descriptives"))
  paper_csv(paper_tests(get(paste0(rq, "_tests")) %>% filter(metric %in% PAPER_METRICS), c("setup", "prompt_size")) %>%
              mutate(.m = match(metric, PAPER_METRICS)) %>% sort_paper() %>% arrange(.m) %>% select(-.m),
            paste0(rq, "_tests"))
}

if (exists("rq4_tests")) {
  s4 <- read_csv(file.path(OUTPUT_DIR, "RQ4_summary.csv"), show_col_types = FALSE)
  paper_csv(s4 %>% filter(metric == "energy_j_per_output_token") %>%
              select(config, prompt_size, setup, mean, median, sd, min, max) %>%
              mutate(across(c(mean, median, sd, min, max), ~ round(.x, 3))) %>%
              left_join(s4 %>% filter(metric == "wall_time_s") %>%
                          transmute(config, prompt_size, setup, time_mean = round(mean, 2), time_sd = round(sd, 2)),
                        by = c("config", "prompt_size", "setup")) %>% sort_paper(), "rq4_descriptives")
  f4_metrics <- c("energy_j_per_output_token", "wall_time_s", "client_j_per_output_token")
  pw <- paper_tests(rq4_tests %>% filter(metric %in% f4_metrics), c("config", "prompt_size")) %>%
    mutate(lvl = ifelse(grepl("^Remote \\+100", comparison), "d100", "d500")) %>%
    select(config, prompt_size, metric, lvl, test, p_holm, effect, effect_size) %>%
    pivot_wider(names_from = lvl, values_from = c(test, p_holm, effect, effect_size))
  paper_csv(rq4_friedman %>% filter(metric %in% f4_metrics) %>%
              transmute(config, prompt_size, metric, friedman_chi2 = round(chi2, 1), friedman_p_holm = signif(p_holm, 3)) %>%
              left_join(pw, by = c("config", "prompt_size", "metric")) %>%
              mutate(.m = match(metric, f4_metrics)) %>% sort_paper() %>% arrange(.m) %>% select(-.m),
            "rq4_tests")
}

paper_csv(bind_rows(sci_by_config %>% mutate(across(c(setup, config), as.character)), sci_total) %>%
            transmute(setup, config, grid_g_per_kwh = round(grid_g_per_kwh_weighted, 0),
                      client_operational_mg = round(sci_client_op_mg, 3), server_operational_mg = round(sci_server_op_mg, 3),
                      laptop_embodied_mg = round(sci_laptop_emb_mg, 3), server_embodied_mg = round(sci_server_emb_mg, 3),
                      sci_mg_per_token = round(sci_mg_per_token, 3), sci_low = round(sci_low_mg_per_token, 2),
                      sci_high = round(sci_high_mg_per_token, 2), total_g = round(total_g, 1)), "sci")

# Figures referenced in results.tex.
# Only the column-width paper_*.png versions are used in the report.
unlink(list.files(REPORT_FIG_DIR, pattern = "^(RQ|SCI|paper_).*\\.png$", full.names = TRUE))
REPORT_FIGURES <- list.files(PLOTS_DIR, pattern = "^paper_.*\\.png$")
invisible(file.copy(file.path(PLOTS_DIR, REPORT_FIGURES), REPORT_FIG_DIR, overwrite = TRUE))

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
cat("Paper tables: ", normalizePath(OUTPUT_DIR, mustWork = FALSE), "/paper_table_*.csv | figures: ", normalizePath(REPORT_FIG_DIR, mustWork = FALSE), "\n", sep = "")
