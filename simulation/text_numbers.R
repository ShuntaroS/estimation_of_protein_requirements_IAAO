# Numbers quoted in the manuscript and the response letter, computed from the tables written by
# simulation_report.qmd (render it first). The measures follow Section 2.3 of the manuscript.
# Usage (from code_revision/): Rscript simulation/text_numbers.R <out.json>
suppressMessages(library(tidyverse))
out <- commandArgs(trailingOnly = TRUE)[1]

P <- read_csv("output/simulation_population.csv", show_col_types = FALSE)
I <- read_csv("output/simulation_individual.csv", show_col_types = FALSE)
PC <- read_csv("output/simulation_population_converged.csv", show_col_types = FALSE) |>
  left_join(P |> select(scenario, method, bias0 = bias, cov0 = coverage), by = c("scenario", "method"))

m4 <- "Model 4 (HN 0.3)"
tang <- "Tang (FO)"
num <- function(x, d = 2) {
  v <- sprintf(paste0("%.", d, "f"), x)
  v <- sub("^-(0\\.0+)$", "\\1", v)  # no minus sign for values that round to zero
  sub("^-", "−", v)
}
rng <- function(x, d = 2) { a <- num(min(x), d); b <- num(max(x), d); if (a == b) a else paste(a, "to", b) }
pct <- function(x) sprintf("%.1f%%", 100 * x)
prng <- function(x) { a <- pct(min(x)); b <- pct(max(x)); if (a == b) a else paste(a, "to", b) }
sel <- function(d, s, m) d |> filter(sd_cp %in% s, method %in% m)
all_sd <- c(0, 0.1, 0.2)

# RMSE of Model 4 compared with the other models shown in the figure (Models 2 and 3, Tang model)
w <- P |> filter(method %in% c("Model 2", "Model 3", tang, m4)) |>
  select(scenario, sd_cp, method, rmse) |> pivot_wider(names_from = method, values_from = rmse) |>
  mutate(other = pmin(`Model 2`, `Model 3`, .data[[tang]]), best = .data[[m4]] < other)
n_best <- sum(w$best[w$sd_cp > 0])
hn <- P |> filter(str_detect(method, "Model 4")) |> select(scenario, method, bias, coverage) |>
  pivot_wider(names_from = method, values_from = c(bias, coverage))
sp <- P |> filter(sd_cp == 0, n == 8, levels_per_participant == 4)
s0_other <- w |> filter(sd_cp == 0) |> semi_join(P |> filter(!(n == 8 & levels_per_participant == 4)), by = "scenario")
i11 <- I |> filter(sd_cp == 0.2, n == 16, levels_per_participant == 7)
i_lev <- I |> filter(method == m4, sd_cp > 0) |> group_by(levels_per_participant) |> summarise(rmse = mean(rmse))

res <- list(
  reps = as.character(min(P$n_rep)),
  # population-level change-point: coverage
  m4_cov_all = prng(sel(P, all_sd, m4)$coverage),
  s0_m4_cov = prng(sel(P, 0, m4)$coverage),
  s12_m4_cov = prng(sel(P, c(0.1, 0.2), m4)$coverage),
  s0_m23_cov = prng(sel(P, 0, c("Model 2", "Model 3"))$coverage),
  s0_sparse_cov_m2 = pct(sp$coverage[sp$method == "Model 2"]),
  s12_m2_cov = prng(sel(P, c(0.1, 0.2), "Model 2")$coverage),
  s12_m3_cov = prng(sel(P, c(0.1, 0.2), "Model 3")$coverage),
  s2_m2_cov = prng(sel(P, 0.2, "Model 2")$coverage),
  s2_m3_cov = prng(sel(P, 0.2, "Model 3")$coverage),
  # population-level change-point: bias
  m4_bias_all = rng(sel(P, all_sd, m4)$bias),
  m2_bias_all = rng(sel(P, all_sd, "Model 2")$bias),
  m3_bias_all = rng(sel(P, all_sd, "Model 3")$bias),
  s0_m23_bias = rng(sel(P, 0, c("Model 2", "Model 3"))$bias),
  s0_m4_bias = rng(sel(P, 0, m4)$bias),
  s1_m23_bias = rng(sel(P, 0.1, c("Model 2", "Model 3"))$bias),
  s1_m4_bias = rng(sel(P, 0.1, m4)$bias),
  s2_m2_bias = rng(sel(P, 0.2, "Model 2")$bias),
  s2_m3_bias = rng(sel(P, 0.2, "Model 3")$bias),
  s2_m4_bias = rng(sel(P, 0.2, m4)$bias),
  # population-level change-point: RMSE
  m4_rmse_all = rng(sel(P, all_sd, m4)$rmse),
  m2_rmse_all = rng(sel(P, all_sd, "Model 2")$rmse),
  m3_rmse_all = rng(sel(P, all_sd, "Model 3")$rmse),
  best_rmse = if (n_best == sum(w$sd_cp > 0)) "in all eight scenarios" else sprintf("in %d of the %d scenarios", n_best, sum(w$sd_cp > 0)),
  n_best = n_best,
  n_best_s0 = sum(w$best[w$sd_cp == 0]),
  s0_sparse_rmse_m2 = num(sp$rmse[sp$method == "Model 2"], 3),
  s0_sparse_rmse_m3 = num(sp$rmse[sp$method == "Model 3"], 3),
  s0_sparse_rmse_m4 = num(sp$rmse[sp$method == m4], 3),
  s0_other_max_drmse = num(max(s0_other$other - s0_other[[m4]]), 3),
  # Tang model (first-order approximation)
  tang_bias_all = rng(sel(P, all_sd, tang)$bias),
  tang_rmse_all = rng(sel(P, all_sd, tang)$rmse),
  tang_cov_all = prng(sel(P, all_sd, tang)$coverage),
  tang_noint = pct(mean(sel(P, all_sd, tang)$no_interval)),
  tang_noint_rng = prng(sel(P, all_sd, tang)$no_interval),
  # Models 2 and 3: intervals
  open_m2 = pct(max(sel(P, all_sd, "Model 2")$open_limit)),
  open_m3 = pct(max(sel(P, all_sd, "Model 3")$open_limit)),
  noest_m2 = pct(max(sel(P, all_sd, "Model 2")$no_estimate)),
  noest_m3 = pct(max(sel(P, all_sd, "Model 3")$no_estimate)),
  # Model 4: relaxed prior and convergence
  hn1_dbias = num(max(abs(hn[[paste0("bias_", m4)]] - hn[["bias_Model 4 (HN 1)"]])), 3),
  hn1_dcov = pct(max(abs(hn[[paste0("coverage_", m4)]] - hn[["coverage_Model 4 (HN 1)"]]))),
  div_primary = pct(max(sel(P, all_sd, m4)$divergent_any)),
  div_relaxed = pct(max(sel(P, all_sd, "Model 4 (HN 1)")$divergent_any)),
  rhat_primary = pct(max(sel(P, all_sd, m4)$rhat_gt_1.01)),
  rhat_relaxed = pct(max(sel(P, all_sd, "Model 4 (HN 1)")$rhat_gt_1.01)),
  conv_dbias = num(max(abs(PC$bias - PC$bias0)), 3),
  conv_dcov = pct(max(abs(PC$coverage - PC$cov0))),
  conv_min_reps = as.character(min(PC$n_rep)),
  # participant-specific change-points
  ind_rmse_m4_s12 = rng(sel(I, c(0.1, 0.2), m4)$rmse),
  ind_rmse_m2_s12 = rng(sel(I, c(0.1, 0.2), "Model 2")$rmse),
  ind_rmse_m3_s12 = rng(sel(I, c(0.1, 0.2), "Model 3")$rmse),
  ind_rmse_m4 = num(i11$rmse[i11$method == m4]),
  ind_rmse_m2 = num(i11$rmse[i11$method == "Model 2"]),
  ind_rmse_m3 = num(i11$rmse[i11$method == "Model 3"]),
  ind_rmse_7 = num(i_lev$rmse[i_lev$levels_per_participant == 7]),
  ind_rmse_4 = num(i_lev$rmse[i_lev$levels_per_participant == 4]),
  ind_cov_s12 = prng(sel(I, c(0.1, 0.2), m4)$coverage),
  ind_cov_s0 = prng(sel(I, 0, m4)$coverage)
)
jsonlite::write_json(res, out, auto_unbox = TRUE, pretty = TRUE)
cat(readLines(out), sep = "\n")
