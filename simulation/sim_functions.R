# Simulation study for the R1 revision (PONE-D-26-20307)
#
# ADEMP (Morris et al., Stat Med 2019;38:2074-102)
#   Aims      : performance of Models 2-4 for the population change-point (EAR), and of
#               Model 4 for the SD of change-points and participant-specific change-points
#   Data      : y_ij = b0 + u0_i + (b1 + u1_i) * min(x_ij - cp_i, 0) + e_ij,
#               cp_i = cp + u3_i, u3_i ~ N(0, sd_cp^2); values based on the fits to the
#               four datasets (Wu-like scale); intakes 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8 g/kg/day
#   Estimands : cp (population change-point), sd_cp, cp_i
#   Methods   : Model 2 and Model 3 as in the main analysis (REML profile over candidate
#               change-points 0-2.5 g/kg/day, AIC-based interval),
#               Model 4 as in the main analysis (population change-point bounded to (0, 2.5)
#               with a Normal(0, 1) prior on the logit scale) with half-normal(0, 0.3) and
#               half-normal(0, 1) priors for sd_cp; random change-point model of Tang et al. (2014)
#               (only the change-point is random; maximum likelihood with the exact likelihood, "ML",
#               and with the first-order approximation, "FO", as in Mao et al. 2020)
#   Performance: bias, empirical SE, RMSE, coverage and width of 95% intervals (with MCSE)

source("functions.R")

truth <- list(
  plateau = 1.3, slope = -0.8, cp = 0.9,
  sd_int = 0.15, sd_slope = 0.15, sd_resid = 0.15
)
intake_levels <- c(0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8)

scenarios <- expand_grid(
  sd_cp = c(0, 0.1, 0.2),
  n = c(8, 16),
  levels_per_participant = c(7, 4)
) |>
  mutate(scenario = row_number(), .before = 1)

simulate_data <- function(sd_cp, n, levels_per_participant, seed) {
  set.seed(seed)
  cp_i <- truth$cp + rnorm(n, 0, sd_cp)
  u0 <- rnorm(n, 0, truth$sd_int)
  u1 <- rnorm(n, 0, truth$sd_slope)
  d <- map_dfr(seq_len(n), function(i) {
    x <- if (levels_per_participant >= length(intake_levels)) intake_levels
         else sort(sample(intake_levels, levels_per_participant))
    mu <- truth$plateau + u0[i] + (truth$slope + u1[i]) * pmin(x - cp_i[i], 0)
    tibble(id = i, protein = x, outcome = mu + rnorm(length(x), 0, truth$sd_resid))
  }) |>
    mutate(id = factor(id))
  list(data = d, cp_i = tibble(id = as.character(seq_len(n)), true_cp = cp_i))
}

# Model 4 with both priors for sd_cp. The population change-point is summarised by the posterior
# median (est), as in the analysis of the published data; the posterior mean (est_mean) is kept
# for checking.
fit_model4_sim <- function(sm, d, cp_i, seed, iter = 2000, warmup = 1000) {
  pop <- list()
  ind <- list()
  for (psd in c(0.3, 1)) {
    m <- if (psd == 0.3) "Model 4 (HN 0.3)" else "Model 4 (HN 1)"
    r <- tryCatch(
      suppressWarnings(run_model4(sm, d, prior_sd_cp = psd, lower = 0, upper = 2.5,
                                  seed = seed, iter = iter, warmup = warmup, step = 0.5)),
      error = function(e) NULL)
    if (is.null(r)) {
      pop[[m]] <- tibble(method = m, est = NA_real_, lower = NA_real_, upper = NA_real_, secs = NA_real_)
      next
    }
    e <- rstan::extract(r$fit, pars = c("betakp", "u_sd", "alpha_tosave"))
    cp <- as.vector(e$betakp)
    sdcp <- e$u_sd[, 3]
    diag <- posterior::summarise_draws(
      posterior::as_draws_array(as.array(r$fit, pars = c("betakp", "u_sd"))), "rhat")
    pop[[m]] <- tibble(
      method = m, est = median(cp), est_mean = mean(cp),
      lower = quantile(cp, 0.025, names = FALSE), upper = quantile(cp, 0.975, names = FALSE),
      sd_cp_est = median(sdcp),
      sd_cp_lower = quantile(sdcp, 0.025, names = FALSE),
      sd_cp_upper = quantile(sdcp, 0.975, names = FALSE),
      max_rhat = max(diag$rhat, na.rm = TRUE),
      n_divergent = rstan::get_num_divergent(r$fit),
      secs = r$secs
    )
    ii <- individual_model4(r)
    ind[[m]] <- ii |>
      left_join(cp_i, by = "id") |>
      transmute(id, true_cp, method = m, est = cp_median, lower = cp_lower, upper = cp_upper)
    rm(r, e)
  }
  list(pop = pop, ind = ind)
}

# One replicate: fit all methods and return population- and participant-level results
run_replicate <- function(sm, sd_cp, n, levels_per_participant, seed,
                          grid_step = 0.01, iter = 2000, warmup = 1000) {
  sim <- simulate_data(sd_cp, n, levels_per_participant, seed)
  d <- sim$data
  grid <- seq(0, 2.5, by = grid_step)

  pop <- list()
  ind <- list()

  # Models 2 and 3
  for (m in c("Model 2", "Model 3")) {
    t0 <- Sys.time()
    r <- tryCatch(profile_mixed_cp(d, random_slope = (m == "Model 3"), method = "REML", grid = grid),
                  error = function(e) NULL)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (is.null(r) || length(r$cp) == 0) {
      pop[[m]] <- tibble(method = m, est = NA_real_, lower = NA_real_, upper = NA_real_, secs = secs)
    } else {
      pop[[m]] <- tibble(method = m, est = r$cp, lower = r$lower, upper = r$upper, secs = secs)
      # the common change-point is used as every participant's change-point
      ind[[m]] <- sim$cp_i |> mutate(method = m, est = r$cp, lower = NA_real_, upper = NA_real_)
    }
  }

  # Model 4
  m4 <- fit_model4_sim(sm, d, sim$cp_i, seed, iter = iter, warmup = warmup)
  pop <- c(pop, m4$pop)
  ind <- c(ind, m4$ind)

  # Random change-point model of Tang et al. (2014)
  for (ap in c("ML", "FO")) {
    m <- sprintf("Tang (%s)", ap)
    t0 <- Sys.time()
    r <- tryCatch(fit_tang(d, ap, individual = (ap == "ML")), error = function(e) NULL)
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (is.null(r)) {
      pop[[m]] <- tibble(method = m, est = NA_real_, lower = NA_real_, upper = NA_real_, secs = secs)
      next
    }
    pop[[m]] <- r$pop |> transmute(method = m, est, lower, upper, sd_cp_est = tau, secs = secs,
                                   tau_at_bound, mu_at_bound)
    if (!is.null(r$ind)) {
      ind[[m]] <- r$ind |> left_join(sim$cp_i, by = "id") |>
        transmute(id, true_cp, method = m, est = cp_median, lower = cp_lower, upper = cp_upper)
    }
  }

  # intake levels below and above each participant's true change-point
  n_below_true <- d |>
    mutate(id = as.character(id)) |>
    left_join(sim$cp_i, by = "id") |>
    group_by(id) |>
    summarise(n_below_true = sum(protein < true_cp), n_above_true = sum(protein >= true_cp), .groups = "drop")

  list(
    pop = bind_rows(pop),
    ind = bind_rows(ind) |> left_join(n_below_true, by = "id")
  )
}
