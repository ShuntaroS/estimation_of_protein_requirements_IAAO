# Functions for the R1 revision analyses (PONE-D-26-20307)
#
# Changes from the submitted code (analyses_4studies_2506.qmd in the version of this repository at submission):
#   * Model 4: the bounds of the population change-point are passed to Stan. In the submitted
#     code, purrr::map2(..., betakp_lower = 0, betakp_upper = 2.5, ~run_stan_model(.x, .y))
#     dropped these arguments, so the defaults (0.5, 2.0) were used.
#   * Model 4: the scale of the half-normal prior for the SD of the random change-point is an
#     argument (primary analysis 0.3; sensitivity analysis 1).
#   * Models 2 and 3: lme() with REML (default) and the AIC-based interval, as submitted. The
#     candidate change-points now span 0-2.5 g/kg/day by 0.01 (submitted: 0.5-2.5), the same
#     range as the bounds of the population change-point in Model 4. Candidates at or below the
#     lowest intake cannot be fitted and are skipped. The estimates and intervals for the four
#     datasets are identical with both ranges.

suppressPackageStartupMessages({
  library(tidyverse)
  library(nlme)
  library(rstan)
  library(posterior)
})

# ---- Data --------------------------------------------------------------------

load_ds <- function(path) {
  read_rds(path) |>
    mutate(
      manuscript = case_when(
        manuscript == "1: Rafii, 2015"  ~ "a: Rafii, 2015",
        manuscript == "2: Packer, 2017" ~ "b: Packer, 2017",
        manuscript == "3: Mao, 2020"    ~ "c: Mao, 2020",
        manuscript == "4: Wu, 2024"     ~ "d: Wu, 2024"
      )
    )
}

# ---- Models 2 and 3: profile over candidate change-points --------------------

# Fits Model 2 (random intercept) or Model 3 (random intercept and slope) for a
# fixed change-point cp.
fit_mixed_at <- function(data, cp, random_slope = FALSE, method = "REML") {
  d <- mutate(data, X = (protein < cp) * (protein - cp))
  if (random_slope) {
    lme(outcome ~ X, random = ~ X | id, data = d, method = method,
        control = lmeControl(opt = "optim"))
  } else {
    lme(outcome ~ X, random = ~ 1 | id, data = d, method = method)
  }
}

# Change-point = grid value with the smallest AIC; interval = grid values closest to the
# minimum on each side whose AIC is at least (minimum AIC + 4), as in the submitted code.
# A limit is NA when the AIC does not reach (minimum AIC + 4) within the grid on that side.
profile_mixed_cp <- function(data, random_slope = FALSE, method = "REML",
                             grid = seq(0, 2.5, by = 0.01)) {
  aic <- map_dbl(grid, function(cp) {
    if (all(data$protein >= cp)) return(NA_real_)
    fit <- tryCatch(fit_mixed_at(data, cp, random_slope, method), error = function(e) NULL)
    if (is.null(fit)) NA_real_ else AIC(fit)
  })
  ok <- !is.na(aic)
  g <- grid[ok]
  a <- aic[ok]
  i <- which.min(a)
  lo <- g[g < g[i] & a >= a[i] + 4]
  hi <- g[g > g[i] & a >= a[i] + 4]
  tibble(
    cp = g[i],
    lower = if (length(lo)) max(lo) else NA_real_,
    upper = if (length(hi)) min(hi) else NA_real_,
    profile = list(tibble(cp = g, AIC = a))
  )
}

predict_mixed <- function(fit, data, cp, step = 0.01) {
  nd <- expand_grid(
    id = factor(levels(data$id), levels = levels(data$id)),
    protein = seq(min(data$protein), max(data$protein), by = step)
  ) |>
    mutate(X = (protein < cp) * (protein - cp))
  pr <- predict(fit, newdata = nd, level = 0:1)
  nd |> mutate(pred_fixed = pr$predict.fixed, pred_ind = pr$predict.id)
}

# ---- Model 4: Bayesian random change-point model -----------------------------

run_model4 <- function(sm, data, prior_sd_cp = 0.3, lower = 0, upper = 2.5,
                       seed = 1, chains = 4, iter = 4000, warmup = 1000,
                       adapt_delta = 0.99, max_treedepth = 20,
                       save_pred = FALSE, step = 0.01) {
  ds_id <- data |> distinct(id) |> mutate(id2 = row_number())
  d <- data |> left_join(ds_id, by = "id") |> select(-id) |> rename(id = id2)
  pred <- expand_grid(
    id_pred = unique(d$id),
    protein_pred = seq(min(d$protein), max(d$protein), by = step)
  )
  stan_data <- list(
    N = nrow(d), Npat = n_distinct(d$id), id = d$id,
    protein = d$protein, y = d$outcome,
    Npred = nrow(pred), Npat_pred = n_distinct(pred$id_pred),
    id_pred = pred$id_pred, protein_pred = pred$protein_pred,
    betakp_lower = lower, betakp_upper = upper,
    prior_sd_cp = prior_sd_cp
  )
  pars <- c("beta", "betakp", "y_sd", "u_sd", "u_Sigma", "alpha_tosave", "lp__")
  if (save_pred) pars <- c(pars, "y_mu_pred")
  t0 <- Sys.time()
  fit <- sampling(
    sm, data = stan_data, pars = pars, seed = seed,
    chains = chains, iter = iter, warmup = warmup,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth),
    refresh = 0
  )
  list(
    fit = fit, data = d, ds_id = ds_id, pred = pred,
    secs = as.numeric(difftime(Sys.time(), t0, units = "secs")),
    settings = list(prior_sd_cp = prior_sd_cp, lower = lower, upper = upper)
  )
}

# Population-level summaries and MCMC diagnostics
summarise_model4 <- function(res) {
  fit <- res$fit
  e <- rstan::extract(fit, pars = c("betakp", "u_sd"))
  arr <- as.array(fit, pars = c("betakp", "u_sd", "beta", "y_sd", "alpha_tosave"))
  diag <- posterior::summarise_draws(posterior::as_draws_array(arr), "rhat", "ess_bulk", "ess_tail")
  cp <- as.vector(e$betakp)
  sd_cp <- e$u_sd[, 3]
  q <- function(x) quantile(x, c(0.025, 0.5, 0.975), names = FALSE)
  tibble(
    cp_mean = mean(cp), cp_median = median(cp), cp_lower = q(cp)[1], cp_upper = q(cp)[3],
    sd_cp_median = median(sd_cp), sd_cp_lower = q(sd_cp)[1], sd_cp_upper = q(sd_cp)[3],
    max_rhat = max(diag$rhat, na.rm = TRUE),
    min_ess_bulk = min(diag$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(diag$ess_tail, na.rm = TRUE),
    n_divergent = rstan::get_num_divergent(fit),
    n_max_treedepth = rstan::get_num_max_treedepth(fit),
    min_bfmi = min(rstan::get_bfmi(fit)),
    secs = res$secs
  )
}

# Participant-specific change-points (posterior median and 95% credible interval)
individual_model4 <- function(res) {
  e <- rstan::extract(res$fit, pars = "alpha_tosave")
  cp <- e$alpha_tosave[, , 3, drop = FALSE][, , 1]
  map_dfr(seq_len(ncol(cp)), function(i) {
    x <- res$data$protein[res$data$id == i]
    tibble(
      id = as.character(res$ds_id$id[res$ds_id$id2 == i]),
      n_obs = length(x), xmin = min(x), xmax = max(x),
      cp_median = median(cp[, i]),
      cp_lower = quantile(cp[, i], 0.025, names = FALSE),
      cp_upper = quantile(cp[, i], 0.975, names = FALSE)
    )
  })
}

# Posterior-median curves (population curve with random effects set to zero, and
# participant-specific curves) for plotting
curves_model4 <- function(res) {
  e <- rstan::extract(res$fit, pars = c("beta", "betakp", "y_mu_pred"))
  xs <- seq(min(res$data$protein), max(res$data$protein), by = 0.01)
  pop <- map_dfr(xs, function(x) {
    mu <- e$beta[, 1] + e$beta[, 2] * pmin(x - as.vector(e$betakp), 0)
    tibble(protein = x, fit = median(mu))
  })
  ind <- res$pred |>
    mutate(fit = apply(e$y_mu_pred, 2, median)) |>
    left_join(res$ds_id, by = c("id_pred" = "id2")) |>
    transmute(id = as.character(id), protein = protein_pred, fit)
  list(population = pop, individual = ind)
}

# ---- Random change-point model of Tang et al. (2014): maximum likelihood ------------
#
# y_ij = b0 + b1 * min(x_ij - cp_i, 0) + e_ij,  cp_i = mu + u_i,  u_i ~ N(0, tau^2),  e_ij ~ N(0, sigma^2)
# Only the change-point varies among participants (intercept and slope are common), as in
# Tang et al. (2014), Mao et al. (2020), and Wu et al. (2024), who fitted this model with SAS PROC
# NLMIXED. Two approximations of the likelihood are available:
#   "ML": exact marginal likelihood. Between consecutive intakes of a participant, the mean function
#         is linear in the change-point, so the integral over the normal random change-point is a sum
#         of truncated normal integrals (closed form);
#   "FO": first-order approximation (the model is linearized in the random effect at u = 0), as
#         used by Mao et al. (2020); it reproduces the published estimates of Mao et al. and Wu et al.
#         from the reconstructed data.
# mu (the EAR) is restricted to (0, 2.5) g/kg/day, the same range as for Models 2-4.
# 95% CI: Wald interval with t quantile, df = number of participants - 1 (PROC NLMIXED default:
# number of subjects minus number of random effects).

tang_nodes <- function(k = 401, zmax = 7) {
  z <- seq(-zmax, zmax, length.out = k)
  list(z = z, logw = dnorm(z, log = TRUE) + log(z[2] - z[1]))
}

# log-likelihood contributions on the grid: participants x nodes
tang_ll_grid <- function(theta, d, nodes) {
  b0 <- theta[1]; b1 <- theta[2]; mu <- theta[3]; tau <- exp(theta[4]); sigma <- exp(theta[5])
  cp <- mu + tau * nodes$z
  f <- b0 + b1 * pmin(outer(d$protein, cp, "-"), 0)
  ll <- dnorm(d$outcome, f, sigma, log = TRUE)  # recycled over columns (obs x nodes)
  rowsum(ll, d$idx, reorder = TRUE)
}

# exact marginal log-likelihood (closed form; see above)
tang_loglik_exact <- function(theta, d) {
  b0 <- theta[1]; b1 <- theta[2]; mu <- theta[3]; tau2 <- exp(2 * theta[4]); s2 <- exp(2 * theta[5])
  sum(vapply(split(seq_len(nrow(d)), d$idx), function(r) {
    o <- order(d$protein[r]); x <- d$protein[r][o]; y <- d$outcome[r][o]; n <- length(x)
    a <- y - b0 - b1 * x                 # residual + b1 * cp for intakes below the change-point
    rr <- (y - b0)^2                     # squared residual for intakes at or above it
    k <- 0:n                             # number of intakes below the change-point
    S1 <- c(0, cumsum(a)); S2 <- c(0, cumsum(a^2)); R <- c(rev(cumsum(rev(rr))), 0)
    A <- k * b1^2 / s2 + 1 / tau2
    B <- b1 * S1 / s2 - mu / tau2
    m <- -B / A
    L <- c(-Inf, x); U <- c(x, Inf)      # change-point in (L, U]
    zu <- (U - m) * sqrt(A); zl <- (L - m) * sqrt(A)
    # log(pnorm(zu) - pnorm(zl)), computed in the tail that is numerically stable
    up <- zl > 0
    ldp <- ifelse(up, pnorm(-zl, log.p = TRUE) + log1p(-exp(pnorm(-zu, log.p = TRUE) - pnorm(-zl, log.p = TRUE))),
                  pnorm(zu, log.p = TRUE) + log1p(-exp(pnorm(zl, log.p = TRUE) - pnorm(zu, log.p = TRUE))))
    lk <- -0.5 * n * log(2 * pi * s2) - (S2 + R) / (2 * s2) - mu^2 / (2 * tau2) + B^2 / (2 * A) -
      0.5 * log(tau2 * A) + ldp
    lk <- lk[is.finite(lk)]
    mx <- max(lk)
    mx + log(sum(exp(lk - mx)))
  }, numeric(1)))
}

tang_loglik <- function(theta, d, nodes) {
  g <- tang_ll_grid(theta, d, nodes)
  a <- sweep(g, 2, nodes$logw, "+")
  m <- apply(a, 1, max)
  sum(m + log(rowSums(exp(a - m))))
}

# first-order approximation: y_i ~ N(f_i(mu), tau^2 z_i z_i' + sigma^2 I), z_i = d f / d cp = -b1 * 1(x < mu)
tang_loglik_fo <- function(theta, d) {
  b0 <- theta[1]; b1 <- theta[2]; mu <- theta[3]; tau <- exp(theta[4]); sigma <- exp(theta[5])
  sum(vapply(split(seq_len(nrow(d)), d$idx), function(r) {
    x <- d$protein[r]; y <- d$outcome[r]
    f <- b0 + b1 * pmin(x - mu, 0)
    z <- -b1 * (x < mu)
    # Sherman-Morrison form of the multivariate normal log-density
    e <- y - f
    a <- tau^2 / sigma^2
    zz <- sum(z^2); ez <- sum(e * z)
    quad <- (sum(e^2) - a * ez^2 / (1 + a * zz)) / sigma^2
    logdet <- length(y) * log(sigma^2) + log1p(a * zz)
    -0.5 * (length(y) * log(2 * pi) + logdet + quad)
  }, numeric(1)))
}

fit_tang <- function(data, approx = c("ML", "FO"), lower = 0, upper = 2.5,
                     starts = c(0.5, 0.8, 1.1, 1.4, 1.8), nodes = tang_nodes(), level = 0.95,
                     individual = TRUE) {
  approx <- match.arg(approx)
  d <- data |> mutate(idx = as.integer(factor(id)))
  n_id <- max(d$idx)
  nll <- function(th) {
    v <- if (approx == "ML") -tang_loglik_exact(th, d) else -tang_loglik_fo(th, d)
    if (is.finite(v)) v else 1e10
  }
  lb <- c(-Inf, -Inf, lower, log(1e-3), log(1e-3))
  ub <- c(Inf, Inf, upper, log(3), log(10))
  fits <- map(starts, function(s) {
    hi <- d$protein >= s
    b0 <- if (any(hi)) mean(d$outcome[hi]) else mean(d$outcome)
    lo <- d$protein < s
    b1 <- if (sum(lo) > 2) unname(coef(lm(outcome ~ protein, data = d[lo, ]))[2]) else -1
    th0 <- c(b0, b1, s, log(0.1), log(sd(d$outcome) / 2))
    tryCatch(optim(th0, nll, method = "L-BFGS-B", lower = lb, upper = ub,
                   control = list(maxit = 5000, factr = 1e2)), error = function(e) NULL)
  })
  fits <- compact(fits)
  if (!length(fits)) return(NULL)
  best <- fits[[which.min(map_dbl(fits, "value"))]]
  th <- best$par
  # Wald SEs from the numerical Hessian (Richardson extrapolation); if log(tau) is at its lower
  # bound or the full Hessian is not positive definite, tau is held fixed
  se_mu <- NA_real_; se_rda <- NA_real_
  at_bound <- th[4] <= lb[4] + 1e-6
  # ML: the exact likelihood is smooth, so Richardson extrapolation is used. FO: the approximate
  # likelihood is not smooth in mu (the linearization changes at each intake), so a finite-difference
  # Hessian with step 1e-3 is used, as in general-purpose software.
  H <- tryCatch(if (approx == "ML") numDeriv::hessian(nll, th) else optimHess(th, nll),
                error = function(e) NULL)
  pd <- function(V) !is.null(V) && all(is.finite(V)) && all(eigen(V, symmetric = TRUE, only.values = TRUE)$values > 0)
  if (!is.null(H)) {
    V <- if (!at_bound) tryCatch(solve(H), error = function(e) NULL) else NULL
    if (pd(V)) {
      se_mu <- sqrt(V[3, 3])
      g <- c(0, 0, 1, 1.96 * exp(th[4]), 0)  # gradient of RDA = mu + 1.96 * tau
      se_rda <- sqrt(drop(t(g) %*% V %*% g))
    } else {
      keep <- c(1, 2, 3, 5)
      V <- tryCatch(solve(H[keep, keep]), error = function(e) NULL)
      if (pd(V)) se_mu <- sqrt(V[3, 3])
    }
  }
  q <- qt(1 - (1 - level) / 2, df = n_id - 1)
  rda <- th[3] + 1.96 * exp(th[4])
  pop <- tibble(est = th[3], lower = th[3] - q * se_mu, upper = th[3] + q * se_mu,
                tau = exp(th[4]), sigma = exp(th[5]), b0 = th[1], b1 = th[2],
                rda = rda, rda_lower = rda - q * se_rda, rda_upper = rda + q * se_rda,
                tau_at_bound = at_bound, mu_at_bound = th[3] <= lower + 1e-6 || th[3] >= upper - 1e-6,
                loglik = -best$value, convergence = best$convergence)
  if (!individual) return(list(pop = pop, ind = NULL))
  # participant-specific change-points: posterior of u_i given the data at the estimates
  g <- sweep(tang_ll_grid(th, d, nodes), 2, nodes$logw, "+")
  post <- exp(g - apply(g, 1, max))
  post <- post / rowSums(post)
  cp_grid <- th[3] + exp(th[4]) * nodes$z
  wq <- function(p, prob) approx(cumsum(p), cp_grid, xout = prob, ties = "ordered", rule = 2)$y
  ids <- levels(factor(d$id))
  ind <- tibble(
    id = as.character(ids),
    cp_median = apply(post, 1, wq, prob = 0.5),
    cp_lower = apply(post, 1, wq, prob = 0.025),
    cp_upper = apply(post, 1, wq, prob = 0.975)
  )
  list(pop = pop, ind = ind)
}
