# Posterior medians of the nuisance parameters in the four datasets (primary setting),
# used to choose realistic data-generating values for the simulation study
source("functions.R")
rstan_options(auto_write = TRUE); options(mc.cores = 4)
ds <- load_ds("data/ds_4studies.rds")
dbm <- ds |> group_nest(manuscript) |> mutate(data = map(data, ~ mutate(.x, id = fct_drop(id))))
sm <- stan_model("model4_random_cp.stan")
out <- map2_dfr(dbm$data, dbm$manuscript, function(d, ms) {
  r <- run_model4(sm, d)
  e <- rstan::extract(r$fit, pars = c("beta", "betakp", "u_sd", "y_sd", "u_Sigma"))
  cor <- apply(e$u_Sigma, 1, function(S) cov2cor(S)[upper.tri(S)])
  tibble(manuscript = ms,
         plateau = median(e$beta[, 1]), slope = median(e$beta[, 2]), cp = median(e$betakp),
         sd_int = median(e$u_sd[, 1]), sd_slope = median(e$u_sd[, 2]), sd_cp = median(e$u_sd[, 3]),
         sd_resid = median(e$y_sd),
         cor_int_slope = median(cor[1, ]), cor_int_cp = median(cor[2, ]), cor_slope_cp = median(cor[3, ]),
         drop = median(-e$beta[, 2] * (e$betakp - min(d$protein))))
})
options(width = 200)
print(out |> mutate(across(where(is.numeric), ~ round(.x, 3))) |> as.data.frame())
write_csv(out, "simulation/truth_candidates.csv")
