# Usage (from code_revision/):
#   Rscript simulation/run_simulation.R <rep_from> <rep_to> <n_cores> [scenario ids ...]
# Results are saved per scenario and replicate range, so batches can be added later.
args <- commandArgs(trailingOnly = TRUE)
rep_from <- as.integer(args[1])
rep_to <- as.integer(args[2])
n_cores <- as.integer(args[3])
scen_ids <- if (length(args) > 3) as.integer(args[-(1:3)]) else NULL

source("simulation/sim_functions.R")
rstan_options(auto_write = TRUE)
sm <- stan_model("model4_random_cp.stan")

todo <- if (is.null(scen_ids)) scenarios else filter(scenarios, scenario %in% scen_ids)
log_file <- "simulation/results/progress.log"

for (k in seq_len(nrow(todo))) {
  sc <- todo[k, ]
  out_file <- sprintf("simulation/results/scenario_%02d_reps%03d-%03d.rds", sc$scenario, rep_from, rep_to)
  if (file.exists(out_file)) next
  t0 <- Sys.time()
  res <- parallel::mclapply(rep_from:rep_to, function(r) {
    options(mc.cores = 1)  # chains run sequentially within each worker
    seed <- sc$scenario * 100000 + r
    out <- tryCatch(
      run_replicate(sm, sc$sd_cp, sc$n, sc$levels_per_participant, seed),
      error = function(e) list(error = conditionMessage(e))
    )
    out$rep <- r
    out$seed <- seed
    out
  }, mc.cores = n_cores, mc.preschedule = FALSE)
  saveRDS(list(scenario = sc, reps = rep_from:rep_to, results = res), out_file)
  cat(sprintf("%s scenario %d reps %d-%d done (%.1f min)\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
              sc$scenario, rep_from, rep_to, as.numeric(difftime(Sys.time(), t0, units = "mins"))),
      file = log_file, append = TRUE)
}
