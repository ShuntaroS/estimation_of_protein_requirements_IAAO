# Fig 2: simulation study, (A) bias, (B) RMSE, and (C) coverage of the 95% interval of the
# population-level change-point.
# Run with this folder (code_revision/figures/) as the working directory: Rscript make_fig2_simulation.R
# (simulation_report.qmd also runs this script after writing the input file.)
# Input (written by code_revision/simulation_report.qmd):
#   sim_performance_population.csv  performance for each scenario (sd_cp x n x levels_per_participant)
#                                   and method, 500 replicates each: bias, mcse_bias, rmse, coverage,
#                                   mcse_coverage (proportions), and other measures
# Output: Fig2_simulation.tiff (600 dpi, LZW) and Fig2_simulation.pdf in this folder.

library(readr)
library(dplyr)
library(ggplot2)
library(patchwork)

# methods to show (CSV name = label in the figure), in the order of the legend
methods <- c("Model 2" = "Model 2", "Model 3" = "Model 3", "Tang (FO)" = "Tang model",
             "Model 4 (HN 0.3)" = "Model 4")
model_colors <- c("Model 2" = "#4269d0", "Model 3" = "#6cc5b0", "Tang model" = "#d4a017", "Model 4" = "#ff725c")
design_levels <- c("n = 8, 7 levels", "n = 8, 4 of 7 levels", "n = 16, 7 levels", "n = 16, 4 of 7 levels")
dodge <- position_dodge(width = 0.6)

perf <- read_csv("sim_performance_population.csv", show_col_types = FALSE) |>
  filter(method %in% names(methods)) |>
  mutate(model = factor(methods[method], levels = methods),
         design = factor(sprintf("n = %d, %s", n, ifelse(levels_per_participant == 7, "7 levels", "4 of 7 levels")),
                         levels = design_levels),
         sd_cp = factor(sd_cp))

# one panel; error bars are 95% Monte Carlo intervals (value +/- 1.96 MCSE) when `se` is given
# (without `se`, the bars have zero length so that all panels share one legend)
panel <- function(data, y, se = NULL, ref = NULL, ylab, strips = FALSE) {
  data <- mutate(data, value = .data[[y]], half = if (is.null(se)) 0 else 1.96 * .data[[se]])
  p <- ggplot(data, aes(sd_cp, value, colour = model))
  if (!is.null(ref)) p <- p + geom_hline(yintercept = ref, linetype = "dashed", colour = "grey50")
  p +
    geom_pointrange(aes(ymin = value - half, ymax = value + half), position = dodge, size = 0.25) +
    facet_wrap(~ design, nrow = 1) +
    scale_colour_manual(values = model_colors, name = NULL) +
    labs(x = NULL, y = ylab) +
    theme_bw(base_size = 10, base_family = "Arial") +
    theme(panel.grid.minor = element_blank(), strip.background = element_blank(),
          strip.text = if (strips) element_text() else element_blank())
}

p_bias <- panel(perf, "bias", se = "mcse_bias", ref = 0, ylab = "Bias\n(g/kg/day)", strips = TRUE)
p_rmse <- panel(perf, "rmse", ylab = "RMSE") + expand_limits(y = 0)
p_cov <- panel(mutate(perf, coverage = 100 * coverage, mcse_coverage = 100 * mcse_coverage),
               "coverage", se = "mcse_coverage", ref = 95, ylab = "Coverage of 95%\ninterval (%)") +
  labs(x = "SD of participant-specific change-points (g/kg/day)")

fig <- (p_bias / p_rmse / p_cov) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "A", tag_prefix = "(", tag_suffix = ")") &
  theme(legend.position = "bottom", plot.tag = element_text(face = "bold"))

ggsave("Fig2_simulation.tiff", fig, width = 7.5, height = 6.5, units = "in", dpi = 600,
       device = ragg::agg_tiff, compression = "lzw")
ggsave("Fig2_simulation.pdf", fig, width = 7.5, height = 6.5, units = "in", device = cairo_pdf)
