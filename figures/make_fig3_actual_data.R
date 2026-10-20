# Fig 3 (Fig 2 at submission): observed data and population-level fits of Models 2-4 and the Tang model.
# Run with this folder (code_revision/figures/) as the working directory: Rscript make_fig3_actual_data.R
# (analysis_R1.qmd also runs this script after writing the input files.)
# Input (written by code_revision/analysis_R1.qmd):
#   fig3_observed.csv      observed data (manuscript, id, protein, outcome)
#   fig3_curves.csv        population-level fitted lines (manuscript, model, protein, fit)
#   fig3_changepoints.csv  population-level change-points (manuscript, model, cp, lower, upper);
#                          Models 2/3: AIC-based 95% CI, Tang model: Wald 95% CI,
#                          Model 4: posterior median and 95% credible interval
# Output: Fig3_actual_data.tiff (600 dpi, LZW) and Fig3_actual_data.pdf in this folder.

library(readr)
library(dplyr)
library(ggplot2)

models <- c("Model 2", "Model 3", "Tang model", "Model 4")
model_colors <- c("Model 2" = "#4269d0", "Model 3" = "#6cc5b0", "Tang model" = "#d4a017", "Model 4" = "#ff725c")
study_labels <- c(
  "a: Rafii, 2015"  = "(A) Rafii et al. (2015)",
  "b: Packer, 2017" = "(B) Packer et al. (2017)",
  "c: Mao, 2020"    = "(C) Mao et al. (2020)",
  "d: Wu, 2024"     = "(D) Wu et al. (2024)"
)

# point shapes for participants 1-16 within each study (up to 16 participants per study)
participant_shapes <- c(16, 17, 15, 18, 1, 2, 0, 5, 6, 3, 4, 8, 7, 10, 12, 13)

obs <- read_csv("fig3_observed.csv", show_col_types = FALSE) |>
  mutate(id = factor(id, levels = as.character(seq_along(participant_shapes))))
curves <- read_csv("fig3_curves.csv", show_col_types = FALSE) |> mutate(model = factor(model, levels = models))
cps <- read_csv("fig3_changepoints.csv", show_col_types = FALSE) |> mutate(model = factor(model, levels = models))

fig <- ggplot() +
  geom_point(data = obs, aes(protein, outcome, shape = id), colour = "grey40", size = 1.5, alpha = 0.8) +
  geom_line(data = curves, aes(protein, fit, colour = model), linewidth = 0.8) +
  geom_vline(data = cps, aes(xintercept = cp, colour = model), linetype = "dashed", linewidth = 0.5,
             show.legend = FALSE) +
  scale_colour_manual(values = model_colors, name = NULL) +
  scale_shape_manual(values = participant_shapes, guide = "none") +
  scale_x_continuous(breaks = seq(0, 2.5, by = 0.5)) +
  facet_wrap(~ manuscript, scales = "free_y", labeller = as_labeller(study_labels)) +
  expand_limits(y = 0) +
  labs(x = "Protein intake (g/kg/day)", y = expression({}^{13} * CO[2])) +
  theme_bw(base_size = 11, base_family = "Arial") +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom",
        strip.background = element_blank(), strip.text = element_text(size = 12, hjust = 0, face = "bold"))

# same size as the submitted Figure 2 (7 x 6 in)
ggsave("Fig3_actual_data.tiff", fig, width = 7, height = 6, units = "in", dpi = 600,
       device = ragg::agg_tiff, compression = "lzw")
ggsave("Fig3_actual_data.pdf", fig, width = 7, height = 6, units = "in", device = cairo_pdf)
