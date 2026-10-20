# Change-point estimation of protein requirements with the IAAO method: code and data

This repository contains the data and R code for the following article:

> Sato S, Hatamoto Y, Matsumoto M. Change-Point Estimation of Protein Requirements Allowing for Inter-Individual Variation Using the Indicator Amino Acid Oxidation Method: A Bayesian Random Change-Point Model. *PLOS ONE* (under review).

The code compares models for estimating protein requirements (the change-point at which <sup>13</sup>CO<sub>2</sub> excretion stops decreasing) from indicator amino acid oxidation (IAAO) data:

- **Model 2**: mixed-effects piecewise regression with a random intercept and a change-point common to all participants (REML; the change-point is chosen by AIC over candidate values from 0 to 2.5 g/kg/day, with an AIC-based 95% interval).
- **Model 3**: as Model 2, with a random intercept and a random slope.
- **Tang model**: nonlinear mixed-effects model in which only the change-point varies among participants (Tang et al., 2014), estimated by maximum likelihood with the first-order approximation, with a Wald 95% interval.
- **Model 4** (proposed): Bayesian mixed-effects model in which the intercept, slope, and change-point vary among participants, estimated with Stan.

The models are evaluated in a simulation study and applied to data extracted from four published IAAO studies.

## Contents

| Path | Description |
|---|---|
| `data/ds_4studies.rds`, `data/ds_4studies.csv` | Data extracted from the figures of four published IAAO studies (same content in both files) |
| `functions.R` | Functions for Models 2 and 3, the Tang model, and Model 4 |
| `model4_random_cp.stan` | Stan code for Model 4 |
| `analysis_R1.qmd`, `analysis_R1.html` | Analysis of the four datasets and its report |
| `simulation/` | Simulation study: data generation and analysis of one replicate (`sim_functions.R`), runners (`run_simulation.R`, `run_all.sh`), saved results (`results/`), numbers quoted in the article (`text_numbers.R`), and the fits used to choose the true values (`extract_truth_candidates.R`, `truth_candidates.csv`) |
| `simulation_report.qmd`, `simulation_report.html` | Summary of the simulation study and its report |
| `figures/make_fig2_simulation.R`, `figures/make_fig3_actual_data.R` | Scripts for Figs 2 and 3, called by the two reports |

## Data

`ds_4studies` has one row per observation, with the columns `manuscript` (study), `id` (participant within the study), `protein` (protein intake, g/kg/day), and `outcome` (<sup>13</sup>CO<sub>2</sub> excretion as plotted in the original figure). The values were extracted from the figures of the original articles by two independent researchers, and discrepancies were resolved by discussion.

| Study | Participants | Observations |
|---|---|---|
| Rafii et al., J Nutr 2015;145:18-24 | 12 | 71 (82 reported; some points overlapped in the figure) |
| Packer et al., Front Nutr 2017;4:64 | 7 | 47 (the figure shows 47 points, although the text reports 45 trials) |
| Mao et al., J Nutr 2020;150:1208-13 | 14 | 74 |
| Wu et al., Br J Nutr 2024;131:1377-83 | 16 | 106 |

Please cite the original articles when using these data.

## Requirements

The analyses were run with R 4.6.1 and Quarto 1.8.27 on macOS (Apple M4 Max). R packages: tidyverse 2.0.0, nlme 3.1-169, rstan 2.32.7, posterior 1.7.0, numDeriv 2016.8-1.1, gt 1.3.0, patchwork 1.3.2, ragg 1.5.2, and jsonlite 2.0.0. rstan requires a C++ toolchain; see the rstan installation guide. The fonts in the figures are Arial.

## How to reproduce the results

Run the following commands in the root folder of this repository.

1. **Analysis of the four datasets** (about 5 minutes):

   ```
   quarto render analysis_R1.qmd
   ```

   This writes Table 2, the sensitivity analysis of the prior for the SD of the random change-point, and other tables to `output/`, and Fig 3 and S1 Fig to `figures/`.

2. **Simulation study** (optional; about 5 hours with 14 cores). The results are saved in `simulation/results/`, and `run_simulation.R` skips scenarios whose result files already exist. To run the simulation from scratch, remove or move the files in `simulation/results/` and run:

   ```
   bash simulation/run_all.sh 14
   ```

   The argument is the number of cores.

3. **Summary of the simulation study** (about 1 minute):

   ```
   quarto render simulation_report.qmd
   ```

   This writes the performance measures to `output/` and Fig 2 to `figures/`.

4. **Numbers quoted in the article** (after step 3):

   ```
   Rscript simulation/text_numbers.R output/text_numbers.json
   ```

The random seeds of the data generation and of Stan are fixed, so the results are reproducible on the same platform; small differences may occur on other platforms or compilers.

## Correspondence to the article

| Article | Source |
|---|---|
| Fig 2 | `figures/make_fig2_simulation.R` (data: `figures/sim_performance_population.csv`, written by `simulation_report.qmd`) |
| Fig 3 | `figures/make_fig3_actual_data.R` (data: `figures/fig3_*.csv`, written by `analysis_R1.qmd`) |
| Table 2 | `output/table2_revised.csv` and `output/table2_full.csv` for Models 2 to 4, and `output/table_tang_published_data.csv` (rows `FO`) for the Tang model (`analysis_R1.qmd`) |
| S1 Fig | `figures/S1_Fig.pdf` (`analysis_R1.qmd`) |
| S1 Table | `output/simulation_population.csv`, `output/simulation_individual.csv`, and `output/simulation_population_converged.csv` (`simulation_report.qmd`) |
| S2 Table | `output/table_sensitivity.csv` (`analysis_R1.qmd`) |

## Changes from the version at submission

The version of this repository at submission (release v1.0) contains `analyses_4studies_2506.qmd`. The main changes in this version are:

1. **Model 4**: the bounds of the population-level change-point, (0, 2.5) g/kg/day, are now passed to Stan; in the submitted code, these arguments were dropped and the default bounds, (0.5, 2.0), were used. The scale of the half-normal prior for the SD of the random change-point is now an argument (0.3 in the primary analysis and 1 in the sensitivity analysis). Posterior distributions are summarized by medians and 95% equal-tailed credible intervals.
2. **Models 2 and 3**: the candidate change-points now range from 0 to 2.5 g/kg/day, the same range as in Model 4 (submitted: 0.5 to 2.5). The estimates and intervals for the four datasets are unchanged.
3. **Tang model and simulation study**: added.
4. **Figures**: in the figure of the fitted lines, the submitted code drew only the first six participants of each study because of the default palette of point shapes; all participants are now shown.

## License

The code is released under the MIT License (see `LICENSE`). The data were extracted from figures in the published articles listed above.

## Contact

Shuntaro Sato (shuntaro.sato.stat@gmail.com)
