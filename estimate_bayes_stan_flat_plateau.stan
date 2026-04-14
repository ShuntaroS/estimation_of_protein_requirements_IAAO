data {
  int<lower=0> N;                   // number of observations
  int<lower=0> Npat;                // number of individuals
  real<lower=0> y[N];               // outcome data
  real<lower=0> protein[N];         // protein data
  int<lower=1, upper=Npat> id[N];   // individual id for each observation

  real<lower=0> betakp_lower;       // lower bound for mean change point
  real<lower=0> betakp_upper;       // upper bound for mean change point

  int<lower=0> Npred;               // number of predicted observations
  int<lower=0> Npat_pred;           // number of individuals to predict
  real<lower=0> protein_pred[Npred];// protein for prediction
  int<lower=1, upper=Npat> id_pred[Npred]; // id for each predicted observation
}

parameters {
  // fixed effects
  vector[2] beta;        // beta[1]=level at knot, beta[2]=slope before

  // betakp を無拘束化
  real betakp_raw;       // unconstrained

  // random effects (non-centered)
  vector<lower=0>[3] u_sd;           // sds of random effects: level, slope_before, knot
  cholesky_factor_corr[3] L_u_Corr;  // cholesky factor of correlation matrix
  vector[3] z_u[Npat];               // standard normal latent variables

  // residual sd
  real<lower=0> y_sd;
}

transformed parameters {
  real betakp;                 // transformed mean knot
  matrix[3,3] L_u;             // cholesky factor of covariance (sd * corr)
  vector[3] u[Npat];           // random effects in original scale
  vector[3] alpha[Npat];       // individual parameters
  real y_mu[N];

  // betakp: map R -> (lower, upper)
  betakp =
    betakp_lower +
    (betakp_upper - betakp_lower) * inv_logit(betakp_raw);

  // random effects: non-centered
  L_u = diag_pre_multiply(u_sd, L_u_Corr);
  for (i in 1:Npat) {
    u[i] = L_u * z_u[i];
  }

  // apply random effects to get individual parameters
  for (i in 1:Npat) {
    alpha[i,1] = beta[1] + u[i,1];
    alpha[i,2] = beta[2] + u[i,2];
    alpha[i,3] = betakp  + u[i,3];
  }

  // regression equation (continuous piecewise linear, flat plateu after breakpoint)
  for (j in 1:N) {
    if (protein[j] < alpha[id[j],3])
      y_mu[j] = alpha[id[j],1] + alpha[id[j],2] * (protein[j] - alpha[id[j],3]);
    else
      // slope = 0, so just level at knot
      y_mu[j] = alpha[id[j],1];
  }
}

model {
  // priors: fixed effects
  beta[1] ~ normal(0, 10);
  beta[2] ~ normal(0, 10);

  // betakp_raw prior (weakly informative on logit scale)
  betakp_raw ~ normal(0, 1);

  // priors: SDs
  u_sd[1] ~ normal(0, 5);
  u_sd[2] ~ normal(0, 5);
  
  // knot の個体差は特に不安定になりやすいので，やや強めに正則化
  u_sd[3] ~ normal(0, 0.3);

  // residual sd
  y_sd ~ normal(0, 5);

  // correlation prior
  L_u_Corr ~ lkj_corr_cholesky(2);

  // latent z_u
  for (i in 1:Npat)
    z_u[i] ~ normal(0, 1);

  // likelihood
  y ~ normal(y_mu, y_sd);
}

generated quantities {

  real y_rep[N];
  vector[N] log_lik;

  for (n in 1:N) {
    y_rep[n] = normal_rng(y_mu[n], y_sd);
    log_lik[n] = normal_lpdf(y[n] | y_mu[n], y_sd);
  }

  real y_pred[Npred];
  real y_mu_pred[Npred];

  corr_matrix[3] u_Corr;
  matrix[3,3] u_Sigma;

  vector[3] alpha_tosave[Npat_pred];

  // save alpha for plotting predictions
  for (i in 1:Npat_pred)
    alpha_tosave[i] = alpha[i];

  // prediction
  for (j in 1:Npred) {
    if (protein_pred[j] < alpha[id_pred[j],3])
      y_mu_pred[j] = alpha[id_pred[j],1] + alpha[id_pred[j],2] * (protein_pred[j] - alpha[id_pred[j],3]);
    else
      y_mu_pred[j] = alpha[id_pred[j],1];

    y_pred[j] = normal_rng(y_mu_pred[j], y_sd);
  }

  // recover correlation and covariance
  u_Corr = multiply_lower_tri_self_transpose(L_u_Corr);
  u_Sigma = quad_form_diag(u_Corr, u_sd);
}
