library(readr)
library(dplyr)
library(broom)
library(purrr)
library(ggplot2)

#Load data
df <- read_csv("/Users/daviddejonghe/Documents/Psychologie/Masterproef/simulationsHmfc/hmfc_with_symptoms.csv")

#filter outlier
df <- df %>%
  filter(sigmasq < 12) %>%
  mutate(
    sex_num = as.numeric(Sex)
  )

#Standardize predictors
df_reg <- df %>%
  mutate(
    Age_z = as.numeric(scale(Age)),
    AnxiousDepression_z = as.numeric(scale(AnxiousDepression)),
    Compulsivity_z = as.numeric(scale(Compulsivity)),
    SocialWithdrawal_z = as.numeric(scale(SocialWithdrawal))
  )

###############################
# Single regression
###############################

#function to run single pred
run_regression_single <- function(df, outcome, predictor, add_age = TRUE, add_sex = TRUE, quadratic = FALSE) {
  
  X_terms <- predictor
  
  if (quadratic) {
    X_terms <- c(X_terms, paste0("I(", predictor, "^2)"))
  }
  
  if (add_age) X_terms <- c(X_terms, "Age_z")
  if (add_sex) X_terms <- c(X_terms, "sex_num")
  
  formula_text <- paste(outcome, "~", paste(X_terms, collapse = " + "))
  model <- lm(as.formula(formula_text), data = df)
  
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Outcome:", outcome, "\n")
  cat("Predictor:", predictor, "\n")
  cat("Quadratic:", quadratic, "\n")
  cat(strrep("=", 60), "\n", sep = "")
  print(summary(model))
  
  return(model)
}

predictors_all <- c("AnxiousDepression_z", "Compulsivity_z", "SocialWithdrawal_z")

single_models <- list()

for (pred in predictors_all) {
  
  single_models[[paste(pred, "mad_criterion", sep = "_")]] <- run_regression_single(
    df_reg, outcome = "mad_criterion", predictor = pred, quadratic = TRUE
  )
  
  single_models[[paste(pred, "sd_criterion", sep = "_")]] <- run_regression_single(
    df_reg, outcome = "sd_criterion", predictor = pred, quadratic = TRUE
  )
}

extract_single_model_results <- function(single_models) {
  predictor_label_map <- c(
    "AnxiousDepression_z" = "AD",
    "Compulsivity_z" = "CIT",
    "SocialWithdrawal_z" = "SW"
  )
  
  outcome_label_map <- c(
    "a" = "a",
    "sigmasq" = "σ²",
    "mad_criterion" = "MAD criterion",
    "sd_criterion" = "SD criterion"
  )
  
  rows <- list()
  
  for (name in names(single_models)) {
    model <- single_models[[name]]
    td <- tidy(model, conf.int = TRUE)
    
    pred <- names(coef(model))[names(coef(model)) %in% names(predictor_label_map)]
    outcome <- all.vars(formula(model))[1]
    
    rows[[name]] <- td %>%
      filter(term == pred) %>%
      mutate(
        predictor = predictor_label_map[pred],
        outcome = outcome_label_map[outcome]
      ) %>%
      select(predictor, outcome, beta = estimate, ci_low = conf.low, ci_high = conf.high, p = p.value)
  }
  
  bind_rows(rows)
}

#quadratic extraction
extract_quadratic_results <- function(single_models) {
  predictor_label_map <- c(
    "AnxiousDepression_z" = "AD",
    "Compulsivity_z" = "CIT",
    "SocialWithdrawal_z" = "SW"
  )
  
  outcome_label_map <- c(
    "mad_criterion" = "MAD criterion",
    "sd_criterion" = "SD criterion"
  )
  
  rows <- list()
  
  for (name in names(single_models)) {
    model <- single_models[[name]]
    td <- tidy(model, conf.int = TRUE)
    outcome <- all.vars(formula(model))[1]
    
    # bepaal welke predictor in dit model zit
    pred <- names(coef(model))[names(coef(model)) %in% names(predictor_label_map)]
    quad_term <- paste0("I(", pred, "^2)")
    
    rows[[name]] <- td %>%
      filter(term == quad_term) %>%
      mutate(
        predictor = predictor_label_map[pred],
        outcome = outcome_label_map[outcome]
      ) %>%
      select(predictor, outcome, beta = estimate, ci_low = conf.low, ci_high = conf.high, p = p.value)
  }
  
  bind_rows(rows)
}

quad_plot_df <- extract_quadratic_results(single_models)
quad_plot_df

single_plot_df <- extract_single_model_results(single_models)
single_plot_df

# Plot single models
single_plot_df <- single_plot_df %>%
  mutate(
    predictor = factor(predictor, levels = c("AD", "CIT", "SW")),
    outcome = factor(outcome, levels = c("a", "σ²", "SD criterion", "MAD criterion"))
  )

ggplot(single_plot_df, aes(x = predictor, y = beta, fill = outcome)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    position = position_dodge(width = 0.75),
    width = 0.2
  ) +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  geom_text(
    data = subset(single_plot_df, p < .05),
    aes(y = pmin(ci_low - 0.03, -0.03), label = "*"),
    position = position_dodge(width = 0.75),
    vjust = 1,
    size = 6
  ) +
  labs(
    title = "Associations between symptom dimensions and criterion fluctuation parameters",
    x = "Predictor",
    y = "Regression coefficient",
    fill = "Outcome"
  ) +
  theme_minimal(base_size = 12)

#quad plot
quad_plot_df <- quad_plot_df %>%
  mutate(
    predictor = factor(predictor, levels = c("AD", "CIT", "SW")),
    outcome = factor(outcome, levels = c("SD criterion", "MAD criterion"))
  )

ggplot(quad_plot_df, aes(x = predictor, y = beta, fill = outcome)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    position = position_dodge(width = 0.75),
    width = 0.2
  ) +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  geom_text(
    data = subset(quad_plot_df, p < .05),
    aes(y = ifelse(beta < 0, ci_low - 0.01, ci_high + 0.01), label = "*"),
    position = position_dodge(width = 0.75),
    vjust = ifelse(subset(quad_plot_df, p < .05)$beta < 0, 1, 0),
    size = 6
  ) +
  labs(
    title = "Quadratic effects of symptom dimensions on criterion fluctuation parameters",
    x = "Predictor",
    y = "Quadratic regression coefficient",
    fill = "Outcome"
  ) +
  theme_minimal(base_size = 12)

#check social withdrawal
ggplot(df_reg, aes(x = SocialWithdrawal_z, y = mad_criterion)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = TRUE) +
  theme_minimal(base_size = 12) +
  labs(
    x = "Social Withdrawal (z)",
    y = "MAD criterion",
    title = "Quadratic association between Social Withdrawal and MAD criterion"
  )

#Check model fit
m_lin_SW_mad <- run_regression_single(
  df_reg, outcome = "mad_criterion", predictor = "SocialWithdrawal_z", quadratic = FALSE
)

m_quad_SW_mad <- run_regression_single(
  df_reg, outcome = "mad_criterion", predictor = "SocialWithdrawal_z", quadratic = TRUE
)

anova(m_lin_SW_mad, m_quad_SW_mad)
AIC(m_lin_SW_mad, m_quad_SW_mad)