library(tidymodels)
library(censored)
library(multilevelmod)
library(lme4)
library(workflowsets)

tidymodels_prefer()
theme_set(theme_minimal())


# Preparing data ----------------------------------------------------------

data(ames, package = "modeldata")

ggplot(ames, aes(x = Sale_Price)) + 
  geom_histogram(bins = 50, col= "white") +
  scale_x_log10()

ames <- ames |> mutate(Sale_Price = log10(Sale_Price))


# Splitting data ----------------------------------------------------------

set.seed(1234)

ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)


# Fitting models ----------------------------------------------------------

lm_fit <- linear_reg() |> 
  set_engine("lm") |> 
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

lm_fit

lm_fit |> extract_fit_engine()
lm_fit |> extract_fit_engine() |> vcov()
lm_fit |> extract_fit_engine() |> summary()

lm_coef <- lm_fit |> extract_fit_engine() |> summary() |> coef()

broom::tidy(lm_fit)

ames_test_small <- ames_test |> slice(1:5)

ames_test_small |> 
  select(Sale_Price) |> 
  bind_cols(predict(lm_fit, ames_test_small)) |> 
  bind_cols(predict(lm_fit, ames_test_small, type = "pred_int"))

tree_fit <- decision_tree(min_n = 2) |> 
  set_engine("rpart") |> 
  set_mode("regression") |> 
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

ames_test_small %>% 
  select(Sale_Price) %>% 
  bind_cols(predict(tree_fit, ames_test_small))


# Using workflow ----------------------------------------------------------

lm_mdl <- linear_reg() |> 
  set_engine("lm") |> 
  translate()

lm_wfl <- workflow() |> 
  add_model(lm_mdl) |> 
  add_formula(Sale_Price ~ Longitude + Latitude)

lm_fit <- fit(lm_wfl, ames_train)
predict(lm_fit, ames_test_small)

lm_wfl <- workflow() |> 
  add_model(lm_mdl) |> 
  add_variables(outcomes = Sale_Price, predictors = c(Longitude, Latitude))

lm_fit <- fit(lm_wfl, ames_train)
predict(lm_fit, ames_test_small)

multilevel_spec <- linear_reg() |> set_engine("lmer")
multilevel_wfl <- workflow() |> 
  add_variables(outcome = distance, predictors = c(Sex, age, Subject)) |> 
  add_model(multilevel_spec, formula = distance ~ Sex + (age | Subject))
multilevel_fit <- fit(multilevel_wfl, data = nlme::Orthodont)
multilevel_fit

parametric_spec <- survival_reg()
parametric_wfl <- workflow() |> 
  add_variables(outcome = c(fustat, futime), predictors = c(age, rx)) |>
  add_model(parametric_spec, formula = Surv(futime, fustat) ~ age + strata(rx))
parametric_fit <- fit(parametric_wfl, data = ovarian)
parametric_fit


# Multiple workflow sets --------------------------------------------------

lm_mdl <- linear_reg() |> set_engine("lm")

lm_formulas <- list(
  "longitude" = Sale_Price ~ Longitude,
  "latitude" = Sale_Price ~ Latitude,
  "coordinates" = Sale_Price ~ Longitude + Latitude,
  "neighborhood" = Sale_Price ~ Neighborhood
)

lm_workflows <- workflow_set(preproc = lm_formulas, models = list(lm = lm_mdl))
lm_workflows$info
extract_workflow(lm_workflows, id = "longitude_lm")


lm_models <- lm_workflows |> 
  mutate(fit = map(info, ~fit(.$workflow[[1]], ames_train)))

lm_models$fit[[3]]


# Evaluation --------------------------------------------------------------

final_lm_res <- last_fit(lm_wfl, ames_split)

final_lm_res
extract_workflow(final_lm_res)
collect_metrics(final_lm_res)
collect_predictions(final_lm_res)


# Feature engineering -----------------------------------------------------

simple_ames <- recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built +
                        Bldg_Type + Latitude + Longitude, data = ames_train) |> 
  step_log(Gr_Liv_Area, base = 10) |> 
  step_other(Neighborhood, threshold = 0.01) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) |> 
  step_ns(c(Latitude, Longitude), deg_free = 20) |> 
  step_normalize(matches("(_SF$|^Gr_Liv_)")) |> 
  step_pca(matches("(_SF$|^Gr_Liv_)"))

stringr::str_subset(names(ames), "(_SF$|^Gr_Liv_)")
simple_ames
tidy(simple_ames)

lm_wfl <- workflow() |> 
  add_model(lm_mdl) |> 
  add_recipe(simple_ames)
lm_wfl

lm_fit <- fit(lm_wfl, ames_train)
predict(lm_fit, ames_test %>% slice(1:3))

extract_fit_engine(lm_fit) |> tidy()
extract_fit_parsnip(lm_fit)
extract_recipe(lm_fit, estimated = TRUE) |> tidy()


# Model evaluation --------------------------------------------------------

ames_test_res <- predict(lm_fit, ames_test) |> 
  bind_cols(ames_test |> select(Sale_Price))
ames_test_res

ggplot(ames_test_res, aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = .3) +
  geom_abline(lty = 2) +
  coord_obs_pred()

rmse(ames_test_res, truth = Sale_Price, estimate = .pred)

ames_metrics <- metric_set(rmse, mae, rsq)
ames_metrics(ames_test_res, truth = Sale_Price, estimate = .pred)

# classification settings
data("two_class_example")
two_class_example

conf_mat(two_class_example, truth = truth, estimate = predicted)
accuracy(two_class_example, truth = truth, estimate = predicted)
mcc(two_class_example, truth = truth, estimate = predicted)
f_meas(two_class_example, truth = truth, estimate = predicted)

classification_metrics <- metric_set(accuracy, mcc, f_meas)
classification_metrics(two_class_example, truth = truth, estimate = predicted)

classification_curve <- roc_curve(two_class_example, truth, Class1)
roc_auc(classification_curve)

autoplot(classification_curve)

f_meas(hpc_cv, truth = obs, estimate = pred, estimator = "macro_weighted")
classification_metrics(hpc_cv, truth = obs, estimate = pred, estimator = "macro")
classification_metrics(hpc_cv, truth = obs, estimate = pred, estimator = "micro")

roc_auc(hpc_cv, obs, c("VF", "F", "M", "L"))

hpc_cv |> 
  group_by(Resample) |> 
  accuracy(obs, pred)

hpc_cv %>% 
  group_by(Resample) %>% 
  roc_curve(obs, VF, F, M, L) %>% 
  autoplot()
