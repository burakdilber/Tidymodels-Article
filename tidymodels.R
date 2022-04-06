library(tidymodels)

library(rio)
ENB2012_data <- import(file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')


###------------------------recipes---------------##
##recipes

## heating load

enb_recipe_hl <- 
  recipe(Y1 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = ENB2012_data) %>%
  step_sqrt(all_predictors())
enb_recipe_hl

## cooling load
enb_recipe_cl <- 
  recipe(Y2 ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = ENB2012_data) %>%
  step_sqrt(all_predictors())
enb_recipe_cl


##rsample
set.seed(123)

enb_split <- initial_split(ENB2012_data, prop = 0.75)
enb_split

enb_train <- training(enb_split)
enb_test  <- testing(enb_split)

enb_cv <- vfold_cv(enb_train, v = 10)


### ------------------------ Linear Regression ----------------------------

## Heating Load

linear_model <- 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

set.seed(123)
linear_wf <-
  workflow() %>%
  add_model(linear_model) %>% 
  add_recipe(enb_recipe_hl)
linear_wf

linear_results <-
  linear_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
            )

linear_results %>%
  collect_metrics()

param_final <- linear_results %>%
  select_best(metric = "rmse")
param_final

linear_wf <- linear_wf %>%
  finalize_workflow(param_final)
linear_wf

linear_fit <- linear_wf %>%
  last_fit(enb_split)

test_performance <- linear_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


## Cooling Load

linear_model <- 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

set.seed(123)
linear_wf <-
  workflow() %>%
  add_model(linear_model) %>% 
  add_recipe(enb_recipe_cl)
linear_wf

linear_results <-
  linear_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

linear_results %>%
  collect_metrics()

param_final <- linear_results %>%
  select_best(metric = "rmse")
param_final

linear_wf <- linear_wf %>%
  finalize_workflow(param_final)
linear_wf

linear_fit <- linear_wf %>%
  last_fit(enb_split)

test_performance <- linear_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)

### ------------------------ K - Nearest Neighbor ----------------------------

## Heating Load

knn_model <- 
  nearest_neighbor( neighbors = tune(),
                    weight_func = tune(),
                    dist_power = tune()
  ) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

set.seed(123)
knn_wf <-
  workflow() %>%
  add_model(knn_model) %>% 
  add_recipe(enb_recipe_hl)
knn_wf

knn_results <-
  knn_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

knn_results %>%
  collect_metrics()

param_final <- knn_results %>%
  select_best(metric = "rmse")
param_final

knn_wf <- knn_wf %>%
  finalize_workflow(param_final)
knn_wf

knn_fit <- knn_wf %>%
  last_fit(enb_split)

test_performance <- knn_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


## Cooling Load

knn_model <- 
  nearest_neighbor( neighbors = tune(),
                    weight_func = tune(),
                    dist_power = tune()
  ) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

set.seed(123)
knn_wf <-
  workflow() %>%
  add_model(knn_model) %>% 
  add_recipe(enb_recipe_cl)
knn_wf

knn_results <-
  knn_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

knn_results %>%
  collect_metrics()

param_final <- knn_results %>%
  select_best(metric = "rmse")
param_final

knn_wf <- knn_wf %>%
  finalize_workflow(param_final)
knn_wf

knn_fit <- knn_wf %>%
  last_fit(enb_split)

test_performance <- knn_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)

### ------------------------ Support Vector Machine ----------------------------

## Heating Load

svm_model <- 
  svm_rbf(  cost = tune(),
            rbf_sigma = tune(),
            margin = tune()
  ) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

set.seed(123)
svm_wf <-
  workflow() %>%
  add_model(svm_model) %>% 
  add_recipe(enb_recipe_hl)
svm_wf

svm_results <-
  svm_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

svm_results %>%
  collect_metrics()

param_final <- svm_results %>%
  select_best(metric = "rmse")
param_final

svm_wf <- svm_wf %>%
  finalize_workflow(param_final)
svm_wf

svm_fit <- svm_wf %>%
  last_fit(enb_split)

test_performance <- svm_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


## Cooling Load

svm_model <- 
  svm_rbf(  cost = tune(),
            rbf_sigma = tune(),
            margin = tune()
  ) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

set.seed(123)
svm_wf <-
  workflow() %>%
  add_model(svm_model) %>% 
  add_recipe(enb_recipe_cl)
svm_wf

svm_results <-
  svm_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

svm_results %>%
  collect_metrics()

param_final <- svm_results %>%
  select_best(metric = "rmse")
param_final

svm_wf <- svm_wf %>%
  finalize_workflow(param_final)
svm_wf

svm_fit <- svm_wf %>%
  last_fit(enb_split)

test_performance <- svm_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)


### ------------------------ Decision Trees ----------------------------

## Heating Load

decision_model <- 
  decision_tree( tree_depth = tune(),
                 min_n = tune(),
                 cost_complexity = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

set.seed(123)
decision_wf <-
  workflow() %>%
  add_model(decision_model) %>% 
  add_recipe(enb_recipe_hl)
decision_wf

decision_results <-
  decision_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

decision_results %>%
  collect_metrics()

param_final <- decision_results %>%
  select_best(metric = "rmse")
param_final

decision_wf <- decision_wf %>%
  finalize_workflow(param_final)
decision_wf

decision_fit <- decision_wf %>%
  last_fit(enb_split)

test_performance <- decision_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


## Cooling Load

decision_model <- 
  decision_tree( tree_depth = tune(),
                 min_n = tune(),
                 cost_complexity = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

set.seed(123)
decision_wf <-
  workflow() %>%
  add_model(decision_model) %>% 
  add_recipe(enb_recipe_cl)
decision_wf

decision_results <-
  decision_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

decision_results %>%
  collect_metrics()

param_final <- decision_results %>%
  select_best(metric = "rmse")
param_final

decision_wf <- decision_wf %>%
  finalize_workflow(param_final)
decision_wf

decision_fit <- decision_wf %>%
  last_fit(enb_split)

test_performance <- decision_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)


### ------------------------ Random Forest ----------------------------

## Heating Load

rf_model <- 
  rand_forest( mtry = tune(),
               trees = tune(),
               min_n = tune()
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

set.seed(123)
rf_wf <-
  workflow() %>%
  add_model(rf_model) %>% 
  add_recipe(enb_recipe_hl)
rf_wf

rf_results <-
  rf_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

rf_results %>%
  collect_metrics()

param_final <- rf_results %>%
  select_best(metric = "rmse")
param_final

rf_wf <- rf_wf %>%
  finalize_workflow(param_final)
rf_wf

rf_fit <- rf_wf %>%
  last_fit(enb_split)

test_performance <- rf_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


## Cooling Load

rf_model <- 
  rand_forest( mtry = tune(),
               trees = tune(),
               min_n = tune()
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

set.seed(123)
rf_wf <-
  workflow() %>%
  add_model(rf_model) %>% 
  add_recipe(enb_recipe_cl)
rf_wf

rf_results <-
  rf_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

rf_results %>%
  collect_metrics()

param_final <- rf_results %>%
  select_best(metric = "rmse")
param_final

rf_wf <- rf_wf %>%
  finalize_workflow(param_final)
rf_wf

rf_fit <- rf_wf %>%
  last_fit(enb_split)

test_performance <- rf_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)


### ------------------------ XGBoost ----------------------------

## Heating Load

boost_model <- 
  boost_tree( tree_depth = tune(),
              trees = tune(),
              mtry = tune(),
              min_n = tune(),
              sample_size = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

set.seed(123)
boost_wf <-
  workflow() %>%
  add_model(boost_model) %>% 
  add_recipe(enb_recipe_hl)
boost_wf

boost_results <-
  boost_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

boost_results %>%
  collect_metrics()

param_final <- boost_results %>%
  select_best(metric = "rmse")
param_final

boost_wf <- boost_wf %>%
  finalize_workflow(param_final)
boost_wf

boost_fit <- boost_wf %>%
  last_fit(enb_split)

test_performance <- boost_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


ggplot(test_performance, aes(x=Y1, y=.pred)) + 
  geom_point(alpha = 0.7) +
  geom_abline(col = "blue", lty = 2) +
  coord_obs_pred() +
  labs(title = "Isıtma Yükleri", x = "Gerçek Değerler", y = "Tahminlenen Değerler") +
  theme_bw() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_rect(colour = "black", size=2))

## Cooling Load

boost_model <- 
  boost_tree( tree_depth = tune(),
              trees = tune(),
              mtry = tune(),
              min_n = tune(),
              sample_size = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

set.seed(123)
boost_wf <-
  workflow() %>%
  add_model(boost_model) %>% 
  add_recipe(enb_recipe_cl)
boost_wf

boost_results <-
  boost_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

boost_results %>%
  collect_metrics()

param_final <- boost_results %>%
  select_best(metric = "rmse")
param_final

boost_wf <- boost_wf %>%
  finalize_workflow(param_final)
boost_wf

boost_fit <- boost_wf %>%
  last_fit(enb_split)

test_performance <- boost_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)

ggplot(test_performance, aes(x=Y2, y=.pred)) + 
  geom_point(alpha = 0.7) +
  geom_abline(col = "blue", lty = 2) +
  coord_obs_pred() +
  labs(title = "Soğutma Yükleri", x = "Gerçek Değerler", y = "Tahminlenen Değerler") +
  theme_bw() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_rect(colour = "black", size=2))


### ------------------------ Artificial Neural Networks ----------------------------

## Heating Load

nn_model <- 
  mlp( hidden_units = tune(),
       penalty = tune(),
       epochs = tune()
  ) %>% 
  set_engine("nnet") %>% 
  set_mode("regression")

set.seed(123)
nn_wf <-
  workflow() %>%
  add_model(nn_model) %>% 
  add_recipe(enb_recipe_hl)
nn_wf

nn_results <-
  nn_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

nn_results %>%
  collect_metrics()

param_final <- nn_results %>%
  select_best(metric = "rmse")
param_final

nn_wf <- nn_wf %>%
  finalize_workflow(param_final)
nn_wf

nn_fit <- nn_wf %>%
  last_fit(enb_split)

test_performance <- nn_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y1, estimate = .pred)


## Cooling Load

nn_model <- 
  mlp( hidden_units = tune(),
       penalty = tune(),
       epochs = tune()
  ) %>% 
  set_engine("nnet") %>% 
  set_mode("regression")

set.seed(123)
nn_wf <-
  workflow() %>%
  add_model(nn_model) %>% 
  add_recipe(enb_recipe_cl)
nn_wf

nn_results <-
  nn_wf %>% 
  tune_grid(resamples = enb_cv,
            metrics = metric_set(rmse, rsq, mae)
  )

nn_results %>%
  collect_metrics()

param_final <- nn_results %>%
  select_best(metric = "rmse")
param_final

nn_wf <- nn_wf %>%
  finalize_workflow(param_final)
nn_wf

nn_fit <- nn_wf %>%
  last_fit(enb_split)

test_performance <- nn_fit %>% collect_predictions()
test_performance

enb_metrics <- metric_set(rmse, rsq, mae)
enb_metrics(data = test_performance, truth = Y2, estimate = .pred)

