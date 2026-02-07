# random_forest.R
# Random forest classification and feature importance analysis for autumn olive
# detection across phenological stages. Includes balanced sampling,
# VIF multicollinearity check, hyperparameter tuning with 10-fold CV,
# permutation importance, and SHAP analysis.
#
# Set `stage` below to run for each phenological period:
#   "early", "late", "peak", "senescence"
#
# Inputs: data/train/<stage>/ containing per-flight CSVs with spectral bands and indices
# Outputs: class metrics, permutation importance, SHAP values, PR-AUC

library(tidyverse)
library(caret)
library(randomForest)
library(usdm)
library(MLmetrics)
library(fastshap)
library(future.apply)
library(progressr)
library(doParallel)
library(ROCR)
library(dplyr)


# Data loading and preprocessing

set.seed(23)

stage <- "early"  # "early", "late", "peak", "senescence"

tables_folder <- file.path("data/train", stage)
csv_files <- list.files(path = tables_folder, pattern = "\\.csv$", full.names = TRUE)
cat("# of tables:", length(csv_files), "\n")
cat(paste(csv_files, collapse = "\n"), "\n")

# columns
expected_cols <- c("UniqueID", "ClassType", "ClassCode", "Blue", "Green", "Red", "RedEdge", "NIR", 
                   "NDVI", "NDRE", "GNDVI", "BNDVI", "LCI", "GCI", "RECI", "SRI", "GRNDVI", "OSAVI", 
                   "EVI2", "ReGI", "GRVI", "CVI", "GBVI")

# read and extract a FlightID from the csv and combine
data_list <- lapply(csv_files, function(file) {
  data <- read.csv(file, stringsAsFactors = FALSE)
  
  flight_id <- tools::file_path_sans_ext(basename(file))
  data$FlightID <- flight_id
  
  # combine unique id and flight id
  data$Instance <- paste0(data$UniqueID, "_", flight_id)
  
  missing_cols <- setdiff(expected_cols, names(data))
  if (length(missing_cols) > 0) {
    warning(basename(file), " is missing columns: ", paste(missing_cols, collapse = ", "))
    return(NULL)
  }
  
  data$ClassType <- as.factor(data$ClassType)
  data$ClassCode <- as.factor(data$ClassCode)
  
  return(data)
})
cat_data <- bind_rows(data_list)


# Exploratory data analysis and preprocessing

points_counts <- cat_data %>% 
  dplyr::group_by(FlightID, ClassType) %>% 
  dplyr::summarise(n = n(), .groups = "drop")

ggplot(points_counts, aes(x = ClassType, y = FlightID, fill = n)) +
  geom_tile(color = "white") +
  geom_text(aes(label = n), color = "black") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Counts of Points per Flight per Class",
       x = "class",
       y = "FlightID") +
  theme_minimal() +
  theme(axis.text = element_text(angle = 25, hjust = 1))

cat("\nclass counts\n")
print(table(cat_data$ClassType))
cat("\nmissing values\n")
print(sapply(cat_data, function(x) sum(is.na(x))))

# balanced sampling
# grabbing points from each flight with repeated sampling
target_sample <- 1000

balanced_data <- cat_data %>%
  group_by(FlightID, ClassType) %>%
  slice_sample(n = target_sample, replace = TRUE) %>%
  ungroup()

cat("\nclass counts after balancing\n")
print(table(balanced_data$ClassType))


# Check multicollinearity using VIF

predictor_vars <- c("Blue", "Green", "Red", "RedEdge", "NIR", 
                    "NDVI", "NDRE", "GNDVI", "BNDVI", "LCI", "GCI", "RECI", "SRI", "GRNDVI", "OSAVI", 
                    "EVI2", "ReGI", "GRVI", "CVI", "GBVI")

# extract predictor data
predictor_data1 <- balanced_data %>% dplyr::select(all_of(predictor_vars))

# VIF
vif_values1 <- usdm::vif(predictor_data1)
cat("\nVIF for predictor vars:\n")
print(vif_values1)

vif_df1 <- data.frame(Variables = vif_values1$Variables, VIF = vif_values1$VIF)
vif_plot1 <- ggplot(vif_df1, aes(x = reorder(Variables, VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "VIF for Predictor Variables", x = "Variable", y = "VIF")
print(vif_plot1)

# mean reflectance of color bands plot
color_bands <- balanced_data %>% 
  dplyr::select(ClassType, Blue, Green, Red, RedEdge, NIR)

# pivot the data from wide to long for plotting
color_bands_long <- tidyr::pivot_longer(color_bands, 
                                        cols = c("Blue", "Green", "Red", "RedEdge", "NIR"), 
                                        names_to = "Band", 
                                        values_to = "Value")

color_bands_long <- color_bands_long %>%
  mutate(Band = factor(Band, 
                       levels = c("Blue", "Green", "Red", "RedEdge", "NIR"),
                       labels = c("Blue (460-490 nm)", 
                                  "Green (540-560 nm)", 
                                  "Red (655-685 nm)", 
                                  "RedEdge (710-720 nm)", 
                                  "NIR (830-850 nm)")))

# calc mean reflectance across bands and class
color_bands_summary <- color_bands_long %>%
  group_by(ClassType, Band) %>%
  summarise(MeanValue = mean(Value, na.rm = TRUE), .groups = "drop")

color_band_plot <- ggplot(color_bands_summary, aes(x = Band, y = MeanValue, group = ClassType, color = ClassType)) +
  geom_line(linewidth = 1) +
  labs(title = "Mean Reflectance of Color Bands by Class", 
       x = "Color Band", 
       y = "Mean Reflectance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(color_band_plot)

write.csv(color_bands_summary, file.path("outputs", paste0(stage, "_mean_reflectance_color_bands.csv")), row.names = FALSE)


# Train/test split

set.seed(23)
train_index <- caret::createDataPartition(balanced_data$ClassType, p = 0.7, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data  <- balanced_data[-train_index, ]


# Random forest modeling with hyperparameter tuning and 10-fold CV

train_control <- caret::trainControl(method = "cv", 
                                     number = 10, 
                                     classProbs = TRUE,
                                     summaryFunction = multiClassSummary,
                                     verboseIter = TRUE)

# grid search for number of variable candidates at each node
tune_grid <- expand.grid(mtry = seq(2, length(predictor_vars), by = 1))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(23)
rf_model <- caret::train(
  x = train_data[, predictor_vars],
  y = train_data$ClassType,
  method = "rf",
  metric = "Accuracy",
  tuneGrid = tune_grid,
  trControl = train_control,
  ntree = 500
)

stopCluster(cl)

rf_final <- rf_model$finalModel

cat("\nmodel summary:\n")
print(rf_model)
cat("\nbest mtry:\n")
print(rf_model$bestTune)
plot(rf_model, main = "random forest tuning results (mtry)")


# Variable importance and diagnostic plots from the RF model

var_importance <- caret::varImp(rf_model, scale = FALSE)
cat("\nvar importance for random forest:\n")
print(var_importance)
plot(var_importance, main = "variable importance - RF model")


# Model validation

# predict to test set
predictions <- predict(rf_model, newdata = test_data[, predictor_vars])
conf_matrix <- caret::confusionMatrix(predictions, test_data$ClassType)
print(conf_matrix)

overall_accuracy <- conf_matrix$overall['Accuracy']
cat("\noverall accuracy:", overall_accuracy,"\n")

# calc per class metrics
class_metrics <- data.frame(conf_matrix$byClass)
write.csv(class_metrics, file.path("outputs", paste0(stage, "_class_metrics.csv")), row.names = TRUE)
print(class_metrics)

# calc macro f1 for classes
precision <- class_metrics$Pos.Pred.Value
recall <- class_metrics$Sensitivity
f1_scores <- 2 * (precision * recall) / (precision + recall)
macro_f1 <- mean(f1_scores, na.rm = TRUE)
cat("\nmacro F1 score:", macro_f1,"\n")

cm_df <- as.data.frame(conf_matrix$table)
conf_matrix_plot <- ggplot(cm_df, aes(Prediction, Reference)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq)) +
  labs(title = "Confusion Matrix")
print(conf_matrix_plot)

# PR/AUC

pred_prob <- predict(rf_model, newdata = test_data[, predictor_vars], type = "prob")[, "AutumnOlive"]

# create binary labels for autumn olive (1 = autumn olive, 0 = everything else)
actual <- ifelse(test_data$ClassType == "AutumnOlive", 1, 0)

# ROCR prediction object
pred_obj <- prediction(pred_prob, actual)

# compute precision recall
pr_perf <- performance(pred_obj, "prec", "rec")

pdf(file.path("outputs", paste0(stage, "_PR_AUC_AutumnOlive.pdf")))
plot(pr_perf, colorize = TRUE,
     main = "Precision-Recall Curve for 'AutumnOlive'",
     xlab = "Recall", ylab = "Precision")
dev.off()

# compute PR AUC
pr_auc_perf <- performance(pred_obj, measure = "aucpr")
pr_auc_value <- pr_auc_perf@y.values[[1]]
cat("\nPR AUC for autumnolive:", pr_auc_value, "\n")

pr_auc_df <- data.frame(Class = "AutumnOlive", PR_AUC = pr_auc_value)
write.csv(pr_auc_df, file.path("outputs", paste0(stage, "_pr_auc_AutumnOlive.csv")), row.names = FALSE)


# Permutation importance

class_labels <- levels(train_data$ClassType)

perm_importance_all <- list()

for (class in class_labels) {
  cat("\ncomputing permutation importance for:", class, "\n")
  
  # baseline predict probs for computing class
  baseline_probs <- predict(rf_model, newdata = test_data[, predictor_vars], type = "prob")[, class]
  
  perm_importance <- sapply(predictor_vars, function(feature) {
    test_data_perm <- test_data
    test_data_perm[[feature]] <- sample(test_data_perm[[feature]])
    new_probs <- predict(rf_model, newdata = test_data_perm[, predictor_vars], type = "prob")[, class]
    mean(baseline_probs - new_probs)
  })
  
  perm_importance_all[[class]] <- data.frame(Feature = predictor_vars, Importance = perm_importance, Class = class) %>% 
    arrange(desc(Importance))
}

perm_importance_df_all <- bind_rows(perm_importance_all)

write.csv(perm_importance_df_all, file.path("outputs", paste0(stage, "_perm_importance_all.csv")), row.names = FALSE)

for (class in class_labels) {
  cat("\npermutation importance for", class, ":\n")
  print(perm_importance_all[[class]])
}


# SHAP analysis
# subset data for SHAP because it takes forever
num_class <- nlevels(train_data$ClassType)
points_per_class <- round(10000 / num_class)

shap_subset <- train_data %>%
  group_by(ClassType) %>%
  slice_sample(n = points_per_class, replace = TRUE) %>%
  ungroup()
print(table(shap_subset$ClassType))

set.seed(23)
plan(multisession, workers = detectCores() - 1)
handlers(global = TRUE)

shap_values_all <- list()

for (class in class_labels) {
  cat("\ncomputing SHAP values for:", class, "\n")
  
  # compute SHAP values for x iterations (nsim = 1 per core)
  progressr::with_progress({
    p <- progressr::progressor(along = 1:30)
    
    shap_values_list <- future_lapply(1:30, function(i) { 
      p()
      
      pred_fun <- get("predict.randomForest", envir = asNamespace("randomForest"))
      
      fastshap::explain(
        rf_final, 
        X = shap_subset[, predictor_vars], 
        pred_wrapper = function(model, newdata) {
          as.vector(pred_fun(model, newdata = newdata, type = "prob")[, class])
        },
        nsim = 1
      )
    }, future.seed = TRUE)
  })
  
  # average SHAP values across all iterations
  shap_values <- Reduce("+", shap_values_list) / length(shap_values_list)
  plan(sequential)
  
  # compute global SHAP values for current class
  global_shap <- sapply(predictor_vars, function(v) {
    mean(abs(shap_values[[v]]), na.rm = TRUE)
  })
  
  shap_values_all[[class]] <- data.frame(Feature = predictor_vars, SHAP = global_shap, Class = class) %>%
    arrange(desc(SHAP))
}

shap_importance_df_all <- bind_rows(shap_values_all)
write.csv(shap_importance_df_all, file.path("outputs", paste0(stage, "_global_shap_importance_all.csv")), row.names = FALSE)

for (class in class_labels) {
  cat("\nglobal SHAP importance for", class, ":\n")
  print(shap_values_all[[class]])
}
