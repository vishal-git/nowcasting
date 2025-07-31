# Nowcasting Models for Marketing Cost and COGS

Use the following set of features as potential predictors:
```
'aov_eur',
'available_stock_value_after_discount_complete_eur',
'available_stock_value_after_discount_complete_eur lag(n=7)',
'avg_temp',
'cpc',
'cr_tracked_%',
'cr_tracked_% lag(n=7)',
'demand_sales_eur',
'dgm_expectation_perc',
'email_recipients',
'email_visits',
'internalWeeks_until_SeasonalSaleStart',
'internal_Week_of_FW_Season',
'internal_Week_of_SS_Season',
'is_Peak_Driving_Public_Holiday_week',
'is_Sun_to_Mon_Shift_week',
'is_black_week_event',
'is_email_campaign_type_deal',
'is_email_campaign_type_liveshop',
'is_email_campaign_type_newsletter',
'is_percentage_on_top',
'is_percentage_on_top_applicable',
'is_season_sale_event',
'is_temp_drop_flag',
'marketing_budget',
'number_days_after_last_event',
'number_days_till_next_event',
'number_orders',
'number_visits',
'sku_with_discount_%',
'stock_discount_rate_total_%',
'target_cpr'
```

**Project Title**\
COGS & Marketing Cost Model Retraining

**Table of Contents**

1. [Overview](#overview)
2. [Data Acquisition](#data-acquisition)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling Approach](#modeling-approach)
   - 5.1 [Baseline Models](#baseline-models)
   - 5.2 [Time Frame Experiments](#time-frame-experiments)
6. [Feature Selection](#feature-selection)
   - 6.1 [Forward Selection](#forward-selection)
   - 6.2 [Backward Elimination](#backward-elimination)
   - 6.3 [Combined Selection](#combined-selection)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Automation Script](#automation-script)
9. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

This project implements a robust pipeline to predict Cost of Goods Sold (COGS) and Marketing Cost using a variety of machine learning techniques. The goal is to identify the most predictive features, optimize models, and package the entire workflow into a repeatable script.

Key highlights:

- Comprehensive Exploratory Data Analysis (EDA)
- Multiple modeling algorithms (Linear Regression, Random Forest, Gradient Descent, XGBoost)
- Rigorous feature selection (forward, backward, combined)
- Automated hyperparameter tuning via GridSearchCV
- Modular script for rapid experimentation and comparison

---

## Data Acquisition

1. **Data Sources**

   - Primary dataset containing weekly COGS and Marketing spend
   - Additional external datasets (if any) for enrichment

2. **Initial Checks**

   - Verifying presence of all required columns
   - Ensuring consistent data types and formats

---

## Exploratory Data Analysis (EDA)

1. **Correlation Analysis**

   - Computed pairwise correlations to identify strong linear relationships between features and targets
   - Visualized correlation matrix heatmap for quick overview

2. **Feature Distribution**

   - Plotted histograms and density plots for each numerical feature
   - Boxplots to detect outliers and anomalies

3. **Time-Series Trends**

   - Weekly trend analysis for COGS and Marketing Cost
   - Seasonality and rolling-statistics plots

4. **Missing Value Analysis**

   - Counted and visualized missing values per column

---

## Data Preprocessing

1. **Missing Value Handling**

   - There are some columns has missing values, when we dig deeper into those columns, we found that those are very old data like 2022, so we decide to drop those columns.


2. **Final Dataset**

   - Consolidated cleaned features and target variables into training-ready DataFrame

---

## Modeling Approach

### Baseline Models

We evaluated the following algorithms to establish baselines:

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Descent (SGDRegressor)**
- **XGBoost Regressor**

Each model was trained with default hyperparameters to gauge initial performance.

### Time Frame Experiments

To assess the effect of historical context, we ran experiments on multiple data windows:

- **52 weeks** (1 year)
- **78 weeks** (1.5 years)
- **104 weeks** (2 years)
- **Full dataset**

The window yielding the highest baseline accuracy determined the data scope for subsequent steps.

---

## Feature Selection

To identify the most predictive subset of features, three complementary techniques were applied for each model and time frame:

### Forward Selection

1. Start with no features.
2. Iteratively add the feature that provides the best cross-validated performance boost.
3. Stop when no additional feature improves the model significantly.

### Backward Elimination

1. Start with all features.
2. Iteratively remove the least significant feature (smallest impact on performance).
3. Stop when all remaining features contribute positively.

### Combined Selection

- Combine both forward and backward selection methods.

*Features frozen:* Once the optimal subset is determined for a given model and timeframe, those features are fixed for hyperparameter tuning.

---

## Hyperparameter Tuning

Using **GridSearchCV**, we finely tuned model-specific parameters:

- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **XGBoost**: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`
- **GradientBoostingRegressor**: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`
- **Linear Regression**: no hyperparameters, further experiment with Lasso and Ridge models.

Cross-validation folds and scoring metric (e.g., RMSE) were standardized across experiments. Whichever model is leading, we select it to fine-tune to get the maximum score.

---

## Automation Script

\`\` encapsulates the entire workflow for rapid iteration: `3.1_testing_scrript.py`

- **Configurable Parameters**: model type, timeframe, feature selection method, hyperparameter grid
- **Steps Executed**:
  1. Load and preprocess data
  2. Select features as provided by notebooks
  3. Train model 
  4. Save individual model results and combined summary

*Results Output:*

- CSV files with metrics per experiment
- Save model file as a JOBLIB for deployment

---

## Evaluation Metrics

- Performance metrics: MAPE, RMSE, MAE, R²

---
