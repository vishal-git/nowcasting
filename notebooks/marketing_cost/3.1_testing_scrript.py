# Script use for training and validating models
# here we have multiple variable that we can change and quickly do the experimentation on 4 models
# get results in expected format

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xg
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

# create results folder if not exists
import os
if not os.path.exists('results'):
    os.makedirs('results')

def get_model_results(model_obj, X_validation, y_valid, model_name=None):
    """
    Compute a suite of regression metrics and return both a summary dict and a detailed DataFrame.

    Parameters
    ----------
    model_obj : trained regression model
    X_validation : pd.DataFrame
        Features for validation
    y_valid : pd.Series
        True target values
    model_name : str, optional
        Identifier for the model (defaults to class name)

    Returns
    -------
    summary : dict
        Dictionary containing metrics and model identifier
    results_df : pd.DataFrame
        DataFrame with columns ['Predicted', 'Actual', 'Error', 'Absolute_Percent_Error']
    """
    # Generate predictions
    y_pred = model_obj.predict(X_validation)

    # Calculate core metrics
    mae = mean_absolute_error(y_valid, y_pred)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    evs = explained_variance_score(y_valid, y_pred)

    # Print detailed performance
    print(f"\nModel: {model_name or type(model_obj).__name__}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Explained Variance: {evs:.4f}")

    # Compile summary dictionary
    summary = {
        'model_name': model_name or type(model_obj).__name__,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'explained_variance': evs
    }

    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'Predicted': y_pred,
        'Actual': y_valid.values
    }, index=y_valid.index)
    results_df['Error'] = results_df['Actual'] - results_df['Predicted']
    results_df['Absolute_Percent_Error'] = (results_df['Error'].abs() / results_df['Actual']) * 100

    # Clean column names
    results_df.columns = ['Predicted', 'Actual', 'Error', 'Absolute_Percent_Error']

    return summary, results_df


if __name__ == '__main__':
    # Configuration
    EXPERIMENT_NAME = 'GradientBoost_model_104_weeks_backward_forward'
    DATA_PATH = 'Data_for_taining_14072025.csv'
    TARGET = 'marketing_cost'
    FEATURES = [
        'number_orders',
 'avg_temp',
 'is_season_sale_event',
 'cr_tracked_%',
 'number_days_after_last_event',
 'internal_Week_of_SS_Season',
 'available_stock_value_after_discount_complete_eur',
 'aov_eur',
 'is_Peak_Driving_Public_Holiday_week',
 'stock_discount_rate_total_%',
 'is_percentage_on_top_applicable',
 'is_percentage_on_top',
 'is_black_week_event',
 'number_days_till_next_event',
 'internalWeeks_until_SeasonalSaleStart',
 'cpc',
 'is_Sun_to_Mon_Shift_week',
 'email_recipients',
 'is_temp_drop_flag',
 'number_visits'
    ]

    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df = df.iloc[1:-1]  # drop first and last week

    number_of_weeks_to_be_consider = 104

    # Validation setup: last 8 weeks
    val = df.iloc[-8:]
    val_start = val.index[0]

    # 1 year training window before validation
    train_1y = df.loc[val_start - pd.DateOffset(weeks=number_of_weeks_to_be_consider) : val_start - pd.DateOffset(weeks=1)]

    X_train = train_1y[FEATURES]
    y_train = train_1y[TARGET]
    X_valid = val[FEATURES]
    y_valid = val[TARGET]

    print('Training data range', X_train.index.min(), X_train.index.max())
    print('Validation data range', X_valid.index.min(), X_valid.index.max())

    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42, min_samples_split=0.04, n_estimators= 300, max_depth= 7),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, min_samples_split=0.04, n_estimators= 300, max_depth= 4, learning_rate=0.2),
        'XGBoost': xg.XGBRegressor(random_state=42, seed=123, min_samples_split=0.04, n_estimators= 300, max_depth= 7)
    }

    # Containers for results
    summary_list = []
    details_dict = {}

    # Train, evaluate, and collect results
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print('Training accuracy',model.score(X_train, y_train))
        print('Validation accuracy',model.score(X_valid, y_valid))
        summary, detail_df = get_model_results(model, X_valid, y_valid, model_name=name)
        summary_list.append(summary)
        details_dict[name] = detail_df

    # Save summary to CSV
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(f'results/model_performance_summary_{EXPERIMENT_NAME}.csv', index=False)
    print("\nSaved model_performance_summary.csv")

    # Save detailed validation results for each model
    for name, df_detail in details_dict.items():
        filename = f"{name}_validation_results_{EXPERIMENT_NAME}.csv"
        df_detail.to_csv(f'results/{filename}')
        print(f"Saved {filename}")

    # === Additional summary CSV with per-date errors ===
    dates = [d.strftime('%Y-%m-%d') for d in val.index]
    rows = []
    for summary in summary_list:
        name = summary['model_name']
        # parameters and features (static)
        params = 'Default'
        feats = 'All'
        # Data period
        data_period = f"data range\n{train_1y.index.min().strftime('%Y-%m-%d')}\n{val.index.max().strftime('%Y-%m-%d')}"
        # Gather errors for each validation date
        pct_errors = details_dict[name]['Absolute_Percent_Error'].values
        mean_err = np.mean(pct_errors)
        row = {
            'Name': name,
            'Data': data_period,
            'Parameters': params,
            'Features': feats,
            'MAE': summary['mae'],
            'RMSE': summary['rmse'],
            'R-Squared': summary['r2'],
            'MAPE': summary['mape']
        }
        # add each date column
        for idx, d in enumerate(dates):
            row[d] = pct_errors[idx]
        row['Mean of error of dates'] = mean_err
        rows.append(row)

    extra_df = pd.DataFrame(rows)
    # Order columns
    extra_cols = ['Name', 'Data', 'Parameters', 'Features', 'MAE', 'RMSE', 'R-Squared', 'MAPE'] + dates + ['Mean of error of dates']
    extra_df = extra_df[extra_cols]
    extra_df.to_csv(f'results/detailed_model_summary_{number_of_weeks_to_be_consider}_{EXPERIMENT_NAME}.csv', index=False)
    print(f"Saved detailed_model_summary_{number_of_weeks_to_be_consider}.csv")