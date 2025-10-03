import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score, brier_score_loss, log_loss
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.model_selection import RandomizedSearchCV

# Load configuration
with open('/Users/jackweekly/Desktop/NBA/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define paths from config
processed_data_path = config['data_paths']['processed']
figures_path = config['data_paths']['figures']

# --- Determinism ---
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
# For reproducibility with LightGBM, you might also need to set environment variables
# os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
# os.environ['LGBM_ALLOCATOR_RANDOM_SEED'] = str(GLOBAL_SEED)
print(f"Global seed set to: {GLOBAL_SEED}")

# Load data with all engineered features
print("Loading data with all engineered features...")
df = pd.read_parquet(os.path.join(processed_data_path, "games_with_all_features.parquet"))
print("Data loaded.")

# Convert gameDate to datetime and sort for chronological processing
df['gameDate'] = pd.to_datetime(df['gameDate'])
df = df.sort_values(by='gameDate').reset_index(drop=True)

# Define target variables
df['home_win'] = (df['winner'] == df['hometeamId']).astype(int)
df['margin'] = df['homeScore'] - df['awayScore']
df['total'] = df['homeScore'] + df['awayScore']

# Add home-court flag (1 for home team, 0 for away team perspective)
df['is_home'] = 1 # This is from the perspective of the home team in the row

# Calculate league-average home edge (average margin for home teams)
league_avg_home_edge = df['margin'].mean()
df['league_avg_home_edge'] = league_avg_home_edge

# Select features for the model
features = [
    'is_home',
    'league_avg_home_edge',
    # Home team lagged features
    'avg_team_points_scored_last_5',
    'avg_team_points_allowed_last_5',
    'avg_team_points_scored_last_10',
    'avg_team_points_allowed_last_10',
    'days_since_last_game',
    'win_streak',

    # Away team lagged features
    'avg_team_points_scored_last_5_away_lag',
    'avg_team_points_allowed_last_5_away_lag',
    'avg_team_points_scored_last_10_away_lag',
    'avg_team_points_allowed_last_10_away_lag',
    'days_since_last_game_away_lag',
    'win_streak_away_lag',
]

# Drop rows with NaN values in selected features and target for simplicity
df_model = df.dropna(subset=features + ['home_win', 'margin', 'total'])

# --- Consistency Check: Null Audit ---
print("\nRunning null audit on features...")
nan_counts = df_model[features].isnull().sum()
nan_features = nan_counts[nan_counts > 0]

if not nan_features.empty:
    print("WARNING: NaNs found in the following features after dropping rows:")
    print(nan_features)
else:
    print("No NaNs found in the selected features.")


# --- Walk-Forward Validation Setup ---
def run_walk_forward_validation(df_model, features, target_name, model_type='classification', max_folds=None):
    all_predictions = []
    all_true_values = []
    all_probabilities = [] # For classification models
    fold_results = [] # To store metrics for each fold

    # Determine unique years and months for walk-forward splits
    df_model['year_month'] = df_model['gameDate'].dt.to_period('M')
    unique_months = sorted(df_model['year_month'].unique())

    # Define a minimum training period (e.g., first 12 months)
    min_train_months = 2

    if len(unique_months) < min_train_months + 1:
        print("Not enough unique months for walk-forward validation with the specified minimum training period.")
        return None, None, None, None

    for i in range(min_train_months, len(unique_months)):
        if max_folds is not None and (i - min_train_months) >= max_folds:
            break
        train_end_month = unique_months[i-1]
        test_month = unique_months[i]

        train_df = df_model[df_model['year_month'] <= train_end_month]
        test_df = df_model[df_model['year_month'] == test_month]

        if test_df.empty:
            continue

        X_train, y_train = train_df[features], train_df[target_name]
        X_test, y_test = test_df[features], test_df[target_name]

        fold_metrics = {
            'train_end_month': str(train_end_month),
            'test_month': str(test_month),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        if model_type == 'classification':
            # Define parameter distributions for RandomizedSearchCV
            lgbm_param_dist = {
                'num_leaves': [10, 20],
                'max_depth': [3, 5],
                'min_data_in_leaf': [20, 50],
                'feature_fraction': [0.7, 0.9],
                'bagging_fraction': [0.7, 0.9],
                'lambda_l1': [0.1, 1],
                'lambda_l2': [0.1, 1],
                'min_split_gain': [0, 0.1],
                'n_estimators': [10, 20] # Small for quick testing
            }
            rf_param_dist = {
                'n_estimators': [10, 20], # Small for quick testing
                'max_depth': [3, 5],
                'max_features': [0.7, 0.9]
            }

            # LightGBM Classifier with RandomizedSearchCV
            lgbm_model = lgb.LGBMClassifier(random_state=GLOBAL_SEED, n_jobs=-1)
            lgbm_search = RandomizedSearchCV(lgbm_model, lgbm_param_dist, n_iter=3, cv=2, scoring='neg_log_loss', random_state=GLOBAL_SEED, n_jobs=-1)
            lgbm_search.fit(X_train, y_train)
            best_lgbm = lgbm_search.best_estimator_
            lgbm_probs = best_lgbm.predict_proba(X_test)[:, 1]

            # Random Forest Classifier with RandomizedSearchCV
            rf_model = RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=-1)
            rf_search = RandomizedSearchCV(rf_model, rf_param_dist, n_iter=3, cv=2, scoring='neg_log_loss', random_state=GLOBAL_SEED, n_jobs=-1)
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            rf_probs = best_rf.predict_proba(X_test)[:, 1]

            # Simple ensemble: average probabilities
            ensemble_probs = (rf_probs + lgbm_probs) / 2
            ensemble_preds = (ensemble_probs > 0.5).astype(int)

            all_predictions.extend(ensemble_preds)
            all_probabilities.extend(ensemble_probs)

            # Calculate fold metrics for classification
            fold_metrics['accuracy'] = accuracy_score(y_test, ensemble_preds)
            fold_metrics['brier_score'] = brier_score_loss(y_test, ensemble_probs)
            fold_metrics['log_loss'] = log_loss(y_test, ensemble_probs)

        else: # Regression
            # Define parameter distributions for RandomizedSearchCV
            lgbm_param_dist = {
                'num_leaves': [10, 20],
                'max_depth': [3, 5],
                'min_data_in_leaf': [20, 50],
                'feature_fraction': [0.7, 0.9],
                'bagging_fraction': [0.7, 0.9],
                'lambda_l1': [0.1, 1],
                'lambda_l2': [0.1, 1],
                'min_split_gain': [0, 0.1],
                'n_estimators': [10, 20] # Small for quick testing
            }
            rf_param_dist = {
                'n_estimators': [10, 20], # Small for quick testing
                'max_depth': [3, 5],
                'max_features': [0.7, 0.9]
            }

            # LightGBM Regressor with RandomizedSearchCV
            lgbm_model = lgb.LGBMRegressor(random_state=GLOBAL_SEED, n_jobs=-1)
            lgbm_search = RandomizedSearchCV(lgbm_model, lgbm_param_dist, n_iter=3, cv=2, scoring='neg_mean_absolute_error', random_state=GLOBAL_SEED, n_jobs=-1)
            lgbm_search.fit(X_train, y_train)
            best_lgbm = lgbm_search.best_estimator_
            lgbm_preds = best_lgbm.predict(X_test)

            # Random Forest Regressor with RandomizedSearchCV
            rf_model = RandomForestRegressor(random_state=GLOBAL_SEED, n_jobs=-1)
            rf_search = RandomizedSearchCV(rf_model, rf_param_dist, n_iter=3, cv=2, scoring='neg_mean_absolute_error', random_state=GLOBAL_SEED, n_jobs=-1)
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            rf_preds = best_rf.predict(X_test)

            # Simple ensemble: average predictions
            ensemble_preds = (rf_preds + lgbm_preds) / 2
            all_predictions.extend(ensemble_preds)

            # Calculate fold metrics for regression
            fold_metrics['mae'] = mean_absolute_error(y_test, ensemble_preds)
            fold_metrics['r2'] = r2_score(y_test, ensemble_preds)

# --- Model Training for Pregame Home Win Probability ---
print("\nStarting walk-forward validation for Pregame Home Win Probability...")
win_prob_preds, win_prob_true, win_prob_probs, win_prob_fold_results = run_walk_forward_validation(df_model.copy(), features, 'home_win', 'classification', max_folds=3)

if win_prob_preds is not None:
    # Save fold results
    win_prob_fold_results_df = pd.DataFrame(win_prob_fold_results)
    win_prob_fold_results_df.to_parquet(os.path.join(figures_path, "win_prob_backtest_ledger.parquet"), index=False)
    print(f"Win Probability backtest ledger saved to {os.path.join(figures_path, 'win_prob_backtest_ledger.parquet')}")

if win_prob_preds is not None:
    print("\nEvaluating Pregame Home Win Probability Model...")
    accuracy = accuracy_score(win_prob_true, win_prob_preds)
    brier = brier_score_loss(win_prob_true, win_prob_probs)
    ll = log_loss(win_prob_true, win_prob_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {ll:.4f}")

    # Constraint Test: Probabilities in [0,1]
    assert (win_prob_probs >= 0).all() and (win_prob_probs <= 1).all(), "Constraint Violation: Predicted probabilities are not within [0, 1]."
    print("Constraint Test Passed: Predicted probabilities are within [0, 1].")

    # Lift Chart (Top-N Confidence Picks)
    win_prob_results = pd.DataFrame({'predicted_prob': win_prob_probs, 'true_outcome': win_prob_true})
    win_prob_results = win_prob_results.sort_values(by='predicted_prob', ascending=False).reset_index(drop=True)

    print("\nLift Chart (Top-N Confidence Picks for Home Win):")
    top_n_percentages = [0.10, 0.20, 0.30] # Top 10%, 20%, 30% most confident predictions

    for p in top_n_percentages:
        n_picks = int(len(win_prob_results) * p)
        if n_picks == 0: continue
        top_picks = win_prob_results.head(n_picks)
        accuracy_top_picks = accuracy_score(top_picks['true_outcome'], (top_picks['predicted_prob'] > 0.5).astype(int))
        overall_accuracy = accuracy_score(win_prob_results['true_outcome'], (win_prob_results['predicted_prob'] > 0.5).astype(int))
        
        print(f"Top {int(p*100)}% ({n_picks} picks): Accuracy = {accuracy_top_picks:.4f}, Overall Accuracy = {overall_accuracy:.4f}")

    # Reliability Diagram
    prob_true, prob_pred = calibration_curve(win_prob_true, win_prob_probs, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Ensemble Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Win Probability)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, "reliability_diagram_win_prob.png"))
    plt.close()
    print(f"Reliability diagram saved to {os.path.join(figures_path, 'reliability_diagram_win_prob.png')}")

    # Save reliability curve data to CSV
    reliability_df = pd.DataFrame({'mean_predicted_probability': prob_pred, 'fraction_of_positives': prob_true})
    reliability_csv_path = os.path.join(figures_path, "reliability_curve_win_prob.csv")
    reliability_df.to_csv(reliability_csv_path, index=False)
    print(f"Reliability curve data saved to {reliability_csv_path}")
else:
    print("Win Probability walk-forward validation could not be performed.")


# --- Model Training for Pregame Predicted Margin ---
print("\nStarting walk-forward validation for Pregame Predicted Margin...")
margin_preds, margin_true, _, margin_fold_results = run_walk_forward_validation(df_model.copy(), features, 'margin', 'regression', max_folds=3)

if margin_preds is not None:
    # Save fold results
    margin_fold_results_df = pd.DataFrame(margin_fold_results)
    margin_fold_results_df.to_parquet(os.path.join(figures_path, "margin_backtest_ledger.parquet"), index=False)
    print(f"Margin Prediction backtest ledger saved to {os.path.join(figures_path, 'margin_backtest_ledger.parquet')}")

if margin_preds is not None:
    print("\nEvaluating Pregame Predicted Margin Model...")
    mae = mean_absolute_error(margin_true, margin_preds)
    r2 = r2_score(margin_true, margin_preds)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Constraint Test: Margin within +/- 70
    assert (margin_preds >= -70).all() and (margin_preds <= 70).all(), "Constraint Violation: Predicted margins are not within [-70, 70]."
    print("Constraint Test Passed: Predicted margins are within [-70, 70].")

    # MAE by predicted margin magnitude bucket
    margin_results = pd.DataFrame({'predicted_margin': margin_preds, 'true_margin': margin_true})
    margin_results['abs_predicted_margin'] = abs(margin_results['predicted_margin'])
    
    # Define buckets (e.g., 0-5, 5-10, 10-15, 15+)
    bins = [0, 5, 10, 15, np.inf]
    labels = ['0-5', '5-10', '10-15', '15+']
    margin_results['margin_bucket'] = pd.cut(margin_results['abs_predicted_margin'], bins=bins, labels=labels, right=False)

    print("\nMAE by Predicted Margin Magnitude Bucket:")
    mae_by_bucket = margin_results.groupby('margin_bucket').apply(lambda x: mean_absolute_error(x['true_margin'], x['predicted_margin']))
    print(mae_by_bucket)
else:
    print("Margin Prediction walk-forward validation could not be performed.")

# --- Model Training for Pregame Predicted Total ---
print("\nStarting walk-forward validation for Pregame Predicted Total...")
total_preds, total_true, _, total_fold_results = run_walk_forward_validation(df_model.copy(), features, 'total', 'regression', max_folds=3)

if total_preds is not None:
    # Save fold results
    total_fold_results_df = pd.DataFrame(total_fold_results)
    total_fold_results_df.to_parquet(os.path.join(figures_path, "total_backtest_ledger.parquet"), index=False)
    print(f"Total Prediction backtest ledger saved to {os.path.join(figures_path, 'total_backtest_ledger.parquet')}")

if total_preds is not None:
    print("\nEvaluating Pregame Predicted Total Model...")
    mae_total = mean_absolute_error(total_true, total_preds)
    r2_total = r2_score(total_true, total_preds)

    print(f"Mean Absolute Error: {mae_total:.4f}")
    print(f"R-squared: {r2_total:.4f}")
else:
    print("Total Prediction walk-forward validation could not be performed.")
