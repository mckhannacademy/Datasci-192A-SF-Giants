"""
Evaluation and comparison utilities for regression models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    Dict with RMSE, MAE, and R²
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def compare_models(
    results: Dict[str, Dict[str, float]],
    title: str = "Model Comparison"
) -> pd.DataFrame:
    """
    Create a comparison table of model results.

    Parameters
    ----------
    results : Dict
        Dict of model_name -> metrics dict
    title : str
        Title for the comparison

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    df = pd.DataFrame(results).T
    df.index.name = 'Model'

    print(f"\n{title}")
    print("="*60)
    print(df.to_string())

    return df


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metric_fn : callable
        Function that takes (y_true, y_pred) and returns a scalar
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)

    Returns
    -------
    Tuple of (point_estimate, lower_bound, upper_bound)
    """
    n = len(y_true)
    bootstrapped_metrics = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        metric = metric_fn(y_true[indices], y_pred[indices])
        bootstrapped_metrics.append(metric)

    point_estimate = metric_fn(y_true, y_pred)
    lower = np.percentile(bootstrapped_metrics, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrapped_metrics, (1 + confidence) / 2 * 100)

    return point_estimate, lower, upper


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Create a scatter plot of predictions vs actual values.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()

    # Add metrics annotation
    metrics = compute_metrics(y_true, y_pred)
    text = f"RMSE: {metrics['rmse']:.3f}\nMAE: {metrics['mae']:.3f}\nR²: {metrics['r2']:.3f}"
    ax.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Create a residual plot.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    residuals = y_true - y_pred

    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(title)

    return ax


def plot_coefficient_importance(
    coefs_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Feature Coefficients",
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot top coefficient magnitudes.

    Parameters
    ----------
    coefs_df : pd.DataFrame
        DataFrame with 'feature' and 'coefficient' columns
    top_n : int
        Number of top features to show
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    top_coefs = coefs_df.head(top_n)

    colors = ['green' if c > 0 else 'red' for c in top_coefs['coefficient']]
    ax.barh(top_coefs['feature'], top_coefs['coefficient'], color=colors)
    ax.set_xlabel('Coefficient')
    ax.set_title(title)
    ax.invert_yaxis()

    return ax


def plot_park_embeddings_2d(
    embeddings_df: pd.DataFrame,
    title: str = "Park Embeddings (PCA Projection)",
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot 2D PCA projection of park embeddings.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        Park embeddings with park names as index
    title : str
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # PCA projection
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_df.values)

    # Plot
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)

    # Add park labels
    for i, park in enumerate(embeddings_df.index):
        ax.annotate(park, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   fontsize=9, ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(title)

    return ax


def compute_park_similarity(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between park embeddings.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        Park embeddings with park names as index

    Returns
    -------
    pd.DataFrame
        Similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(embeddings_df.values)
    return pd.DataFrame(similarity, index=embeddings_df.index, columns=embeddings_df.index)


def find_similar_parks(
    embeddings_df: pd.DataFrame,
    park: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Find most similar parks based on embeddings.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        Park embeddings
    park : str
        Target park
    top_n : int
        Number of similar parks to return

    Returns
    -------
    pd.DataFrame
        Similar parks with similarity scores
    """
    similarity_matrix = compute_park_similarity(embeddings_df)
    similarities = similarity_matrix[park].sort_values(ascending=False)

    # Exclude the park itself
    similarities = similarities[similarities.index != park]

    return similarities.head(top_n).to_frame(name='similarity')


def create_full_comparison_report(
    ridge_k_model,
    ridge_runs_model,
    lasso_k_model,
    lasso_runs_model,
    nn_k_trainer,
    nn_runs_trainer,
    splits: Dict,
    nn_data: Dict,
    output_dir: str = None
):
    """
    Create a comprehensive comparison report of all models.

    Parameters
    ----------
    ridge_k_model : ParkWeatherRegressor
        Ridge model for strikeouts
    ridge_runs_model : ParkWeatherRegressor
        Ridge model for runs
    lasso_k_model : ParkWeatherRegressor
        Lasso model for strikeouts
    lasso_runs_model : ParkWeatherRegressor
        Lasso model for runs
    nn_k_trainer : EmbeddingModelTrainer
        NN model for strikeouts
    nn_runs_trainer : EmbeddingModelTrainer
        NN model for runs
    splits : Dict
        Data splits from data_prep
    nn_data : Dict
        NN data from data_prep
    output_dir : str, optional
        Directory to save plots
    """
    # Collect metrics for strikeouts
    k_results = {
        'Ridge': ridge_k_model.evaluate(splits['X_test'], splits['y_test_strikeouts']),
        'Lasso': lasso_k_model.evaluate(splits['X_test'], splits['y_test_strikeouts']),
        'Neural Net': nn_k_trainer.evaluate(
            nn_data['X_weather_test'],
            nn_data['park_ids_test'],
            nn_data['y_test_strikeouts']
        )
    }

    # Collect metrics for runs
    runs_results = {
        'Ridge': ridge_runs_model.evaluate(splits['X_test'], splits['y_test_runs']),
        'Lasso': lasso_runs_model.evaluate(splits['X_test'], splits['y_test_runs']),
        'Neural Net': nn_runs_trainer.evaluate(
            nn_data['X_weather_test'],
            nn_data['park_ids_test'],
            nn_data['y_test_runs']
        )
    }

    # Print comparison tables
    print("\n" + "="*60)
    compare_models(k_results, "STRIKEOUTS Model Comparison (Test Set)")
    compare_models(runs_results, "RUNS Model Comparison (Test Set)")

    # Create visualization figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

    # Strikeouts predictions
    y_true_k = splits['y_test_strikeouts']
    ridge_pred_k = ridge_k_model.predict(splits['X_test'])
    lasso_pred_k = lasso_k_model.predict(splits['X_test'])
    nn_pred_k = nn_k_trainer.predict(nn_data['X_weather_test'], nn_data['park_ids_test'])

    plot_predictions_vs_actual(y_true_k, ridge_pred_k, "Ridge: Strikeouts", axes[0, 0])
    plot_predictions_vs_actual(y_true_k, lasso_pred_k, "Lasso: Strikeouts", axes[0, 1])
    plot_predictions_vs_actual(y_true_k, nn_pred_k, "NN: Strikeouts", axes[0, 2])

    # Runs predictions
    y_true_runs = splits['y_test_runs']
    ridge_pred_runs = ridge_runs_model.predict(splits['X_test'])
    lasso_pred_runs = lasso_runs_model.predict(splits['X_test'])
    nn_pred_runs = nn_runs_trainer.predict(nn_data['X_weather_test'], nn_data['park_ids_test'])

    plot_predictions_vs_actual(y_true_runs, ridge_pred_runs, "Ridge: Runs", axes[1, 0])
    plot_predictions_vs_actual(y_true_runs, lasso_pred_runs, "Lasso: Runs", axes[1, 1])
    plot_predictions_vs_actual(y_true_runs, nn_pred_runs, "NN: Runs", axes[1, 2])

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / 'model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison plot to {output_path / 'model_comparison.png'}")

    plt.close(fig)

    return k_results, runs_results


# ============================================================================
# RAW vs DEVIATION MODEL COMPARISON UTILITIES
# ============================================================================

def compare_raw_vs_deviation(
    raw_fitter,
    dev_fitter,
    df_test: pd.DataFrame,
    target_name: str = 'strikeouts'
) -> Dict:
    """
    Side-by-side comparison of raw and deviation models.

    Both models are evaluated on RAW scale for fair comparison.

    Parameters
    ----------
    raw_fitter : MixedEffectsModelFitter
        Model trained on raw targets (e.g., away_bat_k)
    dev_fitter : MixedEffectsModelFitter
        Model trained on deviation targets (e.g., deviation_away_k)
    df_test : pd.DataFrame
        Test data with both raw and deviation targets
    target_name : str
        Name for display purposes ('strikeouts' or 'runs')

    Returns
    -------
    Dict containing:
        - raw_model: metrics dict
        - deviation_model: metrics dict on raw scale
        - improvement: percentage improvement in RMSE
        - weather_coef_comparison: DataFrame comparing coefficients
    """
    results = {
        'target': target_name,
        'raw_model': {},
        'deviation_model': {}
    }

    # Filter test data to valid deviations
    df_test_valid = df_test.dropna(subset=[dev_fitter.expected_col]).copy()

    # Try different model complexities
    for model_name in ['park_intercept', 'park_slopes', 'fixed_weather']:
        try:
            # Raw model evaluation
            raw_metrics = raw_fitter.evaluate(df_test_valid, model_name)
            results['raw_model'][model_name] = raw_metrics
        except Exception as e:
            results['raw_model'][model_name] = {'error': str(e)}

        try:
            # Deviation model evaluation ON RAW SCALE
            dev_metrics = dev_fitter.evaluate_on_raw_scale(df_test_valid, model_name)
            results['deviation_model'][model_name] = dev_metrics
        except Exception as e:
            results['deviation_model'][model_name] = {'error': str(e)}

    # Compute improvement for best model
    best_model = 'park_intercept'  # Default
    for model_name in ['park_slopes', 'park_intercept']:
        if (model_name in results['raw_model'] and
            'error' not in results['raw_model'][model_name] and
            model_name in results['deviation_model'] and
            'error' not in results['deviation_model'][model_name]):
            best_model = model_name
            break

    if ('error' not in results['raw_model'].get(best_model, {'error': True}) and
        'error' not in results['deviation_model'].get(best_model, {'error': True})):

        raw_rmse = results['raw_model'][best_model]['RMSE']
        dev_rmse = results['deviation_model'][best_model]['RMSE']
        improvement = (raw_rmse - dev_rmse) / raw_rmse * 100
        results['improvement'] = {
            'model': best_model,
            'raw_rmse': raw_rmse,
            'dev_rmse': dev_rmse,
            'pct_improvement': improvement
        }
    else:
        results['improvement'] = None

    # Compare weather coefficients
    try:
        raw_fe = raw_fitter.get_fixed_effects(best_model)
        dev_fe = dev_fitter.get_fixed_effects(best_model)

        # Merge on parameter name
        raw_fe = raw_fe.rename(columns={
            'Coefficient': 'Raw_Coef',
            'Std_Error': 'Raw_SE',
            'p_value': 'Raw_p'
        })
        dev_fe = dev_fe.rename(columns={
            'Coefficient': 'Dev_Coef',
            'Std_Error': 'Dev_SE',
            'p_value': 'Dev_p'
        })

        coef_comparison = pd.merge(
            raw_fe[['Parameter', 'Raw_Coef', 'Raw_SE', 'Raw_p']],
            dev_fe[['Parameter', 'Dev_Coef', 'Dev_SE', 'Dev_p']],
            on='Parameter',
            how='outer'
        )
        results['weather_coef_comparison'] = coef_comparison
    except Exception as e:
        results['weather_coef_comparison'] = None
        results['coef_error'] = str(e)

    return results


def compare_variance_decomposition(
    raw_fitter,
    dev_fitter,
    model_name: str = 'park_intercept'
) -> pd.DataFrame:
    """
    Compare variance decomposition between raw and deviation models.

    Parameters
    ----------
    raw_fitter : MixedEffectsModelFitter
        Raw target model
    dev_fitter : MixedEffectsModelFitter
        Deviation target model
    model_name : str
        Which model to compare

    Returns
    -------
    pd.DataFrame
        Variance components from both models side-by-side
    """
    try:
        raw_vc = raw_fitter.get_variance_components(model_name)
        raw_vc = raw_vc.rename(columns={
            'Variance': 'Raw_Variance',
            'Pct_of_Total': 'Raw_Pct'
        })

        dev_vc = dev_fitter.get_variance_components(model_name)
        dev_vc = dev_vc.rename(columns={
            'Variance': 'Dev_Variance',
            'Pct_of_Total': 'Dev_Pct'
        })

        comparison = pd.merge(
            raw_vc[['Source', 'Type', 'Raw_Variance', 'Raw_Pct']],
            dev_vc[['Source', 'Type', 'Dev_Variance', 'Dev_Pct']],
            on=['Source', 'Type'],
            how='outer'
        )

        return comparison
    except Exception as e:
        print(f"Error comparing variance decomposition: {e}")
        return None


def bootstrap_model_comparison(
    raw_fitter,
    dev_fitter,
    df_test: pd.DataFrame,
    model_name: str = 'park_intercept',
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict:
    """
    Bootstrap comparison of RMSE between raw and deviation models.

    Parameters
    ----------
    raw_fitter : MixedEffectsModelFitter
        Raw target model
    dev_fitter : MixedEffectsModelFitter
        Deviation target model
    df_test : pd.DataFrame
        Test data
    model_name : str
        Which model to compare
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level

    Returns
    -------
    Dict with bootstrap confidence intervals for RMSE difference
    """
    # Get predictions
    df_valid = df_test.dropna(subset=[dev_fitter.expected_col]).copy()

    y_true = df_valid[raw_fitter.target].values
    y_pred_raw = raw_fitter.predict(df_valid, model_name)
    y_pred_dev = dev_fitter.predict_raw(df_valid, model_name)

    # Filter out NaN predictions
    valid_mask = ~(np.isnan(y_pred_raw) | np.isnan(y_pred_dev))
    y_true = y_true[valid_mask]
    y_pred_raw = y_pred_raw[valid_mask]
    y_pred_dev = y_pred_dev[valid_mask]

    n = len(y_true)
    rmse_diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)

        rmse_raw = np.sqrt(np.mean((y_true[indices] - y_pred_raw[indices]) ** 2))
        rmse_dev = np.sqrt(np.mean((y_true[indices] - y_pred_dev[indices]) ** 2))
        rmse_diffs.append(rmse_raw - rmse_dev)

    point_estimate = np.sqrt(np.mean((y_true - y_pred_raw) ** 2)) - \
                     np.sqrt(np.mean((y_true - y_pred_dev) ** 2))

    lower = np.percentile(rmse_diffs, (1 - confidence) / 2 * 100)
    upper = np.percentile(rmse_diffs, (1 + confidence) / 2 * 100)

    # Positive difference means deviation model is better
    return {
        'rmse_diff': point_estimate,
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence,
        'n_bootstrap': n_bootstrap,
        'n_samples': n,
        'significant': lower > 0 or upper < 0  # CI doesn't contain 0
    }


def plot_raw_vs_deviation_comparison(
    comparison_results: Dict,
    target_name: str = 'Strikeouts',
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot comparison of raw vs deviation model performance.

    Parameters
    ----------
    comparison_results : Dict
        Output from compare_raw_vs_deviation()
    target_name : str
        Name for display
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    raw_rmse = []
    dev_rmse = []

    for model_name in ['fixed_weather', 'park_intercept', 'park_slopes']:
        if (model_name in comparison_results['raw_model'] and
            model_name in comparison_results['deviation_model'] and
            'error' not in comparison_results['raw_model'][model_name] and
            'error' not in comparison_results['deviation_model'][model_name]):

            models.append(model_name)
            raw_rmse.append(comparison_results['raw_model'][model_name]['RMSE'])
            dev_rmse.append(comparison_results['deviation_model'][model_name]['RMSE'])

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, raw_rmse, width, label='Raw Model', color='steelblue')
    bars2 = ax.bar(x + width/2, dev_rmse, width, label='Deviation Model', color='darkorange')

    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('RMSE (on raw scale)')
    ax.set_title(f'{target_name}: Raw vs Deviation Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    return ax


if __name__ == '__main__':
    # Example usage
    from data_prep import load_all_team_data, prepare_features, train_test_split_by_season, prepare_nn_data
    from ridge_lasso_model import train_all_models
    from nn_embedding_model import train_nn_models

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\nPreparing features...")
    X, y_k, y_runs = prepare_features(df, include_interactions=True)
    splits = train_test_split_by_season(df, X, y_k, y_runs)

    print("\nPreparing NN data...")
    nn_data = prepare_nn_data(df)

    # Train all models
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)

    ridge_k, ridge_runs = train_all_models(splits, model_type='ridge')
    lasso_k, lasso_runs = train_all_models(splits, model_type='lasso')
    nn_k, nn_runs = train_nn_models(nn_data, embedding_dim=8, epochs=100)

    # Create comparison report
    k_results, runs_results = create_full_comparison_report(
        ridge_k, ridge_runs,
        lasso_k, lasso_runs,
        nn_k, nn_runs,
        splits, nn_data,
        output_dir=script_dir.parent / 'analysis' / 'model_outputs'
    )
