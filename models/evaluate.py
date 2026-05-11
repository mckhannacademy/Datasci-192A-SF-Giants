"""
Evaluation and comparison utilities for regression models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R²."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def compare_models(results: Dict[str, Dict[str, float]], title: str = "Model Comparison") -> pd.DataFrame:
    """Create and print a comparison table of model results."""
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    print(f"\n{title}")
    print("=" * 60)
    print(df.to_string())
    return df


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, metric_fn: callable,
                                    n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric."""
    n = len(y_true)
    bootstrapped = [metric_fn(y_true[idx := np.random.choice(n, n, replace=True)], y_pred[idx])
                    for _ in range(n_bootstrap)]
    point_estimate = metric_fn(y_true, y_pred)
    return (point_estimate,
            np.percentile(bootstrapped, (1 - confidence) / 2 * 100),
            np.percentile(bootstrapped, (1 + confidence) / 2 * 100))


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                                title: str = "Predictions vs Actual", ax: plt.Axes = None) -> plt.Axes:
    """Create a scatter plot of predictions vs actual values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()

    metrics = compute_metrics(y_true, y_pred)
    ax.annotate(f"RMSE: {metrics['rmse']:.3f}\nMAE: {metrics['mae']:.3f}\nR²: {metrics['r2']:.3f}",
                xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "Residuals", ax: plt.Axes = None) -> plt.Axes:
    """Create a residual plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_pred, y_true - y_pred, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(title)
    return ax


def _add_lowess(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = 'g', frac: float = 0.3):
    """Add LOWESS smoothing line to an axes."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(x)
        smoothed = lowess(y[sorted_idx], x[sorted_idx], frac=frac)
        ax.plot(smoothed[:, 0], smoothed[:, 1], f'{color}-', linewidth=2)
    except ImportError:
        pass


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                               title_prefix: str = "Model", figsize: Tuple[int, int] = (14, 10)) -> Tuple[plt.Figure, Dict]:
    """Create comprehensive residual diagnostic plots (2x2 grid)."""
    from scipy import stats

    residuals = y_true - y_pred
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{title_prefix} - Residual Diagnostics', fontsize=14, fontweight='bold')

    # Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    _add_lowess(ax1, y_pred, residuals, 'g')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')

    # Q-Q Plot
    ax2 = axes[0, 1]
    stats.probplot(std_residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    ax2.get_lines()[0].set_markersize(3)
    ax2.get_lines()[0].set_alpha(0.5)

    # Scale-Location Plot
    ax3 = axes[1, 0]
    sqrt_abs_residuals = np.sqrt(np.abs(std_residuals))
    ax3.scatter(y_pred, sqrt_abs_residuals, alpha=0.5, s=10)
    _add_lowess(ax3, y_pred, sqrt_abs_residuals, 'r')
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location')

    # Histogram
    ax4 = axes[1, 1]
    ax4.hist(std_residuals, bins=30, density=True, alpha=0.7, color='steelblue')
    x = np.linspace(-4, 4, 100)
    ax4.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='Normal')
    ax4.set_xlabel('Standardized Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Residuals')
    ax4.legend()

    plt.tight_layout()

    # Compute diagnostics
    n = len(residuals)
    sample = residuals[np.random.choice(n, min(n, 5000), replace=False)] if n > 5000 else residuals
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    diagnostics = {
        'shapiro_statistic': shapiro_stat, 'shapiro_p_value': shapiro_p,
        'normality_assumption': 'OK' if shapiro_p > 0.05 else 'VIOLATED',
        'residual_mean': np.mean(residuals), 'residual_std': np.std(residuals),
        'residual_skewness': stats.skew(residuals), 'residual_kurtosis': stats.kurtosis(residuals)
    }

    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(residuals)
        diagnostics['durbin_watson'] = dw
        diagnostics['autocorrelation'] = 'OK' if 1.5 < dw < 2.5 else 'POSSIBLE'
    except ImportError:
        pass

    try:
        from scipy.stats import spearmanr
        _, bp_p = spearmanr(y_pred, np.abs(residuals))
        diagnostics['heteroscedasticity_p'] = bp_p
        diagnostics['homoscedasticity'] = 'OK' if bp_p > 0.05 else 'VIOLATED'
    except Exception:
        pass

    print(f"\nResidual Diagnostics: n={n}, Shapiro p={shapiro_p:.4f} ({diagnostics['normality_assumption']})")
    return fig, diagnostics


def run_model_diagnostics(fitter, df_test: pd.DataFrame, model_name: str = 'park_intercept',
                           target_name: str = 'strikeouts', save_path: str = None) -> Dict:
    """Run full diagnostic suite for a mixed-effects model."""
    y_true, y_pred = df_test[fitter.target].values, fitter.predict(df_test, model_name)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]

    fig, diagnostics = plot_residual_diagnostics(y_true, y_pred, f'{target_name.title()} - {model_name}')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {save_path}")

    diagnostics.update({'model_name': model_name, 'target': target_name, 'n_samples': len(y_true)})
    try:
        r2 = fitter.compute_r_squared(model_name)
        diagnostics.update({'marginal_r2': r2['marginal_r2'], 'conditional_r2': r2['conditional_r2']})
    except Exception:
        pass
    return {'figure': fig, 'diagnostics': diagnostics}


def plot_coefficient_importance(coefs_df: pd.DataFrame, top_n: int = 20,
                                 title: str = "Top Feature Coefficients", ax: plt.Axes = None) -> plt.Axes:
    """Plot top coefficient magnitudes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    top = coefs_df.head(top_n)
    colors = ['green' if c > 0 else 'red' for c in top['coefficient']]
    ax.barh(top['feature'], top['coefficient'], color=colors)
    ax.set_xlabel('Coefficient')
    ax.set_title(title)
    ax.invert_yaxis()
    return ax


def plot_park_embeddings_2d(embeddings_df: pd.DataFrame, title: str = "Park Embeddings (PCA)",
                             ax: plt.Axes = None) -> plt.Axes:
    """Plot 2D PCA projection of park embeddings."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings_df.values)
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=100, alpha=0.7)

    for i, park in enumerate(embeddings_df.index):
        ax.annotate(park, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=9, ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(title)
    return ax


def compute_park_similarity(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between park embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings_df.values)
    return pd.DataFrame(sim, index=embeddings_df.index, columns=embeddings_df.index)


def find_similar_parks(embeddings_df: pd.DataFrame, park: str, top_n: int = 5) -> pd.DataFrame:
    """Find most similar parks based on embeddings."""
    sim = compute_park_similarity(embeddings_df)[park].sort_values(ascending=False)
    return sim[sim.index != park].head(top_n).to_frame(name='similarity')


def create_full_comparison_report(ridge_k, ridge_runs, lasso_k, lasso_runs,
                                   nn_k, nn_runs, splits: Dict, nn_data: Dict,
                                   output_dir: str = None):
    """Create a comprehensive comparison report of all models."""
    k_results = {
        'Ridge': ridge_k.evaluate(splits['X_test'], splits['y_test_strikeouts']),
        'Lasso': lasso_k.evaluate(splits['X_test'], splits['y_test_strikeouts']),
        'Neural Net': nn_k.evaluate(nn_data['X_weather_test'], nn_data['park_ids_test'], nn_data['y_test_strikeouts'])
    }
    runs_results = {
        'Ridge': ridge_runs.evaluate(splits['X_test'], splits['y_test_runs']),
        'Lasso': lasso_runs.evaluate(splits['X_test'], splits['y_test_runs']),
        'Neural Net': nn_runs.evaluate(nn_data['X_weather_test'], nn_data['park_ids_test'], nn_data['y_test_runs'])
    }

    compare_models(k_results, "STRIKEOUTS Model Comparison")
    compare_models(runs_results, "RUNS Model Comparison")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

    for row, (y_true, models, target) in enumerate([
        (splits['y_test_strikeouts'], [(ridge_k, 'Ridge'), (lasso_k, 'Lasso'), (nn_k, 'NN')], 'Strikeouts'),
        (splits['y_test_runs'], [(ridge_runs, 'Ridge'), (lasso_runs, 'Lasso'), (nn_runs, 'NN')], 'Runs')
    ]):
        for col, (model, name) in enumerate(models):
            if name == 'NN':
                y_pred = model.predict(nn_data['X_weather_test'], nn_data['park_ids_test'])
            else:
                y_pred = model.predict(splits['X_test'])
            plot_predictions_vs_actual(y_true, y_pred, f"{name}: {target}", axes[row, col])

    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return k_results, runs_results


def compare_raw_vs_deviation(raw_fitter, dev_fitter, df_test: pd.DataFrame,
                              target_name: str = 'strikeouts') -> Dict:
    """Compare raw and deviation models on raw scale."""
    results = {'target': target_name, 'raw_model': {}, 'deviation_model': {}}
    df_valid = df_test.dropna(subset=[dev_fitter.expected_col]).copy()

    for model_name in ['park_intercept', 'park_slopes', 'fixed_weather']:
        try:
            results['raw_model'][model_name] = raw_fitter.evaluate(df_valid, model_name)
        except Exception as e:
            results['raw_model'][model_name] = {'error': str(e)}
        try:
            results['deviation_model'][model_name] = dev_fitter.evaluate_on_raw_scale(df_valid, model_name)
        except Exception as e:
            results['deviation_model'][model_name] = {'error': str(e)}

    best = 'park_intercept'
    for m in ['park_slopes', 'park_intercept']:
        if 'error' not in results['raw_model'].get(m, {}) and 'error' not in results['deviation_model'].get(m, {}):
            best = m
            break

    if 'error' not in results['raw_model'].get(best, {'error': True}) and \
       'error' not in results['deviation_model'].get(best, {'error': True}):
        raw_rmse = results['raw_model'][best]['RMSE']
        dev_rmse = results['deviation_model'][best]['RMSE']
        results['improvement'] = {'model': best, 'raw_rmse': raw_rmse, 'dev_rmse': dev_rmse,
                                  'pct_improvement': (raw_rmse - dev_rmse) / raw_rmse * 100}

    try:
        raw_fe = raw_fitter.get_fixed_effects(best).rename(columns={'Coefficient': 'Raw_Coef', 'p_value': 'Raw_p'})
        dev_fe = dev_fitter.get_fixed_effects(best).rename(columns={'Coefficient': 'Dev_Coef', 'p_value': 'Dev_p'})
        results['weather_coef_comparison'] = pd.merge(
            raw_fe[['Parameter', 'Raw_Coef', 'Raw_p']], dev_fe[['Parameter', 'Dev_Coef', 'Dev_p']],
            on='Parameter', how='outer')
    except Exception:
        results['weather_coef_comparison'] = None
    return results


def compare_variance_decomposition(raw_fitter, dev_fitter, model_name: str = 'park_intercept') -> Optional[pd.DataFrame]:
    """Compare variance decomposition between raw and deviation models."""
    try:
        raw_vc = raw_fitter.get_variance_components(model_name).rename(
            columns={'Variance': 'Raw_Variance', 'Pct_of_Total': 'Raw_Pct'})
        dev_vc = dev_fitter.get_variance_components(model_name).rename(
            columns={'Variance': 'Dev_Variance', 'Pct_of_Total': 'Dev_Pct'})
        return pd.merge(raw_vc[['Source', 'Type', 'Raw_Variance', 'Raw_Pct']],
                        dev_vc[['Source', 'Type', 'Dev_Variance', 'Dev_Pct']],
                        on=['Source', 'Type'], how='outer')
    except Exception as e:
        print(f"Error comparing variance: {e}")
        return None


def bootstrap_model_comparison(raw_fitter, dev_fitter, df_test: pd.DataFrame,
                                model_name: str = 'park_intercept', n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> Dict:
    """Bootstrap comparison of RMSE between raw and deviation models."""
    df_valid = df_test.dropna(subset=[dev_fitter.expected_col]).copy()
    y_true = df_valid[raw_fitter.target].values
    y_pred_raw = raw_fitter.predict(df_valid, model_name)
    y_pred_dev = dev_fitter.predict_raw(df_valid, model_name)

    valid = ~(np.isnan(y_pred_raw) | np.isnan(y_pred_dev))
    y_true, y_pred_raw, y_pred_dev = y_true[valid], y_pred_raw[valid], y_pred_dev[valid]
    n = len(y_true)

    rmse_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        rmse_diffs.append(np.sqrt(np.mean((y_true[idx] - y_pred_raw[idx])**2)) -
                          np.sqrt(np.mean((y_true[idx] - y_pred_dev[idx])**2)))

    point = np.sqrt(np.mean((y_true - y_pred_raw)**2)) - np.sqrt(np.mean((y_true - y_pred_dev)**2))
    lower = np.percentile(rmse_diffs, (1 - confidence) / 2 * 100)
    upper = np.percentile(rmse_diffs, (1 + confidence) / 2 * 100)

    return {'rmse_diff': point, 'ci_lower': lower, 'ci_upper': upper, 'confidence': confidence,
            'n_bootstrap': n_bootstrap, 'n_samples': n, 'significant': lower > 0 or upper < 0}


def plot_raw_vs_deviation_comparison(comparison_results: Dict, target_name: str = 'Strikeouts',
                                      ax: plt.Axes = None) -> plt.Axes:
    """Plot comparison of raw vs deviation model performance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    models, raw_rmse, dev_rmse = [], [], []
    for m in ['fixed_weather', 'park_intercept', 'park_slopes']:
        if ('error' not in comparison_results['raw_model'].get(m, {}) and
            'error' not in comparison_results['deviation_model'].get(m, {})):
            models.append(m)
            raw_rmse.append(comparison_results['raw_model'][m]['RMSE'])
            dev_rmse.append(comparison_results['deviation_model'][m]['RMSE'])

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, raw_rmse, width, label='Raw Model', color='steelblue')
    bars2 = ax.bar(x + width/2, dev_rmse, width, label='Deviation Model', color='darkorange')

    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('RMSE (on raw scale)')
    ax.set_title(f'{target_name}: Raw vs Deviation Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    return ax


if __name__ == '__main__':
    from data_prep import load_all_team_data, prepare_features, train_test_split_by_season, prepare_nn_data
    from ridge_lasso_model import train_all_models
    from nn_embedding_model import train_nn_models

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)
    X, y_k, y_runs = prepare_features(df, include_interactions=True)
    splits = train_test_split_by_season(df, X, y_k, y_runs)
    nn_data = prepare_nn_data(df)

    print("\nTraining models...")
    ridge_k, ridge_runs = train_all_models(splits, model_type='ridge')
    lasso_k, lasso_runs = train_all_models(splits, model_type='lasso')
    nn_k, nn_runs = train_nn_models(nn_data, embedding_dim=8, epochs=100)

    create_full_comparison_report(ridge_k, ridge_runs, lasso_k, lasso_runs, nn_k, nn_runs,
                                   splits, nn_data, output_dir=script_dir.parent / 'analysis' / 'model_outputs')
