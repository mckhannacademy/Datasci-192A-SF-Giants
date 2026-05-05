"""
Park-Weather Interaction Model using Ridge/Lasso Regularization.

This module implements a regularized regression model that captures park-specific
weather sensitivities through explicit interaction terms. This approach provides:

1. **Interpretability**: Clear coefficients for each park's weather sensitivity
2. **Regularization**: Ridge/Lasso prevents overfitting with 90+ interaction terms
3. **Cross-validation**: Automatic alpha selection via RidgeCV/LassoCV

Model Structure:
    y = Intercept
      + (beta_temp + beta_temp_x_park) * temp_f
      + (beta_wspd + beta_wspd_x_park) * wspd_mph
      + (beta_wind + beta_wind_x_park) * wind_cf
      + beta_night * is_night
      + park_intercept

Example Usage:
    >>> from data_prep import load_all_team_data, prepare_park_weather_interactions
    >>> from park_weather_interaction_model import ParkWeatherInteractionModel
    >>>
    >>> df = load_all_team_data('data')
    >>> data = prepare_park_weather_interactions(df)
    >>>
    >>> model = ParkWeatherInteractionModel(target='strikeouts')
    >>> model.fit(data)
    >>> print(model.get_park_weather_coefficients())
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from pathlib import Path


class ParkWeatherInteractionModel:
    """
    Ridge/Lasso regression model with park × weather interaction terms.

    This model captures park-specific weather sensitivities by including:
    - Main weather effects (population average)
    - Park intercepts (baseline differences)
    - Park × weather interactions (park-specific deviations from average)

    Attributes
    ----------
    target : str
        Target variable: 'strikeouts' or 'runs'
    model_type : str
        'ridge' or 'lasso'
    alphas : np.ndarray
        Regularization strengths to try in cross-validation
    model : RidgeCV or LassoCV
        Fitted sklearn model
    feature_names : List[str]
        Names of all features in order
    weather_features : List[str]
        Weather feature names
    parks : List[str]
        Park codes
    coefficients_ : Dict[str, float]
        Named coefficients after fitting
    """

    def __init__(
        self,
        target: str = 'strikeouts',
        model_type: str = 'ridge',
        alphas: np.ndarray = None,
        cv: int = 5
    ):
        """
        Initialize the model.

        Parameters
        ----------
        target : str
            Target variable: 'strikeouts' or 'runs'
        model_type : str
            'ridge' (L2 regularization) or 'lasso' (L1 regularization)
            Ridge recommended for keeping all interaction terms;
            Lasso for automatic feature selection.
        alphas : np.ndarray, optional
            Regularization strengths to try. If None, uses sensible defaults.
        cv : int
            Number of cross-validation folds for alpha selection
        """
        self.target = target
        self.model_type = model_type.lower()
        self.cv = cv

        # Default alphas span a wide range
        if alphas is None:
            self.alphas = np.logspace(-2, 4, 50)  # 0.01 to 10000
        else:
            self.alphas = alphas

        self.model = None
        self.feature_names = None
        self.weather_features = None
        self.parks = None
        self.park_dummies = None
        self.interaction_features = None
        self.coefficients_ = None
        self.scaling_params = None
        self.metrics_ = None

    def fit(self, data: Dict[str, Any]) -> 'ParkWeatherInteractionModel':
        """
        Fit the model to prepared interaction data.

        Parameters
        ----------
        data : Dict
            Output from prepare_park_weather_interactions(), containing:
            - X_train, X_test: Feature matrices
            - y_train_strikeouts, y_train_runs: Target vectors
            - feature_names, weather_features, parks, etc.

        Returns
        -------
        self
            Fitted model
        """
        # Store metadata
        self.feature_names = data['feature_names']
        self.weather_features = data['weather_features']
        self.parks = data['parks']
        self.park_dummies = data['park_dummies']
        self.interaction_features = data['interaction_features']
        self.scaling_params = data.get('scaling_params', {})

        # Get training data
        X_train = data['X_train']
        if self.target == 'strikeouts':
            y_train = data['y_train_strikeouts']
            y_test = data['y_test_strikeouts']
        else:
            y_train = data['y_train_runs']
            y_test = data['y_test_runs']

        X_test = data['X_test']

        # Create and fit model
        if self.model_type == 'ridge':
            self.model = RidgeCV(alphas=self.alphas, cv=self.cv)
        elif self.model_type == 'lasso':
            self.model = LassoCV(alphas=self.alphas, cv=self.cv, max_iter=10000)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X_train, y_train)

        # Store coefficients as dict
        self.coefficients_ = {
            'Intercept': self.model.intercept_
        }
        for name, coef in zip(self.feature_names, self.model.coef_):
            self.coefficients_[name] = coef

        # Compute metrics
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        self.metrics_ = {
            'alpha': self.model.alpha_,
            'n_features': len(self.feature_names),
            'n_nonzero_coefs': np.sum(np.abs(self.model.coef_) > 1e-10),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
        }

        print(f"\n{self.model_type.upper()} Model Fit ({self.target}):")
        print(f"  Optimal alpha: {self.metrics_['alpha']:.4f}")
        print(f"  Non-zero coefficients: {self.metrics_['n_nonzero_coefs']} / {self.metrics_['n_features']}")
        print(f"  Train R²: {self.metrics_['train_r2']:.4f}")
        print(f"  Test R²:  {self.metrics_['test_r2']:.4f}")
        print(f"  Train RMSE: {self.metrics_['train_rmse']:.3f}")
        print(f"  Test RMSE:  {self.metrics_['test_rmse']:.3f}")

        return self

    def get_fixed_effects(self) -> pd.DataFrame:
        """
        Get main (fixed) effect coefficients.

        Returns
        -------
        pd.DataFrame
            Fixed effects with Intercept, weather features, and is_night
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        rows = []

        # Intercept
        rows.append({
            'Parameter': 'Intercept',
            'Coefficient': self.coefficients_['Intercept']
        })

        # Weather features
        for feat in self.weather_features:
            rows.append({
                'Parameter': feat,
                'Coefficient': self.coefficients_.get(feat, 0.0)
            })

        # is_night
        rows.append({
            'Parameter': 'is_night',
            'Coefficient': self.coefficients_.get('is_night', 0.0)
        })

        return pd.DataFrame(rows)

    def get_park_intercepts(self) -> pd.DataFrame:
        """
        Get park-specific intercept adjustments.

        Returns
        -------
        pd.DataFrame
            Park intercepts sorted by value
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        rows = []
        for park in self.parks:
            dummy_name = f'park_{park}'
            rows.append({
                'Park': park,
                'Intercept': self.coefficients_.get(dummy_name, 0.0)
            })

        return pd.DataFrame(rows).sort_values('Intercept', ascending=False)

    def get_park_weather_coefficients(self) -> pd.DataFrame:
        """
        Get park-specific weather coefficients (total effect).

        For each park and weather feature:
        total_effect = main_effect + interaction_effect

        Returns
        -------
        pd.DataFrame
            DataFrame with parks as rows, weather features as columns
            Values are total effects (main + interaction)
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Get main effects
        main_effects = {feat: self.coefficients_.get(feat, 0.0) for feat in self.weather_features}

        # Build park-specific totals
        rows = []
        for park in self.parks:
            row = {'Park': park}
            for feat in self.weather_features:
                interaction_name = f'{feat}_x_{park}'
                interaction_effect = self.coefficients_.get(interaction_name, 0.0)
                total_effect = main_effects[feat] + interaction_effect
                row[feat] = total_effect
                row[f'{feat}_main'] = main_effects[feat]
                row[f'{feat}_interaction'] = interaction_effect
            rows.append(row)

        return pd.DataFrame(rows).set_index('Park')

    def get_interaction_coefficients(self) -> pd.DataFrame:
        """
        Get only the interaction coefficients (deviations from average).

        Returns
        -------
        pd.DataFrame
            DataFrame with parks as rows, weather features as columns
            Values are interaction effects only (park deviation from population)
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        rows = []
        for park in self.parks:
            row = {'Park': park}
            for feat in self.weather_features:
                interaction_name = f'{feat}_x_{park}'
                row[feat] = self.coefficients_.get(interaction_name, 0.0)
            rows.append(row)

        return pd.DataFrame(rows).set_index('Park')

    def summarize_park_sensitivities(self) -> pd.DataFrame:
        """
        Rank parks by their weather sensitivity.

        Returns DataFrame with:
        - Total absolute weather sensitivity (sum of |coefficients|)
        - Individual feature sensitivities
        - Sensitivity rank

        Returns
        -------
        pd.DataFrame
            Parks ranked by weather sensitivity
        """
        coef_df = self.get_park_weather_coefficients()

        # Compute total absolute sensitivity
        sensitivity_cols = self.weather_features
        coef_df['total_sensitivity'] = coef_df[sensitivity_cols].abs().sum(axis=1)

        # Add rank
        coef_df = coef_df.sort_values('total_sensitivity', ascending=False)
        coef_df['rank'] = range(1, len(coef_df) + 1)

        return coef_df[['rank', 'total_sensitivity'] + sensitivity_cols]

    def export_to_dashboard_params(
        self,
        existing_params: Dict = None,
        output_path: str = None
    ) -> Dict:
        """
        Export model coefficients to dashboard_params.json format.

        Parameters
        ----------
        existing_params : Dict, optional
            Existing dashboard params to update. If None, creates new structure.
        output_path : str, optional
            Path to save updated params. If None, just returns dict.

        Returns
        -------
        Dict
            Updated dashboard params with park_weather_interactions
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Start with existing or create new
        if existing_params is None:
            params = {}
        else:
            params = existing_params.copy()

        # Ensure target section exists
        target_key = self.target
        if target_key not in params:
            params[target_key] = {}

        # Get fixed effects
        fe_df = self.get_fixed_effects()
        fixed_effects = {}
        for _, row in fe_df.iterrows():
            fixed_effects[row['Parameter']] = round(row['Coefficient'], 4)

        # Get park intercepts
        park_intercepts = {}
        for park in self.parks:
            dummy_name = f'park_{park}'
            park_intercepts[park] = round(self.coefficients_.get(dummy_name, 0.0), 4)

        # Get park-weather interactions
        park_weather_interactions = {}
        for park in self.parks:
            park_weather_interactions[park] = {}
            for feat in self.weather_features:
                interaction_name = f'{feat}_x_{park}'
                park_weather_interactions[park][feat] = round(
                    self.coefficients_.get(interaction_name, 0.0), 4
                )

        # Update params
        params[target_key]['fixed_effects'] = fixed_effects
        params[target_key]['park_effects'] = park_intercepts
        params[target_key]['park_weather_interactions'] = park_weather_interactions
        params[target_key]['model_r2'] = {
            'train': round(self.metrics_['train_r2'], 4),
            'test': round(self.metrics_['test_r2'], 4)
        }

        # Update metadata if present
        if '_metadata' in params:
            params['_metadata']['model_type'] = f'{self.model_type}_interaction'
            params['_metadata']['features_used'] = self.weather_features + ['is_night']
            params['_metadata']['n_interaction_terms'] = len(self.interaction_features)
            params['_metadata']['optimal_alpha'] = round(self.metrics_['alpha'], 4)

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"Dashboard params saved to {output_path}")

        return params

    def predict(
        self,
        park: str,
        temp_f: float,
        wspd_mph: float,
        wind_cf: float,
        is_night: bool = False,
        standardized: bool = False
    ) -> float:
        """
        Make a single prediction for a specific park and weather conditions.

        Parameters
        ----------
        park : str
            Park code (e.g., 'SF', 'COL')
        temp_f : float
            Temperature (raw or standardized based on `standardized` param)
        wspd_mph : float
            Wind speed (raw or standardized)
        wind_cf : float
            Wind component toward CF (raw or standardized)
        is_night : bool
            Night game indicator
        standardized : bool
            If True, assumes inputs are already standardized.
            If False, will standardize using stored scaling params.

        Returns
        -------
        float
            Predicted value
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if park not in self.parks:
            raise ValueError(f"Unknown park: {park}. Valid parks: {self.parks}")

        # Standardize if needed
        if not standardized and self.scaling_params:
            weather_values = {
                'temp_f': temp_f,
                'wspd_mph': wspd_mph,
                'wind_cf': wind_cf
            }
            for feat in self.weather_features:
                if feat in self.scaling_params:
                    center = self.scaling_params[feat]['center']
                    scale = self.scaling_params[feat]['scale']
                    weather_values[feat] = (weather_values[feat] - center) / scale
            temp_f = weather_values.get('temp_f', temp_f)
            wspd_mph = weather_values.get('wspd_mph', wspd_mph)
            wind_cf = weather_values.get('wind_cf', wind_cf)

        # Build prediction
        pred = self.coefficients_['Intercept']

        # Main weather effects
        weather_vals = {'temp_f': temp_f, 'wspd_mph': wspd_mph, 'wind_cf': wind_cf}
        for feat in self.weather_features:
            pred += self.coefficients_.get(feat, 0.0) * weather_vals[feat]

        # is_night
        pred += self.coefficients_.get('is_night', 0.0) * (1 if is_night else 0)

        # Park intercept
        pred += self.coefficients_.get(f'park_{park}', 0.0)

        # Park-weather interactions
        for feat in self.weather_features:
            interaction_name = f'{feat}_x_{park}'
            pred += self.coefficients_.get(interaction_name, 0.0) * weather_vals[feat]

        return pred


def fit_park_weather_interaction_models(
    data: Dict[str, Any],
    model_type: str = 'ridge'
) -> Tuple['ParkWeatherInteractionModel', 'ParkWeatherInteractionModel']:
    """
    Fit both strikeouts and runs models.

    Parameters
    ----------
    data : Dict
        Output from prepare_park_weather_interactions()
    model_type : str
        'ridge' or 'lasso'

    Returns
    -------
    Tuple[ParkWeatherInteractionModel, ParkWeatherInteractionModel]
        (strikeouts_model, runs_model)
    """
    print("=" * 60)
    print("FITTING PARK-WEATHER INTERACTION MODELS")
    print("=" * 60)

    k_model = ParkWeatherInteractionModel(target='strikeouts', model_type=model_type)
    k_model.fit(data)

    runs_model = ParkWeatherInteractionModel(target='runs', model_type=model_type)
    runs_model.fit(data)

    return k_model, runs_model


# ============================================================================
# MODEL AUDIT FUNCTIONS
# ============================================================================

# Domed stadiums - should have near-zero weather sensitivity
DOMED_STADIUMS = ['TB', 'MIA', 'HOU', 'ARI', 'TOR', 'MIL', 'SEA', 'TEX']


def compare_ridge_vs_lasso(
    data: Dict[str, Any],
    target: str = 'strikeouts'
) -> Dict[str, Any]:
    """
    Compare Ridge vs Lasso models to assess feature selection.

    Lasso (L1) zeros out irrelevant coefficients while Ridge (L2) shrinks
    but keeps all. Comparing them reveals which interactions are truly useful.

    Parameters
    ----------
    data : Dict
        Output from prepare_park_weather_interactions()
    target : str
        'strikeouts' or 'runs'

    Returns
    -------
    Dict with comparison metrics and coefficient analysis
    """
    print(f"\n{'='*60}")
    print(f"RIDGE vs LASSO COMPARISON ({target.upper()})")
    print(f"{'='*60}")

    # Fit Ridge model
    ridge_model = ParkWeatherInteractionModel(target=target, model_type='ridge')
    ridge_model.fit(data)

    # Fit Lasso model
    lasso_model = ParkWeatherInteractionModel(target=target, model_type='lasso')
    lasso_model.fit(data)

    # Count non-zero coefficients
    ridge_nonzero = ridge_model.metrics_['n_nonzero_coefs']
    lasso_nonzero = lasso_model.metrics_['n_nonzero_coefs']

    # Identify zeroed-out features by Lasso
    lasso_zero_features = []
    for name, coef in lasso_model.coefficients_.items():
        if name != 'Intercept' and abs(coef) < 1e-10:
            lasso_zero_features.append(name)

    # Categorize zeroed features
    zero_interactions = [f for f in lasso_zero_features if '_x_' in f]
    zero_parks = [f for f in lasso_zero_features if f.startswith('park_')]
    zero_weather = [f for f in lasso_zero_features if f in data['weather_features']]

    # Performance comparison
    results = {
        'target': target,
        'ridge': {
            'alpha': ridge_model.metrics_['alpha'],
            'n_nonzero': ridge_nonzero,
            'train_r2': ridge_model.metrics_['train_r2'],
            'test_r2': ridge_model.metrics_['test_r2'],
            'train_rmse': ridge_model.metrics_['train_rmse'],
            'test_rmse': ridge_model.metrics_['test_rmse'],
        },
        'lasso': {
            'alpha': lasso_model.metrics_['alpha'],
            'n_nonzero': lasso_nonzero,
            'train_r2': lasso_model.metrics_['train_r2'],
            'test_r2': lasso_model.metrics_['test_r2'],
            'train_rmse': lasso_model.metrics_['train_rmse'],
            'test_rmse': lasso_model.metrics_['test_rmse'],
        },
        'lasso_zeroed_features': lasso_zero_features,
        'lasso_zeroed_interactions': zero_interactions,
        'lasso_zeroed_parks': zero_parks,
        'lasso_zeroed_weather': zero_weather,
        'ridge_model': ridge_model,
        'lasso_model': lasso_model,
    }

    # Print comparison
    print(f"\nNon-zero coefficients:")
    print(f"  Ridge: {ridge_nonzero} / {len(ridge_model.feature_names)}")
    print(f"  Lasso: {lasso_nonzero} / {len(lasso_model.feature_names)}")
    print(f"  Lasso zeroed out: {len(lasso_zero_features)} features")
    print(f"    - Interactions: {len(zero_interactions)}")
    print(f"    - Park dummies: {len(zero_parks)}")
    print(f"    - Weather main: {len(zero_weather)}")

    print(f"\nPerformance:")
    print(f"  Ridge Test R²:  {ridge_model.metrics_['test_r2']:.4f}")
    print(f"  Lasso Test R²:  {lasso_model.metrics_['test_r2']:.4f}")
    print(f"  Ridge Test RMSE: {ridge_model.metrics_['test_rmse']:.3f}")
    print(f"  Lasso Test RMSE: {lasso_model.metrics_['test_rmse']:.3f}")

    return results


def bootstrap_confidence_intervals(
    data: Dict[str, Any],
    target: str = 'strikeouts',
    n_bootstrap: int = 100,
    ci_level: float = 0.95,
    focus_features: List[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for interaction coefficients.

    Parameters
    ----------
    data : Dict
        Output from prepare_park_weather_interactions()
    target : str
        'strikeouts' or 'runs'
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (default 0.95 for 95% CI)
    focus_features : List[str], optional
        List of feature names to focus on. If None, all interactions.
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: feature, coef_mean, coef_std, ci_lower, ci_upper, significant
    """
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({target.upper()})")
    print(f"{'='*60}")
    print(f"Running {n_bootstrap} bootstrap iterations...")

    np.random.seed(random_state)

    X_train = data['X_train']
    if target == 'strikeouts':
        y_train = data['y_train_strikeouts']
    else:
        y_train = data['y_train_runs']

    feature_names = data['feature_names']
    n_samples = len(X_train)

    # Determine which features to track
    if focus_features is None:
        # Focus on interaction features only (they're the concern)
        focus_features = data['interaction_features']

    # Storage for bootstrap coefficients
    coef_samples = {feat: [] for feat in focus_features}
    intercept_samples = []

    # Bootstrap loop
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Fit model (use Ridge with moderate alpha for stability)
        model = Ridge(alpha=100.0)
        model.fit(X_boot, y_boot)

        intercept_samples.append(model.intercept_)
        for j, feat in enumerate(feature_names):
            if feat in focus_features:
                coef_samples[feat].append(model.coef_[j])

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1} / {n_bootstrap} iterations")

    # Compute statistics
    alpha = 1 - ci_level
    rows = []
    for feat in focus_features:
        samples = np.array(coef_samples[feat])
        mean_coef = np.mean(samples)
        std_coef = np.std(samples)
        ci_lower = np.percentile(samples, 100 * alpha / 2)
        ci_upper = np.percentile(samples, 100 * (1 - alpha / 2))

        # Significant if CI doesn't include zero
        significant = (ci_lower > 0) or (ci_upper < 0)

        rows.append({
            'feature': feat,
            'coef_mean': mean_coef,
            'coef_std': std_coef,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant
        })

    result_df = pd.DataFrame(rows)

    # Summary
    n_significant = result_df['significant'].sum()
    print(f"\nResults:")
    print(f"  Total interactions: {len(focus_features)}")
    print(f"  Statistically significant (95% CI excludes 0): {n_significant}")
    print(f"  Not significant: {len(focus_features) - n_significant}")

    # Show significant features
    if n_significant > 0:
        print(f"\nSignificant interactions:")
        sig_df = result_df[result_df['significant']].sort_values('coef_mean', key=abs, ascending=False)
        print(sig_df[['feature', 'coef_mean', 'ci_lower', 'ci_upper']].head(15).to_string(index=False))

    return result_df


def compare_with_baseline_model(
    data: Dict[str, Any],
    target: str = 'strikeouts'
) -> Dict[str, Any]:
    """
    Compare the full interaction model with a simpler baseline (park intercepts only).

    The baseline model has:
    - Main weather effects (shared across all parks)
    - Park intercepts (baseline differences)
    - NO park × weather interactions

    If the baseline performs similarly, interactions are not adding value.

    Parameters
    ----------
    data : Dict
        Output from prepare_park_weather_interactions()
    target : str
        'strikeouts' or 'runs'

    Returns
    -------
    Dict with comparison metrics
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: INTERACTION vs BASELINE ({target.upper()})")
    print(f"{'='*60}")

    # Get data
    X_train = data['X_train']
    X_test = data['X_test']
    if target == 'strikeouts':
        y_train = data['y_train_strikeouts']
        y_test = data['y_test_strikeouts']
    else:
        y_train = data['y_train_runs']
        y_test = data['y_test_runs']

    feature_names = data['feature_names']
    weather_features = data['weather_features']
    park_dummies = data['park_dummies']
    interaction_features = data['interaction_features']

    # Build baseline feature matrix (no interactions)
    baseline_features = weather_features + ['is_night'] + park_dummies
    baseline_idx = [feature_names.index(f) for f in baseline_features]

    X_train_baseline = X_train[:, baseline_idx]
    X_test_baseline = X_test[:, baseline_idx]

    # Fit full interaction model
    full_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    full_model.fit(X_train, y_train)

    y_train_pred_full = full_model.predict(X_train)
    y_test_pred_full = full_model.predict(X_test)

    full_metrics = {
        'alpha': full_model.alpha_,
        'n_features': len(feature_names),
        'train_r2': r2_score(y_train, y_train_pred_full),
        'test_r2': r2_score(y_test, y_test_pred_full),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_full)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_full)),
    }

    # Fit baseline model
    baseline_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    baseline_model.fit(X_train_baseline, y_train)

    y_train_pred_base = baseline_model.predict(X_train_baseline)
    y_test_pred_base = baseline_model.predict(X_test_baseline)

    baseline_metrics = {
        'alpha': baseline_model.alpha_,
        'n_features': len(baseline_features),
        'train_r2': r2_score(y_train, y_train_pred_base),
        'test_r2': r2_score(y_test, y_test_pred_base),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_base)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_base)),
    }

    # Comparison
    results = {
        'target': target,
        'full_model': full_metrics,
        'baseline_model': baseline_metrics,
        'interaction_features_count': len(interaction_features),
        'test_r2_improvement': full_metrics['test_r2'] - baseline_metrics['test_r2'],
        'test_rmse_improvement': baseline_metrics['test_rmse'] - full_metrics['test_rmse'],
    }

    print(f"\nFull Model (with {len(interaction_features)} interactions):")
    print(f"  Features: {full_metrics['n_features']}")
    print(f"  Train R²: {full_metrics['train_r2']:.4f}")
    print(f"  Test R²:  {full_metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {full_metrics['test_rmse']:.3f}")

    print(f"\nBaseline Model (no interactions):")
    print(f"  Features: {baseline_metrics['n_features']}")
    print(f"  Train R²: {baseline_metrics['train_r2']:.4f}")
    print(f"  Test R²:  {baseline_metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {baseline_metrics['test_rmse']:.3f}")

    print(f"\nInteraction Value:")
    print(f"  Test R² improvement: {results['test_r2_improvement']:.4f}")
    print(f"  Test RMSE improvement: {results['test_rmse_improvement']:.3f}")

    if results['test_r2_improvement'] <= 0:
        print(f"\n  ⚠️  WARNING: Interactions HURT test performance!")
        print(f"      Simpler baseline model is preferred.")

    return results


def validate_domed_stadiums(
    model: 'ParkWeatherInteractionModel',
    threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Validate that domed stadiums show near-zero weather effects.

    Domed/retractable roof stadiums should be insulated from weather,
    so their weather interaction coefficients should be near zero.

    Parameters
    ----------
    model : ParkWeatherInteractionModel
        Fitted model
    threshold : float
        Maximum acceptable interaction coefficient magnitude for domed stadiums

    Returns
    -------
    Dict with validation results
    """
    print(f"\n{'='*60}")
    print(f"DOMED STADIUM VALIDATION ({model.target.upper()})")
    print(f"{'='*60}")

    interactions_df = model.get_interaction_coefficients()

    results = {
        'target': model.target,
        'threshold': threshold,
        'domed_stadiums': DOMED_STADIUMS,
        'violations': [],
        'domed_stats': {},
        'outdoor_stats': {},
    }

    # Analyze domed stadiums
    domed_coefs = []
    print(f"\nDomed Stadium Interactions (threshold = {threshold}):")
    print(f"{'Park':<6} {'temp_f':>10} {'wspd_mph':>10} {'wind_cf':>10} {'Status':>10}")
    print("-" * 50)

    for park in DOMED_STADIUMS:
        if park in interactions_df.index:
            row = interactions_df.loc[park]
            coefs = [row['temp_f'], row['wspd_mph'], row['wind_cf']]
            max_abs = max(abs(c) for c in coefs)
            status = "✓ OK" if max_abs < threshold else "✗ FAIL"

            print(f"{park:<6} {coefs[0]:>10.3f} {coefs[1]:>10.3f} {coefs[2]:>10.3f} {status:>10}")

            domed_coefs.extend(coefs)

            if max_abs >= threshold:
                results['violations'].append({
                    'park': park,
                    'temp_f': coefs[0],
                    'wspd_mph': coefs[1],
                    'wind_cf': coefs[2],
                    'max_abs': max_abs
                })

    # Stats for domed stadiums
    results['domed_stats'] = {
        'mean_abs': np.mean(np.abs(domed_coefs)),
        'max_abs': np.max(np.abs(domed_coefs)),
        'std': np.std(domed_coefs),
    }

    # Stats for outdoor stadiums (for comparison)
    outdoor_parks = [p for p in model.parks if p not in DOMED_STADIUMS]
    outdoor_coefs = []
    for park in outdoor_parks:
        if park in interactions_df.index:
            row = interactions_df.loc[park]
            outdoor_coefs.extend([row['temp_f'], row['wspd_mph'], row['wind_cf']])

    results['outdoor_stats'] = {
        'mean_abs': np.mean(np.abs(outdoor_coefs)),
        'max_abs': np.max(np.abs(outdoor_coefs)),
        'std': np.std(outdoor_coefs),
    }

    # Summary
    print(f"\nSummary:")
    print(f"  Domed stadiums checked: {len([p for p in DOMED_STADIUMS if p in interactions_df.index])}")
    print(f"  Violations: {len(results['violations'])}")
    print(f"  Domed mean |coef|: {results['domed_stats']['mean_abs']:.3f}")
    print(f"  Outdoor mean |coef|: {results['outdoor_stats']['mean_abs']:.3f}")

    if len(results['violations']) > 0:
        print(f"\n  ⚠️  WARNING: Domed stadiums have weather effects!")
        print(f"      This suggests the model is fitting noise.")

    return results


def fit_reduced_interaction_model(
    data: Dict[str, Any],
    target: str = 'strikeouts',
    key_parks: List[str] = None,
    exclude_domed: bool = True
) -> Dict[str, Any]:
    """
    Fit a reduced model with interactions only for key outdoor parks.

    This tests whether limiting interactions to physically-relevant parks
    improves out-of-sample performance.

    Parameters
    ----------
    data : Dict
        Output from prepare_park_weather_interactions()
    target : str
        'strikeouts' or 'runs'
    key_parks : List[str], optional
        Parks to include interactions for. If None, uses weather-sensitive parks.
    exclude_domed : bool
        Whether to exclude domed stadiums from interactions

    Returns
    -------
    Dict with reduced model results and comparison
    """
    print(f"\n{'='*60}")
    print(f"REDUCED INTERACTION MODEL ({target.upper()})")
    print(f"{'='*60}")

    if key_parks is None:
        # Default: outdoor parks known for extreme weather effects
        key_parks = ['COL', 'SF', 'CHC', 'BOS', 'MIN', 'CLE', 'NYY', 'KC', 'DET', 'CWS']

    if exclude_domed:
        key_parks = [p for p in key_parks if p not in DOMED_STADIUMS]

    print(f"Key parks for interactions: {key_parks}")

    # Get data
    X_train = data['X_train']
    X_test = data['X_test']
    if target == 'strikeouts':
        y_train = data['y_train_strikeouts']
        y_test = data['y_test_strikeouts']
    else:
        y_train = data['y_train_runs']
        y_test = data['y_test_runs']

    feature_names = data['feature_names']
    weather_features = data['weather_features']
    park_dummies = data['park_dummies']

    # Build reduced feature list
    # Keep: weather, is_night, all park dummies, interactions ONLY for key parks
    reduced_features = weather_features + ['is_night'] + park_dummies

    for feat in data['interaction_features']:
        # Check if interaction is for a key park
        park = feat.split('_x_')[-1]
        if park in key_parks:
            reduced_features.append(feat)

    reduced_idx = [feature_names.index(f) for f in reduced_features if f in feature_names]

    X_train_reduced = X_train[:, reduced_idx]
    X_test_reduced = X_test[:, reduced_idx]

    # Fit full model
    full_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    full_model.fit(X_train, y_train)

    y_test_pred_full = full_model.predict(X_test)
    full_test_r2 = r2_score(y_test, y_test_pred_full)
    full_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_full))

    # Fit reduced model
    reduced_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    reduced_model.fit(X_train_reduced, y_train)

    y_train_pred_red = reduced_model.predict(X_train_reduced)
    y_test_pred_red = reduced_model.predict(X_test_reduced)

    reduced_train_r2 = r2_score(y_train, y_train_pred_red)
    reduced_test_r2 = r2_score(y_test, y_test_pred_red)
    reduced_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_red))

    # Count interactions
    full_interactions = len(data['interaction_features'])
    reduced_interactions = len([f for f in reduced_features if '_x_' in f])

    results = {
        'target': target,
        'key_parks': key_parks,
        'full_model': {
            'n_features': len(feature_names),
            'n_interactions': full_interactions,
            'test_r2': full_test_r2,
            'test_rmse': full_test_rmse,
        },
        'reduced_model': {
            'n_features': len(reduced_features),
            'n_interactions': reduced_interactions,
            'train_r2': reduced_train_r2,
            'test_r2': reduced_test_r2,
            'test_rmse': reduced_test_rmse,
            'alpha': reduced_model.alpha_,
        },
        'improvement': {
            'test_r2': reduced_test_r2 - full_test_r2,
            'test_rmse': full_test_rmse - reduced_test_rmse,
        }
    }

    print(f"\nFull Model:")
    print(f"  Features: {results['full_model']['n_features']} ({full_interactions} interactions)")
    print(f"  Test R²:  {full_test_r2:.4f}")
    print(f"  Test RMSE: {full_test_rmse:.3f}")

    print(f"\nReduced Model:")
    print(f"  Features: {results['reduced_model']['n_features']} ({reduced_interactions} interactions)")
    print(f"  Train R²: {reduced_train_r2:.4f}")
    print(f"  Test R²:  {reduced_test_r2:.4f}")
    print(f"  Test RMSE: {reduced_test_rmse:.3f}")

    print(f"\nImprovement from reducing interactions:")
    print(f"  Test R² change: {results['improvement']['test_r2']:+.4f}")
    print(f"  Test RMSE change: {results['improvement']['test_rmse']:+.3f}")

    if results['improvement']['test_r2'] > 0:
        print(f"\n  ✓ Reduced model performs BETTER on test data!")
    else:
        print(f"\n  Reduced model performs similarly or worse.")

    return results


def run_full_model_audit(
    data: Dict[str, Any],
    n_bootstrap: int = 100
) -> Dict[str, Any]:
    """
    Run the complete model audit checklist.

    This function runs all verification checks:
    1. Ridge vs Lasso comparison
    2. Bootstrap confidence intervals
    3. Baseline model comparison
    4. Domed stadium validation
    5. Reduced interaction model test

    Parameters
    ----------
    data : Dict
        Output from prepare_park_weather_interactions()
    n_bootstrap : int
        Number of bootstrap iterations for CI calculation

    Returns
    -------
    Dict with all audit results
    """
    print("\n" + "=" * 70)
    print("PARK-WEATHER INTERACTION MODEL: FULL AUDIT")
    print("=" * 70)

    audit_results = {
        'strikeouts': {},
        'runs': {},
    }

    for target in ['strikeouts', 'runs']:
        print(f"\n\n{'#' * 70}")
        print(f"AUDITING {target.upper()} MODEL")
        print(f"{'#' * 70}")

        # 1. Ridge vs Lasso comparison
        ridge_lasso = compare_ridge_vs_lasso(data, target=target)
        audit_results[target]['ridge_vs_lasso'] = {
            'ridge_nonzero': ridge_lasso['ridge']['n_nonzero'],
            'lasso_nonzero': ridge_lasso['lasso']['n_nonzero'],
            'lasso_zeroed_interactions': len(ridge_lasso['lasso_zeroed_interactions']),
            'ridge_test_r2': ridge_lasso['ridge']['test_r2'],
            'lasso_test_r2': ridge_lasso['lasso']['test_r2'],
            'ridge_test_rmse': ridge_lasso['ridge']['test_rmse'],
            'lasso_test_rmse': ridge_lasso['lasso']['test_rmse'],
        }

        # 2. Bootstrap confidence intervals
        bootstrap_results = bootstrap_confidence_intervals(
            data, target=target, n_bootstrap=n_bootstrap
        )
        n_sig = bootstrap_results['significant'].sum()
        audit_results[target]['bootstrap_ci'] = {
            'total_interactions': len(bootstrap_results),
            'significant_at_95': n_sig,
            'pct_significant': n_sig / len(bootstrap_results) * 100,
        }

        # 3. Baseline model comparison
        baseline_compare = compare_with_baseline_model(data, target=target)
        audit_results[target]['baseline_comparison'] = {
            'full_test_r2': baseline_compare['full_model']['test_r2'],
            'baseline_test_r2': baseline_compare['baseline_model']['test_r2'],
            'test_r2_improvement': baseline_compare['test_r2_improvement'],
            'full_test_rmse': baseline_compare['full_model']['test_rmse'],
            'baseline_test_rmse': baseline_compare['baseline_model']['test_rmse'],
        }

        # 4. Domed stadium validation
        ridge_model = ridge_lasso['ridge_model']
        domed_validation = validate_domed_stadiums(ridge_model)
        audit_results[target]['domed_validation'] = {
            'n_violations': len(domed_validation['violations']),
            'domed_mean_abs_coef': domed_validation['domed_stats']['mean_abs'],
            'outdoor_mean_abs_coef': domed_validation['outdoor_stats']['mean_abs'],
            'violations': [v['park'] for v in domed_validation['violations']],
        }

        # 5. Reduced interaction model
        reduced_results = fit_reduced_interaction_model(data, target=target)
        audit_results[target]['reduced_model'] = {
            'full_interactions': reduced_results['full_model']['n_interactions'],
            'reduced_interactions': reduced_results['reduced_model']['n_interactions'],
            'full_test_r2': reduced_results['full_model']['test_r2'],
            'reduced_test_r2': reduced_results['reduced_model']['test_r2'],
            'test_r2_improvement': reduced_results['improvement']['test_r2'],
        }

    # Print summary
    print("\n\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    for target in ['strikeouts', 'runs']:
        print(f"\n{target.upper()}:")
        r = audit_results[target]

        print(f"  1. Ridge vs Lasso:")
        print(f"     Lasso zeroed {r['ridge_vs_lasso']['lasso_zeroed_interactions']} interactions")
        print(f"     Lasso test R²: {r['ridge_vs_lasso']['lasso_test_r2']:.4f}")

        print(f"  2. Bootstrap CI:")
        print(f"     Only {r['bootstrap_ci']['significant_at_95']}/{r['bootstrap_ci']['total_interactions']} interactions significant at 95%")

        print(f"  3. Baseline Comparison:")
        print(f"     Interactions add {r['baseline_comparison']['test_r2_improvement']:.4f} test R²")
        if r['baseline_comparison']['test_r2_improvement'] <= 0:
            print(f"     ⚠️  CONCLUSION: Interactions provide no value!")

        print(f"  4. Domed Stadiums:")
        print(f"     {r['domed_validation']['n_violations']} violations (should be 0)")
        if r['domed_validation']['n_violations'] > 0:
            print(f"     ⚠️  CONCLUSION: Model fitting noise in domed stadiums!")

        print(f"  5. Reduced Model:")
        print(f"     Removing {r['reduced_model']['full_interactions'] - r['reduced_model']['reduced_interactions']} interactions")
        print(f"     Test R² change: {r['reduced_model']['test_r2_improvement']:+.4f}")

    return audit_results


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from data_prep import load_all_team_data, prepare_park_weather_interactions

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Park-Weather Interaction Model')
    parser.add_argument('--audit', action='store_true', help='Run full model audit')
    parser.add_argument('--bootstrap', type=int, default=100, help='Bootstrap iterations for CI')
    args = parser.parse_args()

    # Load data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\nPreparing park-weather interaction data...")
    data = prepare_park_weather_interactions(
        df,
        weather_features=['temp_f', 'wspd_mph', 'wind_cf'],
        standardize=True
    )

    if args.audit:
        # Run full model audit
        audit_results = run_full_model_audit(data, n_bootstrap=args.bootstrap)

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        audit_results_json = convert_to_native(audit_results)

        # Save audit results
        audit_path = script_dir / 'model_audit_results.json'
        with open(audit_path, 'w') as f:
            json.dump(audit_results_json, f, indent=2)
        print(f"\nAudit results saved to {audit_path}")

    else:
        # Standard model fitting
        print("\n" + "=" * 60)
        print("FITTING MODELS")
        print("=" * 60)

        # Fit models
        k_model, runs_model = fit_park_weather_interaction_models(data, model_type='ridge')

        # Display results
        print("\n" + "=" * 60)
        print("STRIKEOUTS MODEL - FIXED EFFECTS")
        print("=" * 60)
        print(k_model.get_fixed_effects())

        print("\n" + "=" * 60)
        print("STRIKEOUTS MODEL - PARK WEATHER SENSITIVITIES (TOP 10)")
        print("=" * 60)
        sensitivity_df = k_model.summarize_park_sensitivities()
        print(sensitivity_df.head(10))

        print("\n" + "=" * 60)
        print("RUNS MODEL - FIXED EFFECTS")
        print("=" * 60)
        print(runs_model.get_fixed_effects())

        print("\n" + "=" * 60)
        print("RUNS MODEL - PARK WEATHER SENSITIVITIES (TOP 10)")
        print("=" * 60)
        sensitivity_df = runs_model.summarize_park_sensitivities()
        print(sensitivity_df.head(10))

        # Export to dashboard params
        print("\n" + "=" * 60)
        print("EXPORTING TO DASHBOARD PARAMS")
        print("=" * 60)

        # Load existing params
        params_path = script_dir / 'dashboard_params.json'
        if params_path.exists():
            with open(params_path, 'r') as f:
                existing_params = json.load(f)
        else:
            existing_params = {}

        # Update with both models
        params = k_model.export_to_dashboard_params(existing_params)
        params = runs_model.export_to_dashboard_params(params, str(params_path))

    print("\nDone!")
