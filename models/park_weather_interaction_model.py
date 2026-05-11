"""
Park-Weather Interaction Model using Ridge/Lasso Regularization.

Captures park-specific weather sensitivities through explicit interaction terms:
    y = Intercept + (beta_weather + beta_weather_x_park) * weather + park_intercept

Usage:
    model = ParkWeatherInteractionModel(target='strikeouts')
    model.fit(data)
    print(model.get_park_weather_coefficients())
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import RidgeCV, LassoCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from pathlib import Path

DOMED_STADIUMS = ['TB', 'MIA', 'HOU', 'ARI', 'TOR', 'MIL', 'SEA', 'TEX']


class ParkWeatherInteractionModel:
    """
    Ridge/Lasso regression with park × weather interaction terms.

    Captures: main weather effects, park intercepts, and park × weather interactions.
    """

    def __init__(self, target: str = 'strikeouts', model_type: str = 'ridge',
                 alphas: np.ndarray = None, cv: int = 5):
        self.target = target
        self.model_type = model_type.lower()
        self.cv = cv
        self.alphas = alphas if alphas is not None else np.logspace(-2, 4, 50)
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
        """Fit the model to prepared interaction data."""
        self.feature_names = data['feature_names']
        self.weather_features = data['weather_features']
        self.parks = data['parks']
        self.park_dummies = data['park_dummies']
        self.interaction_features = data['interaction_features']
        self.scaling_params = data.get('scaling_params', {})

        X_train, X_test = data['X_train'], data['X_test']
        y_train = data[f'y_train_{self.target}']
        y_test = data[f'y_test_{self.target}']

        if self.model_type == 'ridge':
            self.model = RidgeCV(alphas=self.alphas, cv=self.cv)
        elif self.model_type == 'lasso':
            self.model = LassoCV(alphas=self.alphas, cv=self.cv, max_iter=10000)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X_train, y_train)

        self.coefficients_ = {'Intercept': self.model.intercept_}
        for name, coef in zip(self.feature_names, self.model.coef_):
            self.coefficients_[name] = coef

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

        print(f"{self.model_type.upper()} ({self.target}): alpha={self.metrics_['alpha']:.4f}, "
              f"R²={self.metrics_['test_r2']:.4f}, RMSE={self.metrics_['test_rmse']:.3f}")
        return self

    def get_fixed_effects(self) -> pd.DataFrame:
        """Get main (fixed) effect coefficients."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")
        rows = [{'Parameter': 'Intercept', 'Coefficient': self.coefficients_['Intercept']}]
        for feat in self.weather_features:
            rows.append({'Parameter': feat, 'Coefficient': self.coefficients_.get(feat, 0.0)})
        rows.append({'Parameter': 'is_night', 'Coefficient': self.coefficients_.get('is_night', 0.0)})
        return pd.DataFrame(rows)

    def get_park_intercepts(self) -> pd.DataFrame:
        """Get park-specific intercept adjustments."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")
        rows = [{'Park': park, 'Intercept': self.coefficients_.get(f'park_{park}', 0.0)} for park in self.parks]
        return pd.DataFrame(rows).sort_values('Intercept', ascending=False)

    def get_park_weather_coefficients(self) -> pd.DataFrame:
        """Get park-specific weather coefficients (total = main + interaction)."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")

        main_effects = {feat: self.coefficients_.get(feat, 0.0) for feat in self.weather_features}
        rows = []
        for park in self.parks:
            row = {'Park': park}
            for feat in self.weather_features:
                interaction = self.coefficients_.get(f'{feat}_x_{park}', 0.0)
                row[feat] = main_effects[feat] + interaction
                row[f'{feat}_main'] = main_effects[feat]
                row[f'{feat}_interaction'] = interaction
            rows.append(row)
        return pd.DataFrame(rows).set_index('Park')

    def get_interaction_coefficients(self) -> pd.DataFrame:
        """Get only the interaction coefficients (park deviations from population)."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")
        rows = [{'Park': park, **{feat: self.coefficients_.get(f'{feat}_x_{park}', 0.0)
                                   for feat in self.weather_features}} for park in self.parks]
        return pd.DataFrame(rows).set_index('Park')

    def summarize_park_sensitivities(self) -> pd.DataFrame:
        """Rank parks by their weather sensitivity."""
        coef_df = self.get_park_weather_coefficients()
        coef_df['total_sensitivity'] = coef_df[self.weather_features].abs().sum(axis=1)
        coef_df = coef_df.sort_values('total_sensitivity', ascending=False)
        coef_df['rank'] = range(1, len(coef_df) + 1)
        return coef_df[['rank', 'total_sensitivity'] + self.weather_features]

    def export_to_dashboard_params(self, existing_params: Dict = None, output_path: str = None) -> Dict:
        """Export model coefficients to dashboard_params.json format."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")

        params = existing_params.copy() if existing_params else {}
        if self.target not in params:
            params[self.target] = {}

        fe_df = self.get_fixed_effects()
        params[self.target]['fixed_effects'] = {row['Parameter']: round(row['Coefficient'], 4) for _, row in fe_df.iterrows()}
        params[self.target]['park_effects'] = {park: round(self.coefficients_.get(f'park_{park}', 0.0), 4) for park in self.parks}
        params[self.target]['park_weather_interactions'] = {
            park: {feat: round(self.coefficients_.get(f'{feat}_x_{park}', 0.0), 4) for feat in self.weather_features}
            for park in self.parks
        }
        params[self.target]['model_r2'] = {'train': round(self.metrics_['train_r2'], 4), 'test': round(self.metrics_['test_r2'], 4)}

        if '_metadata' in params:
            params['_metadata'].update({
                'model_type': f'{self.model_type}_interaction',
                'features_used': self.weather_features + ['is_night'],
                'n_interaction_terms': len(self.interaction_features),
                'optimal_alpha': round(self.metrics_['alpha'], 4)
            })

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"Dashboard params saved to {output_path}")
        return params

    def predict(self, park: str, temp_f: float, wspd_mph: float, wind_cf: float,
                is_night: bool = False, standardized: bool = False) -> float:
        """Make a single prediction for specific park and weather conditions."""
        if self.coefficients_ is None:
            raise ValueError("Model not fitted yet.")
        if park not in self.parks:
            raise ValueError(f"Unknown park: {park}")

        weather_vals = {'temp_f': temp_f, 'wspd_mph': wspd_mph, 'wind_cf': wind_cf}
        if not standardized and self.scaling_params:
            for feat in self.weather_features:
                if feat in self.scaling_params:
                    weather_vals[feat] = (weather_vals[feat] - self.scaling_params[feat]['center']) / self.scaling_params[feat]['scale']

        pred = self.coefficients_['Intercept']
        for feat in self.weather_features:
            pred += self.coefficients_.get(feat, 0.0) * weather_vals[feat]
        pred += self.coefficients_.get('is_night', 0.0) * (1 if is_night else 0)
        pred += self.coefficients_.get(f'park_{park}', 0.0)
        for feat in self.weather_features:
            pred += self.coefficients_.get(f'{feat}_x_{park}', 0.0) * weather_vals[feat]
        return pred


def fit_park_weather_interaction_models(data: Dict[str, Any], model_type: str = 'ridge') -> Tuple:
    """Fit both strikeouts and runs models."""
    k_model = ParkWeatherInteractionModel(target='strikeouts', model_type=model_type)
    k_model.fit(data)
    runs_model = ParkWeatherInteractionModel(target='runs', model_type=model_type)
    runs_model.fit(data)
    return k_model, runs_model


def compare_ridge_vs_lasso(data: Dict[str, Any], target: str = 'strikeouts') -> Dict[str, Any]:
    """Compare Ridge vs Lasso models to assess feature selection."""
    ridge_model = ParkWeatherInteractionModel(target=target, model_type='ridge')
    ridge_model.fit(data)
    lasso_model = ParkWeatherInteractionModel(target=target, model_type='lasso')
    lasso_model.fit(data)

    lasso_zero = [name for name, coef in lasso_model.coefficients_.items()
                  if name != 'Intercept' and abs(coef) < 1e-10]

    print(f"\nRidge vs Lasso ({target}): Ridge {ridge_model.metrics_['n_nonzero_coefs']} nonzero, "
          f"Lasso {lasso_model.metrics_['n_nonzero_coefs']} nonzero, {len(lasso_zero)} zeroed")

    return {
        'target': target,
        'ridge': {'alpha': ridge_model.metrics_['alpha'], 'n_nonzero': ridge_model.metrics_['n_nonzero_coefs'],
                  'test_r2': ridge_model.metrics_['test_r2'], 'test_rmse': ridge_model.metrics_['test_rmse']},
        'lasso': {'alpha': lasso_model.metrics_['alpha'], 'n_nonzero': lasso_model.metrics_['n_nonzero_coefs'],
                  'test_r2': lasso_model.metrics_['test_r2'], 'test_rmse': lasso_model.metrics_['test_rmse']},
        'lasso_zeroed_features': lasso_zero,
        'lasso_zeroed_interactions': [f for f in lasso_zero if '_x_' in f],
        'ridge_model': ridge_model, 'lasso_model': lasso_model,
    }


def bootstrap_confidence_intervals(data: Dict[str, Any], target: str = 'strikeouts',
                                     n_bootstrap: int = 100, ci_level: float = 0.95,
                                     focus_features: List[str] = None, random_state: int = 42) -> pd.DataFrame:
    """Compute bootstrap confidence intervals for interaction coefficients."""
    print(f"Running {n_bootstrap} bootstrap iterations for {target}...")
    np.random.seed(random_state)

    X_train = data['X_train']
    y_train = data[f'y_train_{target}']
    feature_names = data['feature_names']
    n_samples = len(X_train)

    focus_features = focus_features or data['interaction_features']
    coef_samples = {feat: [] for feat in focus_features}

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        model = Ridge(alpha=100.0)
        model.fit(X_train[indices], y_train[indices])
        for j, feat in enumerate(feature_names):
            if feat in focus_features:
                coef_samples[feat].append(model.coef_[j])

    alpha = 1 - ci_level
    rows = []
    for feat in focus_features:
        samples = np.array(coef_samples[feat])
        ci_lower, ci_upper = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({
            'feature': feat, 'coef_mean': np.mean(samples), 'coef_std': np.std(samples),
            'ci_lower': ci_lower, 'ci_upper': ci_upper, 'significant': (ci_lower > 0) or (ci_upper < 0)
        })

    result = pd.DataFrame(rows)
    print(f"  Significant at {ci_level*100:.0f}%: {result['significant'].sum()}/{len(focus_features)}")
    return result


def compare_with_baseline_model(data: Dict[str, Any], target: str = 'strikeouts') -> Dict[str, Any]:
    """Compare full interaction model with baseline (park intercepts only)."""
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data[f'y_train_{target}'], data[f'y_test_{target}']
    feature_names = data['feature_names']

    baseline_features = data['weather_features'] + ['is_night'] + data['park_dummies']
    baseline_idx = [feature_names.index(f) for f in baseline_features]

    full_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    full_model.fit(X_train, y_train)
    full_r2 = r2_score(y_test, full_model.predict(X_test))

    baseline_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    baseline_model.fit(X_train[:, baseline_idx], y_train)
    baseline_r2 = r2_score(y_test, baseline_model.predict(X_test[:, baseline_idx]))

    improvement = full_r2 - baseline_r2
    print(f"Baseline vs Full ({target}): baseline R²={baseline_r2:.4f}, full R²={full_r2:.4f}, "
          f"improvement={improvement:+.4f}")

    return {'target': target, 'baseline_test_r2': baseline_r2, 'full_test_r2': full_r2, 'improvement': improvement}


def validate_domed_stadiums(model: ParkWeatherInteractionModel, threshold: float = 0.15) -> Dict[str, Any]:
    """Validate that domed stadiums show near-zero weather effects."""
    interactions_df = model.get_interaction_coefficients()
    violations = []

    for park in DOMED_STADIUMS:
        if park in interactions_df.index:
            row = interactions_df.loc[park]
            max_abs = max(abs(row[feat]) for feat in model.weather_features)
            if max_abs >= threshold:
                violations.append({'park': park, 'max_abs': max_abs})

    domed_coefs = [interactions_df.loc[p][f] for p in DOMED_STADIUMS if p in interactions_df.index
                   for f in model.weather_features]
    outdoor_coefs = [interactions_df.loc[p][f] for p in model.parks if p not in DOMED_STADIUMS
                     and p in interactions_df.index for f in model.weather_features]

    print(f"Domed validation ({model.target}): {len(violations)} violations, "
          f"domed mean|coef|={np.mean(np.abs(domed_coefs)):.3f}, outdoor={np.mean(np.abs(outdoor_coefs)):.3f}")

    return {
        'target': model.target, 'violations': violations,
        'domed_stats': {'mean_abs': np.mean(np.abs(domed_coefs))},
        'outdoor_stats': {'mean_abs': np.mean(np.abs(outdoor_coefs))}
    }


def fit_reduced_interaction_model(data: Dict[str, Any], target: str = 'strikeouts',
                                    key_parks: List[str] = None, exclude_domed: bool = True) -> Dict[str, Any]:
    """Fit reduced model with interactions only for key outdoor parks."""
    if key_parks is None:
        key_parks = ['COL', 'SF', 'CHC', 'BOS', 'MIN', 'CLE', 'NYY', 'KC', 'DET', 'CWS']
    if exclude_domed:
        key_parks = [p for p in key_parks if p not in DOMED_STADIUMS]

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data[f'y_train_{target}'], data[f'y_test_{target}']
    feature_names = data['feature_names']

    reduced_features = data['weather_features'] + ['is_night'] + data['park_dummies']
    reduced_features += [f for f in data['interaction_features'] if f.split('_x_')[-1] in key_parks]
    reduced_idx = [feature_names.index(f) for f in reduced_features if f in feature_names]

    full_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    full_model.fit(X_train, y_train)
    full_r2 = r2_score(y_test, full_model.predict(X_test))

    reduced_model = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
    reduced_model.fit(X_train[:, reduced_idx], y_train)
    reduced_r2 = r2_score(y_test, reduced_model.predict(X_test[:, reduced_idx]))

    print(f"Reduced model ({target}): {len(data['interaction_features'])} -> "
          f"{len([f for f in reduced_features if '_x_' in f])} interactions, R²={reduced_r2:.4f}")

    return {
        'target': target, 'key_parks': key_parks,
        'full_test_r2': full_r2, 'reduced_test_r2': reduced_r2,
        'improvement': reduced_r2 - full_r2
    }


def run_full_model_audit(data: Dict[str, Any], n_bootstrap: int = 100) -> Dict[str, Any]:
    """Run complete model audit: Ridge vs Lasso, bootstrap CI, baseline comparison, dome validation."""
    print("\nPARK-WEATHER MODEL AUDIT")
    audit_results = {}

    for target in ['strikeouts', 'runs']:
        audit_results[target] = {}
        ridge_lasso = compare_ridge_vs_lasso(data, target)
        audit_results[target]['ridge_vs_lasso'] = {
            'lasso_zeroed': len(ridge_lasso['lasso_zeroed_interactions']),
            'ridge_r2': ridge_lasso['ridge']['test_r2'], 'lasso_r2': ridge_lasso['lasso']['test_r2']
        }

        bootstrap = bootstrap_confidence_intervals(data, target, n_bootstrap)
        audit_results[target]['bootstrap'] = {'significant': bootstrap['significant'].sum()}

        baseline = compare_with_baseline_model(data, target)
        audit_results[target]['baseline'] = baseline

        domed = validate_domed_stadiums(ridge_lasso['ridge_model'])
        audit_results[target]['domed'] = {'violations': len(domed['violations'])}

        reduced = fit_reduced_interaction_model(data, target)
        audit_results[target]['reduced'] = reduced

    return audit_results


if __name__ == '__main__':
    import argparse
    from data_prep import load_all_team_data, prepare_park_weather_interactions

    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true', help='Run full model audit')
    parser.add_argument('--bootstrap', type=int, default=100)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)
    data = prepare_park_weather_interactions(df, weather_features=['temp_f', 'wspd_mph', 'wind_cf'], standardize=True)

    if args.audit:
        audit_results = run_full_model_audit(data, n_bootstrap=args.bootstrap)
        audit_path = script_dir / 'model_audit_results.json'
        with open(audit_path, 'w') as f:
            json.dump(audit_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nAudit saved to {audit_path}")
    else:
        k_model, runs_model = fit_park_weather_interaction_models(data)
        print("\nStrikeouts - Top 5 Weather Sensitive Parks:")
        print(k_model.summarize_park_sensitivities().head())
        print("\nRuns - Top 5 Weather Sensitive Parks:")
        print(runs_model.summarize_park_sensitivities().head())

    print("\nDone!")
