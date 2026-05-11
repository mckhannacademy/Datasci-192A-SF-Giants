"""
Mixed-effects regression models for park-specific weather effects.

Model hierarchy:
1. Null: y = μ + u_team + u_season + ε
2. Fixed Weather: y = μ + β·weather + u_team + u_season + ε
3. Park Intercept: y = μ + β·weather + u_park + u_team + u_season + ε
4. Park Slopes: weather effects vary by park

Random effects capture:
- u_park: Park-specific baseline (dimensions, altitude, etc.)
- u_team: Team quality (independent of where they play)
- u_season: Season-wide trends (ball changes, rule changes)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
import warnings

BASIC_WEATHER_FEATURES = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']
ENHANCED_WEATHER_FEATURES = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'heat_index']
BASIC_RANDOM_SLOPES = ['temp_f', 'wspd_mph']
ENHANCED_RANDOM_SLOPES = ['temp_f', 'air_density']


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        return pd.DataFrame({'feature': available_features, 'VIF': [1.0] * len(available_features)})

    X = df[available_features].dropna()
    if len(X) == 0:
        return pd.DataFrame({'feature': available_features, 'VIF': [np.nan] * len(available_features)})

    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    vif_data = []
    for i, feature in enumerate(available_features):
        try:
            vif = variance_inflation_factor(X_with_const, i + 1)
            vif_data.append({'feature': feature, 'VIF': vif})
        except Exception:
            vif_data.append({'feature': feature, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    vif_df['status'] = vif_df['VIF'].apply(lambda x: 'SEVERE' if x >= 10 else ('MODERATE' if x >= 5 else 'OK'))
    return vif_df


def check_multicollinearity(df: pd.DataFrame, features: List[str], threshold: float = 5.0,
                             verbose: bool = True) -> Tuple[bool, pd.DataFrame]:
    """Check for multicollinearity issues and warn if found."""
    vif_df = compute_vif(df, features)
    has_issues = (vif_df['VIF'] > threshold).any()

    if verbose:
        print(f"\nMulticollinearity Check (VIF > {threshold}):")
        print(vif_df.to_string(index=False))
        if has_issues:
            problematic = vif_df[vif_df['VIF'] > threshold]['feature'].tolist()
            print(f"WARNING: High VIF for: {problematic}")
    return has_issues, vif_df


class MixedEffectsModelFitter:
    """
    Fits a hierarchy of mixed-effects models for away team performance.

    Supports both raw targets (away_bat_k) and deviation targets (actual - expected).
    """

    def __init__(self, target: str, weather_features: List[str] = None,
                 target_type: str = 'raw', expected_col: str = None, raw_target: str = None):
        self.target = target
        self.target_type = target_type
        self.weather_features = weather_features or ['temp_f', 'rhum', 'wspd_mph', 'wind_cf']
        self.models = {}
        self.comparison_df = None
        self._df_train = None
        self.expected_col = expected_col
        self.raw_target = raw_target

        if target_type == 'deviation' and (expected_col is None or raw_target is None):
            raise ValueError("For deviation models, must provide 'expected_col' and 'raw_target'")

    def fit_model_hierarchy(self, df_train: pd.DataFrame, random_slope_features: List[str] = None,
                             method: str = 'lbfgs', maxiter: int = 200,
                             check_vif: bool = True, vif_threshold: float = 5.0) -> Dict[str, MixedLMResults]:
        """Fit a hierarchy of increasingly complex mixed-effects models."""
        self._df_train = df_train.copy()
        random_slope_features = random_slope_features or ['temp_f', 'wspd_mph']

        for col in ['home_team', 'away_team', 'season']:
            if col in self._df_train.columns:
                self._df_train[col] = pd.Series([str(x) for x in self._df_train[col].values],
                                                 dtype='object', index=self._df_train.index)

        weather_formula = ' + '.join(self.weather_features) + ' + is_night'
        print(f"\nFitting Mixed-Effects Models for {self.target}")

        if check_vif:
            has_vif_issues, self._vif_results = check_multicollinearity(
                self._df_train, self.weather_features + ['is_night'], vif_threshold, verbose=True)
            if has_vif_issues:
                warnings.warn("High multicollinearity detected. Model coefficients may be unstable.")

        model_configs = [
            ('null', lambda: self._fit_null_model(method, maxiter)),
            ('fixed_weather', lambda: self._fit_fixed_weather_model(weather_formula, method, maxiter)),
            ('park_intercept', lambda: self._fit_park_intercept_model(weather_formula, method, maxiter)),
            ('park_slopes', lambda: self._fit_park_slopes_model(weather_formula, random_slope_features, method, maxiter)),
        ]

        for i, (name, fit_fn) in enumerate(model_configs, 1):
            print(f"\n[{i}/4] Fitting {name} model...")
            try:
                self.models[name] = fit_fn()
                print(f"      Converged: {self.models[name].converged}")
            except Exception as e:
                print(f"      Failed: {e}")
                self.models[name] = None

        self._create_comparison_table()
        return self.models

    def _fit_null_model(self, method: str, maxiter: int) -> MixedLMResults:
        model = smf.mixedlm(f"{self.target} ~ 1", data=self._df_train,
                            groups="away_team", vc_formula={"season": "0 + C(season)"})
        return model.fit(method=method, maxiter=maxiter)

    def _fit_fixed_weather_model(self, weather_formula: str, method: str, maxiter: int) -> MixedLMResults:
        model = smf.mixedlm(f"{self.target} ~ {weather_formula}", data=self._df_train,
                            groups="away_team", vc_formula={"season": "0 + C(season)"})
        return model.fit(method=method, maxiter=maxiter)

    def _fit_park_intercept_model(self, weather_formula: str, method: str, maxiter: int) -> MixedLMResults:
        model = smf.mixedlm(f"{self.target} ~ {weather_formula}", data=self._df_train,
                            groups="home_team", re_formula="~1",
                            vc_formula={"away_team": "0 + C(away_team)", "season": "0 + C(season)"})
        return model.fit(method=method, maxiter=maxiter)

    def _fit_park_slopes_model(self, weather_formula: str, random_slope_features: List[str],
                                method: str, maxiter: int) -> MixedLMResults:
        re_formula = "~" + " + ".join(random_slope_features)
        model = smf.mixedlm(f"{self.target} ~ {weather_formula}", data=self._df_train,
                            groups="home_team", re_formula=re_formula,
                            vc_formula={"away_team": "0 + C(away_team)", "season": "0 + C(season)"})
        return model.fit(method=method, maxiter=maxiter)

    def _create_comparison_table(self):
        rows = [{'Model': name, 'LogLik': r.llf, 'AIC': r.aic, 'BIC': r.bic,
                 'n_params': r.df_modelwc, 'Converged': r.converged}
                for name, r in self.models.items() if r is not None]
        self.comparison_df = pd.DataFrame(rows)
        if len(self.comparison_df) > 0:
            self.comparison_df = self.comparison_df.sort_values('AIC')

    def get_comparison_table(self) -> pd.DataFrame:
        return self.comparison_df

    def get_fixed_effects(self, model_name: str = 'park_intercept', apply_bonferroni: bool = True) -> pd.DataFrame:
        """Extract fixed effects (average weather effects across all parks)."""
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        fe_df = pd.DataFrame({
            'Parameter': result.fe_params.index, 'Coefficient': result.fe_params.values,
            'Std_Error': result.bse_fe.values, 'z_value': result.tvalues.values[:len(result.fe_params)],
            'p_value': result.pvalues.values[:len(result.fe_params)]
        })

        if apply_bonferroni:
            n_tests = len(fe_df) - 1
            if n_tests > 1:
                alpha_corrected = 0.05 / n_tests
                fe_df['bonferroni_alpha'] = alpha_corrected
                fe_df['significant_corrected'] = fe_df['p_value'] < alpha_corrected
                fe_df.loc[fe_df['Parameter'] == 'Intercept', ['significant_corrected', 'bonferroni_alpha']] = np.nan
        return fe_df

    def get_variance_components(self, model_name: str = 'park_intercept') -> pd.DataFrame:
        """Extract variance components (how much variance from each source)."""
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        components = []
        group_label = 'Group (home_team)' if model_name in ['park_intercept', 'park_slopes'] else 'Group (away_team)'

        try:
            if hasattr(result.cov_re, 'iloc') and result.cov_re.shape[0] > 0:
                group_var = result.cov_re.iloc[0, 0]
            else:
                re = result.random_effects
                group_var = np.var(list(re.values())) if re else 0.0
            components.append({'Source': group_label, 'Variance': group_var, 'Type': 'Random Intercept'})
        except Exception:
            components.append({'Source': group_label, 'Variance': 0.0, 'Type': 'Random Intercept'})

        if hasattr(result, 'vcomp') and result.vcomp is not None:
            vc_names = list(result.vcomp.keys()) if isinstance(result.vcomp, dict) else []
            vc_values = list(result.vcomp.values()) if isinstance(result.vcomp, dict) else result.vcomp
            for i, val in enumerate(vc_values):
                name = vc_names[i] if i < len(vc_names) else f'VC_{i}'
                components.append({'Source': name, 'Variance': val, 'Type': 'Variance Component'})

        components.append({'Source': 'Residual', 'Variance': result.scale, 'Type': 'Residual'})
        vc_df = pd.DataFrame(components)
        total_var = vc_df['Variance'].sum()
        vc_df['Pct_of_Total'] = (vc_df['Variance'] / total_var * 100).round(2) if total_var > 0 else 0.0
        return vc_df

    def get_random_effects(self, model_name: str = 'park_intercept') -> pd.DataFrame:
        """Extract random effects (BLUPs) for each park."""
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        rows = []
        for group, effects in result.random_effects.items():
            row = {'Park': group}
            if isinstance(effects, pd.Series):
                for idx, val in effects.items():
                    idx_str = str(idx)
                    if 'away_team' in idx_str or 'season' in idx_str:
                        continue
                    if idx_str in ['Group', 'Intercept'] or 'home_team' in idx_str:
                        row['Intercept'] = val
                    else:
                        row[idx_str.replace('[', '_').replace(']', '_')] = val
                if 'Intercept' not in row:
                    non_vc = [(i, v) for i, v in effects.items()
                              if 'away_team' not in str(i) and 'season' not in str(i)]
                    if non_vc:
                        row['Intercept'] = non_vc[0][1]
            else:
                row['Intercept'] = effects
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('Park').reset_index(drop=True)
        if 'Intercept' not in df.columns:
            df['Intercept'] = 0.0
        return df

    def compute_r_squared(self, model_name: str = 'park_intercept') -> Dict[str, float]:
        """Compute marginal and conditional R² (Nakagawa & Schielzeth 2013)."""
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        var_fixed = np.var(result.fittedvalues)
        var_residual = result.scale
        var_random = 0

        if hasattr(result, 'cov_re'):
            cov_re = result.cov_re
            var_random += np.trace(cov_re.values if hasattr(cov_re, 'values') else cov_re)
        if hasattr(result, 'vcomp') and result.vcomp is not None:
            var_random += sum(result.vcomp.values()) if isinstance(result.vcomp, dict) else np.sum(result.vcomp)

        total_var = var_fixed + var_random + var_residual
        return {
            'marginal_r2': var_fixed / total_var,
            'conditional_r2': (var_fixed + var_random) / total_var,
            'var_fixed': var_fixed, 'var_random': var_random, 'var_residual': var_residual
        }

    def predict(self, df: pd.DataFrame, model_name: str = 'park_intercept',
                include_random: bool = True) -> np.ndarray:
        """Generate predictions from a fitted model."""
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        df_pred = df.copy()
        for col in ['home_team', 'away_team', 'season']:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].astype(str)

        if include_random:
            return result.predict(df_pred)

        fe_params = result.fe_params
        y_pred = np.full(len(df_pred), fe_params.get('Intercept', 0.0))
        for feature in self.weather_features:
            if feature in df_pred.columns and feature in fe_params.index:
                y_pred += df_pred[feature].values * fe_params[feature]
        if 'is_night' in df_pred.columns and 'is_night' in fe_params.index:
            y_pred += df_pred['is_night'].values * fe_params['is_night']
        return y_pred

    def evaluate(self, df_test: pd.DataFrame, model_name: str = 'park_intercept') -> Dict[str, float]:
        """Evaluate model on test data."""
        y_true = df_test[self.target].values
        y_pred = self.predict(df_test, model_name)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return {
            'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'R2': 1 - ss_res / ss_tot,
            'n_samples': len(y_true)
        }

    def predict_raw(self, df: pd.DataFrame, model_name: str = 'park_intercept',
                    include_random: bool = True) -> np.ndarray:
        """Generate predictions on raw scale (adds back expected values for deviation models)."""
        deviation_pred = self.predict(df, model_name, include_random)
        if self.target_type == 'deviation':
            if self.expected_col not in df.columns:
                raise ValueError(f"Expected column '{self.expected_col}' not in data")
            return deviation_pred + df[self.expected_col].values
        return deviation_pred

    def evaluate_on_raw_scale(self, df_test: pd.DataFrame, model_name: str = 'park_intercept') -> Dict[str, float]:
        """Evaluate model on test data using raw scale."""
        if self.target_type == 'deviation':
            y_true = df_test[self.raw_target].values
            y_pred = self.predict_raw(df_test, model_name)
        else:
            y_true = df_test[self.target].values
            y_pred = self.predict(df_test, model_name)

        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return {
            'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'R2': 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            'n_samples': len(y_true), 'scale': 'raw'
        }

    def summary(self, model_name: str = 'park_intercept') -> str:
        if model_name not in self.models or self.models[model_name] is None:
            return f"Model '{model_name}' not fitted or failed"
        return self.models[model_name].summary().as_text()


def fit_mixed_effects_models(me_data: Dict[str, Any], random_slope_features: List[str] = None,
                              use_enhanced_slopes: bool = None) -> Tuple[MixedEffectsModelFitter, MixedEffectsModelFitter]:
    """Fit mixed-effects models for both strikeouts and runs."""
    is_enhanced = me_data.get('use_enhanced_features', False)
    if use_enhanced_slopes is None:
        use_enhanced_slopes = is_enhanced

    if random_slope_features is None:
        random_slope_features = ENHANCED_RANDOM_SLOPES if (use_enhanced_slopes and 'air_density' in me_data['weather_features']) else BASIC_RANDOM_SLOPES

    weather_features = me_data['weather_features']
    random_slope_features = [f for f in random_slope_features if f in weather_features]

    print(f"\nFitting models with: {weather_features}")
    print(f"Random slopes: {random_slope_features}")

    k_fitter = MixedEffectsModelFitter(target=me_data['target_strikeouts'], weather_features=weather_features)
    k_fitter.fit_model_hierarchy(me_data['df_train'], random_slope_features=random_slope_features)

    runs_fitter = MixedEffectsModelFitter(target=me_data['target_runs'], weather_features=weather_features)
    runs_fitter.fit_model_hierarchy(me_data['df_train'], random_slope_features=random_slope_features)
    return k_fitter, runs_fitter


def fit_deviation_models(me_data: Dict[str, Any], random_slope_features: List[str] = None,
                          use_enhanced_slopes: bool = None) -> Tuple[MixedEffectsModelFitter, MixedEffectsModelFitter]:
    """Fit mixed-effects models using deviation targets."""
    if 'deviation_columns' not in me_data:
        raise ValueError("me_data must come from prepare_deviation_data()")

    is_enhanced = me_data.get('use_enhanced_features', False)
    if use_enhanced_slopes is None:
        use_enhanced_slopes = is_enhanced

    if random_slope_features is None:
        random_slope_features = ENHANCED_RANDOM_SLOPES if (use_enhanced_slopes and 'air_density' in me_data['weather_features']) else BASIC_RANDOM_SLOPES

    weather_features = me_data['weather_features']
    random_slope_features = [f for f in random_slope_features if f in weather_features]
    df_train = me_data['df_train'].dropna(subset=['deviation_away_k', 'deviation_away_runs']).copy()

    print(f"\nFitting DEVIATION models: {len(df_train)} samples")

    k_fitter = MixedEffectsModelFitter(target='deviation_away_k', weather_features=weather_features,
                                        target_type='deviation', expected_col='expected_away_k', raw_target='away_bat_k')
    k_fitter.fit_model_hierarchy(df_train, random_slope_features=random_slope_features)

    runs_fitter = MixedEffectsModelFitter(target='deviation_away_runs', weather_features=weather_features,
                                           target_type='deviation', expected_col='expected_away_runs', raw_target='away_runs_scored')
    runs_fitter.fit_model_hierarchy(df_train, random_slope_features=random_slope_features)
    return k_fitter, runs_fitter


def time_series_cross_validate(df: pd.DataFrame, target: str, weather_features: List[str],
                                n_splits: int = 5, min_train_seasons: int = 2,
                                model_type: str = 'park_intercept', verbose: bool = True) -> Dict[str, Any]:
    """Perform time-series cross-validation using expanding window approach."""
    df = df.sort_values('game_date').copy()
    seasons = sorted(df['season'].astype(int).unique())

    if len(seasons) < min_train_seasons + 1:
        raise ValueError(f"Need at least {min_train_seasons + 1} seasons, got {len(seasons)}")

    n_available = len(seasons) - min_train_seasons
    actual_splits = min(n_splits, n_available)
    test_seasons = seasons[-actual_splits:]
    fold_metrics = []

    for i, test_season in enumerate(test_seasons):
        train_seasons = [s for s in seasons if s < test_season]
        if len(train_seasons) < min_train_seasons:
            continue

        if verbose:
            print(f"\nCV Fold {i+1}/{actual_splits}: Train {train_seasons}, Test {test_season}")

        df_train = df[df['season'].astype(int).isin(train_seasons)].copy()
        df_test = df[df['season'].astype(int) == test_season].copy()

        try:
            fitter = MixedEffectsModelFitter(target=target, weather_features=weather_features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitter._df_train = df_train
                fitter.models['park_intercept'] = fitter._fit_park_intercept_model(
                    ' + '.join(weather_features) + ' + is_night', 'lbfgs', 200)

            metrics = fitter.evaluate(df_test, 'park_intercept')
            fold_metrics.append({
                'fold': i + 1, 'train_seasons': train_seasons, 'test_season': test_season,
                'n_train': len(df_train), 'n_test': len(df_test),
                'RMSE': metrics['RMSE'], 'MAE': metrics['MAE'], 'R2': metrics['R2'],
                'converged': fitter.models['park_intercept'].converged
            })
            if verbose:
                print(f"  RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
            fold_metrics.append({'fold': i + 1, 'test_season': test_season, 'error': str(e)})

    successful = [f for f in fold_metrics if 'RMSE' in f]
    if not successful:
        return {'fold_metrics': fold_metrics, 'error': 'All folds failed'}

    rmse_values = [f['RMSE'] for f in successful]
    r2_values = [f['R2'] for f in successful]
    results = {
        'fold_metrics': fold_metrics,
        'mean_rmse': np.mean(rmse_values), 'std_rmse': np.std(rmse_values),
        'mean_r2': np.mean(r2_values), 'std_r2': np.std(r2_values),
        'n_successful_folds': len(successful), 'n_total_folds': len(fold_metrics)
    }

    if verbose:
        print(f"\nCV Summary: {results['n_successful_folds']}/{results['n_total_folds']} folds")
        print(f"Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        print(f"Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    return results


def compare_basic_vs_enhanced(df: pd.DataFrame, test_start_season: int = 2023) -> Dict[str, Any]:
    """Compare basic weather features vs enhanced physics-based features."""
    from .data_prep import prepare_mixed_effects_data

    print("Comparing Basic vs Enhanced Weather Features")
    me_data_basic = prepare_mixed_effects_data(df, use_enhanced_features=False, test_start_season=test_start_season)
    me_data_enhanced = prepare_mixed_effects_data(df, use_enhanced_features=True, include_interactions=True, test_start_season=test_start_season)

    k_basic, runs_basic = fit_mixed_effects_models(me_data_basic)
    k_enhanced, runs_enhanced = fit_mixed_effects_models(me_data_enhanced)

    results = {'basic': {'strikeouts': {}, 'runs': {}}, 'enhanced': {'strikeouts': {}, 'runs': {}}}

    for model_name in ['park_intercept', 'park_slopes']:
        for fitter, label, target in [(k_basic, 'basic', 'strikeouts'), (k_enhanced, 'enhanced', 'strikeouts'),
                                       (runs_basic, 'basic', 'runs'), (runs_enhanced, 'enhanced', 'runs')]:
            try:
                results[label][target][model_name] = fitter.compute_r_squared(model_name)
            except Exception:
                pass

    print("\nComparison Summary - Marginal R²:")
    for feature_set in ['basic', 'enhanced']:
        for model_name in ['park_intercept', 'park_slopes']:
            for target in ['strikeouts', 'runs']:
                if model_name in results[feature_set][target]:
                    r2 = results[feature_set][target][model_name].get('marginal_r2', 'N/A')
                    if isinstance(r2, float):
                        print(f"  {feature_set:10s} {target:12s} {model_name:15s}: {r2:.4f}")

    results['models'] = {'basic': {'strikeouts': k_basic, 'runs': runs_basic},
                         'enhanced': {'strikeouts': k_enhanced, 'runs': runs_enhanced}}
    results['data'] = {'basic': me_data_basic, 'enhanced': me_data_enhanced}
    return results


if __name__ == '__main__':
    from pathlib import Path
    from data_prep import load_all_team_data, prepare_mixed_effects_data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced features')
    parser.add_argument('--compare', action='store_true', help='Compare basic vs enhanced')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    if args.compare:
        results = compare_basic_vs_enhanced(df)
    else:
        me_data = prepare_mixed_effects_data(df, use_enhanced_features=args.enhanced, include_interactions=args.enhanced)
        k_fitter, runs_fitter = fit_mixed_effects_models(me_data)

        print("\nStrikeouts Model Comparison:")
        print(k_fitter.get_comparison_table())

        print("\nRuns Model Comparison:")
        print(runs_fitter.get_comparison_table())

        print("\nStrikeouts Fixed Effects (park_intercept):")
        print(k_fitter.get_fixed_effects('park_intercept'))

        print("\nTest Set Evaluation:")
        for fitter, name in [(k_fitter, 'Strikeouts'), (runs_fitter, 'Runs')]:
            for model_name in ['fixed_weather', 'park_intercept', 'park_slopes']:
                try:
                    metrics = fitter.evaluate(me_data['df_test'], model_name)
                    print(f"  {name} {model_name}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.4f}")
                except Exception as e:
                    print(f"  {name} {model_name}: Failed - {e}")
