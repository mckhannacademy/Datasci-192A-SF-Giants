"""
Mixed-effects regression models for isolating park-specific weather effects.

This module implements mixed-effects models that control for team quality and
season trends while estimating how weather affects away team performance
differently across ballparks.

================================================================================
MODEL STRUCTURE IN PLAIN ENGLISH
================================================================================

The full model predicts away team strikeouts or runs as:

    y = μ + β₁·temp + β₂·humidity + β₃·wind_speed + β₄·wind_cf + β₅·is_night
        + β₆·air_density + β₇·wind_out_impact
        + u_park + u_team + u_season + ε

Where each component has a specific meaning:

FIXED EFFECTS (β coefficients) - Weather components:
------------------------------------------------------------------------
| Feature          | Description                  | Unit              |
|------------------|------------------------------|-------------------|
| temp_f           | Air temperature              | Fahrenheit        |
| rhum             | Relative humidity            | Percent (0-100)   |
| wspd_mph         | Wind speed magnitude         | Miles per hour    |
| wind_cf          | Wind toward center field     | Projected component |
| wind_lcf         | Wind toward left-center      | Projected component |
| wind_rcf         | Wind toward right-center     | Projected component |
| air_density      | Computed from temp/humid/pres| kg/m³             |
| wind_out_impact  | wspd_mph × wind_cf           | Interaction term  |
| is_night         | Night game indicator         | Binary (0/1)      |
------------------------------------------------------------------------

RANDOM EFFECTS - Plain English Explanations:
------------------------------------------------------------------------

u_park (Park Random Intercept):
    "Some parks just produce more strikeouts/runs than others, regardless of weather"

    Examples:
    - Coors Field (COL): u_park ≈ +1.5 runs → Games average ~1.5 more runs
    - Oracle Park (SF): u_park ≈ -0.6 runs → Games average ~0.6 fewer runs

    Why needed: Parks differ in dimensions, altitude, marine layer, wind patterns.
    The random intercept captures all park-specific effects not explained by weather.

u_team (Team Random Intercept):
    "Some teams strike out more/score more than others, regardless of where they play"

    Examples:
    - A team with strong contact hitters: u_team ≈ -1.2 K
    - A power-hitting team: u_team ≈ +0.8 runs

    Why needed: Controls for team quality so weather effects aren't confounded with
    "good teams just score more"

u_season (Season Random Intercept):
    "Some seasons have more scoring/strikeouts league-wide than others"

    Examples:
    - 2019 (juiced ball year): u_season ≈ +0.5 HR
    - 2014 (dead ball year): u_season ≈ -0.3 runs

    Why needed: Captures rule changes, ball construction changes, league-wide trends

========================================
MODEL HIERARCHY (4 Levels of Complexity)
========================================

1. Null Model:
   y = μ + u_team + u_season + ε
   Just accounts for team and season differences. No weather effects.

2. Fixed Weather:
   y = μ + β·weather + u_team + u_season + ε
   Adds weather as fixed effects (same effect everywhere).
   Assumes temperature affects all parks equally.

3. Park Random Intercept:
   y = μ + β·weather + u_park + u_team + u_season + ε
   Adds park-level baseline differences.
   Weather effects still assumed equal across parks.

4. Park Random Slopes (MOST COMPLEX):
   y = μ + β·weather + (β_park)·weather + u_park + u_team + u_season + ε
   Weather effects can VARY BY PARK.
   Example: Temperature might matter more at Coors (elevation) than at Tropicana (dome)

Enhanced version supports physics-based weather features:
- air_density: Affects ball flight distance (lower density = ball travels farther)
- heat_index: Affects player fatigue
- wind_out_impact: Interaction of wind speed × direction (captures non-linear effects)
- elevation_ft: Stadium elevation
- has_roof: Retractable roof indicator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults
import warnings


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature.

    VIF measures how much the variance of a regression coefficient is
    inflated due to multicollinearity with other features.

    VIF Guidelines:
    - VIF = 1: No correlation with other features
    - VIF < 5: Generally acceptable
    - VIF >= 5: Moderate multicollinearity (investigate)
    - VIF >= 10: Severe multicollinearity (remove or combine features)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the features
    features : List[str]
        List of feature column names to check

    Returns
    -------
    pd.DataFrame
        VIF values for each feature, sorted by VIF descending
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Filter to only features that exist in df
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        return pd.DataFrame({'feature': available_features, 'VIF': [1.0] * len(available_features)})

    # Extract feature matrix
    X = df[available_features].dropna()

    if len(X) == 0:
        warnings.warn("No valid rows for VIF computation")
        return pd.DataFrame({'feature': available_features, 'VIF': [np.nan] * len(available_features)})

    # Add constant for VIF calculation
    X_with_const = np.column_stack([np.ones(len(X)), X.values])

    vif_data = []
    for i, feature in enumerate(available_features):
        try:
            vif = variance_inflation_factor(X_with_const, i + 1)  # +1 because of constant
            vif_data.append({'feature': feature, 'VIF': vif})
        except Exception as e:
            warnings.warn(f"Could not compute VIF for {feature}: {e}")
            vif_data.append({'feature': feature, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    vif_df['status'] = vif_df['VIF'].apply(
        lambda x: 'SEVERE' if x >= 10 else ('MODERATE' if x >= 5 else 'OK')
    )

    return vif_df


def check_multicollinearity(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 5.0,
    verbose: bool = True
) -> Tuple[bool, pd.DataFrame]:
    """
    Check for multicollinearity issues and warn if found.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features
    features : List[str]
        Features to check
    threshold : float
        VIF threshold above which to warn (default: 5.0)
    verbose : bool
        Whether to print warnings

    Returns
    -------
    Tuple[bool, pd.DataFrame]
        (has_issues, vif_dataframe)
        has_issues is True if any VIF exceeds threshold
    """
    vif_df = compute_vif(df, features)

    has_issues = (vif_df['VIF'] > threshold).any()

    if verbose:
        print(f"\n{'='*60}")
        print("MULTICOLLINEARITY CHECK (VIF)")
        print(f"{'='*60}")
        print(f"Threshold: VIF > {threshold}")
        print(f"\n{vif_df.to_string(index=False)}")

        if has_issues:
            problematic = vif_df[vif_df['VIF'] > threshold]['feature'].tolist()
            print(f"\n⚠️  WARNING: High VIF detected for: {problematic}")
            print("Consider removing or combining these features.")
        else:
            print(f"\n✓ All features have VIF < {threshold}")

        print(f"{'='*60}\n")

    return has_issues, vif_df


# Default feature sets
# Simplified: removed wind_lcf/wind_rcf (r>0.94 with wind_cf) and wind_out_impact (p>0.65)
# Keep only features that are both statistically significant and not multicollinear
BASIC_WEATHER_FEATURES = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']
ENHANCED_WEATHER_FEATURES = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'heat_index']

# Default random slope features
# These are the weather features that may vary in effect by park
BASIC_RANDOM_SLOPES = ['temp_f', 'wspd_mph']
ENHANCED_RANDOM_SLOPES = ['temp_f', 'air_density']  # Air density is key for ball flight at altitude


class MixedEffectsModelFitter:
    """
    Fits a hierarchy of mixed-effects models for away team performance.

    This class implements a principled approach to understanding how weather
    affects baseball outcomes while controlling for confounding factors.

    ========================================================================
    WHY MIXED-EFFECTS MODELS?
    ========================================================================

    The challenge: We want to know "does hot weather lead to more home runs?"
    But teams that play in hot weather (Arizona, Texas) might just have different
    offense/pitching than teams in cool weather (San Francisco, Seattle).

    Mixed-effects models solve this by:
    1. Estimating AVERAGE weather effects (fixed effects) across all parks
    2. Allowing each park to have its own BASELINE (random intercepts)
    3. Optionally allowing weather effects to VARY BY PARK (random slopes)
    4. Controlling for team quality and season trends

    ========================================================================
    WEATHER FEATURES EXPLAINED
    ========================================================================

    Basic Features (always included):
    - temp_f: Air temperature in Fahrenheit
        Effect: Higher temps → lower air density → ball travels farther
    - rhum: Relative humidity (0-100%)
        Effect: Higher humidity → slightly lower air density → marginal effect
    - wspd_mph: Wind speed magnitude in mph
        Effect: Raw wind speed (direction matters too!)
    - wind_cf: Wind component toward center field (-1 to +1)
        Effect: Positive = blowing out (helps HRs), Negative = blowing in
    - air_density: Computed from temp/humidity/pressure (kg/m³)
        Effect: DIRECT physics driver - lower density = ball travels farther
        At Coors Field (~5280 ft), air is ~17% less dense than sea level
    - wind_out_impact: wspd_mph × wind_cf (interaction term)
        Effect: Captures that a 20 mph wind blowing out matters WAY more
        than a 5 mph wind blowing out (non-linear interaction)

    Enhanced Features (optional):
    - heat_index: Apparent temperature accounting for humidity
        Effect: Player fatigue, especially late in hot games
    - elevation_ft: Stadium elevation in feet
        Effect: Higher altitude = lower air density (Coors = 5280 ft)
    - has_roof: Whether stadium has retractable roof
        Effect: Nullifies weather when closed

    ========================================================================
    INTERPRETING MODEL OUTPUT
    ========================================================================

    Fixed Effects (β coefficients):
        These are the AVERAGE weather effects across all parks.
        Example: β_temp = +0.02 means each 1°F increase in temperature
        is associated with 0.02 more runs, averaged across all parks.

    Random Effects (u values):
        These are park/team/season-specific adjustments.
        Example: u_COL = +1.5 means Coors Field games have 1.5 more runs
        than predicted by weather alone (captures altitude, dimensions, etc.)

    Variance Components:
        How much of the total variance comes from each source.
        Example: If 40% is from parks, park selection matters a lot.

    ========================================================================
    MODEL HIERARCHY
    ========================================================================

    Models are fit in increasing complexity:
    1. null: Just team + season effects (baseline)
    2. fixed_weather: Add weather (same effect everywhere)
    3. park_intercept: Add park baselines
    4. park_slopes: Allow weather effects to vary by park

    Compare using AIC/BIC - lower is better. If park_slopes has much lower
    AIC than park_intercept, weather effects genuinely vary by park.

    Supports both raw targets (e.g., 'away_bat_k') and deviation targets
    (e.g., 'deviation_away_k' = actual - expected).

    Attributes
    ----------
    target : str
        Target variable name ('away_bat_k' or 'away_runs_scored')
    target_type : str
        Either 'raw' or 'deviation'
    weather_features : List[str]
        Weather feature column names
    models : Dict[str, MixedLMResults]
        Fitted model objects keyed by model name
    comparison_df : pd.DataFrame
        Model comparison statistics (AIC, BIC, log-likelihood)
    expected_col : str
        Column name for expected values (for deviation models)
    raw_target : str
        Column name for raw target (for deviation models evaluation)
    """

    def __init__(
        self,
        target: str,
        weather_features: List[str] = None,
        target_type: str = 'raw',
        expected_col: str = None,
        raw_target: str = None
    ):
        """
        Initialize the model fitter.

        Parameters
        ----------
        target : str
            Target variable ('away_bat_k', 'away_runs_scored', or deviation variants)
        weather_features : List[str], optional
            Weather features to include. Defaults to standard set.
        target_type : str
            'raw' for raw targets, 'deviation' for deviation targets
        expected_col : str, optional
            For deviation models: column with expected values (e.g., 'expected_away_k')
            Required if target_type='deviation'
        raw_target : str, optional
            For deviation models: column with raw actual values (e.g., 'away_bat_k')
            Required if target_type='deviation'
        """
        self.target = target
        self.target_type = target_type
        self.weather_features = weather_features or [
            'temp_f', 'rhum', 'wspd_mph', 'wind_cf'
        ]
        self.models = {}
        self.comparison_df = None
        self._df_train = None

        # Deviation-specific attributes
        self.expected_col = expected_col
        self.raw_target = raw_target

        if target_type == 'deviation':
            if expected_col is None or raw_target is None:
                raise ValueError(
                    "For deviation models, must provide 'expected_col' and 'raw_target'"
                )

    def fit_model_hierarchy(
        self,
        df_train: pd.DataFrame,
        random_slope_features: List[str] = None,
        method: str = 'lbfgs',
        maxiter: int = 200,
        check_vif: bool = True,
        vif_threshold: float = 5.0
    ) -> Dict[str, MixedLMResults]:
        """
        Fit a hierarchy of increasingly complex mixed-effects models.

        Models fitted:
        1. null: Intercept only with away_team and season variance components
        2. fixed_weather: Add weather as fixed effects
        3. park_intercept: Add random intercept for home_team (park)
        4. park_slopes: Add random slopes for weather by park

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with target, weather features, and grouping variables
        random_slope_features : List[str], optional
            Weather features to include as random slopes. Defaults to ['temp_f', 'wspd_mph']
        method : str
            Optimization method for fitting
        maxiter : int
            Maximum iterations for optimization
        check_vif : bool
            Whether to check for multicollinearity before fitting (default: True)
        vif_threshold : float
            VIF threshold for warnings (default: 5.0)

        Returns
        -------
        Dict[str, MixedLMResults]
            Fitted models keyed by name
        """
        self._df_train = df_train.copy()
        random_slope_features = random_slope_features or ['temp_f', 'wspd_mph']

        # Ensure regular Python string types (not pandas StringDtype)
        # statsmodels has issues with pandas StringDtype
        import pandas as pd
        for col in ['home_team', 'away_team', 'season']:
            if col in self._df_train.columns:
                self._df_train[col] = pd.Series(
                    [str(x) for x in self._df_train[col].values],
                    dtype='object',
                    index=self._df_train.index
                )

        weather_formula = ' + '.join(self.weather_features) + ' + is_night'

        print(f"\n{'='*60}")
        print(f"Fitting Mixed-Effects Models for {self.target}")
        print(f"{'='*60}")

        # Check for multicollinearity before fitting
        if check_vif:
            has_vif_issues, vif_df = check_multicollinearity(
                self._df_train,
                self.weather_features + ['is_night'],
                threshold=vif_threshold,
                verbose=True
            )
            self._vif_results = vif_df

            if has_vif_issues:
                warnings.warn(
                    "High multicollinearity detected. Consider removing correlated features. "
                    "Model coefficients may be unstable."
                )

        # Model 1: Null model with team and season variance components
        print("\n[1/4] Fitting null model (team + season random effects)...")
        try:
            self.models['null'] = self._fit_null_model(method, maxiter)
            print(f"      Converged: {self.models['null'].converged}")
        except Exception as e:
            print(f"      Failed: {e}")
            self.models['null'] = None

        # Model 2: Fixed weather effects
        print("\n[2/4] Fitting fixed weather effects model...")
        try:
            self.models['fixed_weather'] = self._fit_fixed_weather_model(
                weather_formula, method, maxiter
            )
            print(f"      Converged: {self.models['fixed_weather'].converged}")
        except Exception as e:
            print(f"      Failed: {e}")
            self.models['fixed_weather'] = None

        # Model 3: Park random intercept
        print("\n[3/4] Fitting park random intercept model...")
        try:
            self.models['park_intercept'] = self._fit_park_intercept_model(
                weather_formula, method, maxiter
            )
            print(f"      Converged: {self.models['park_intercept'].converged}")
        except Exception as e:
            print(f"      Failed: {e}")
            self.models['park_intercept'] = None

        # Model 4: Park random slopes
        print("\n[4/4] Fitting park random slopes model...")
        try:
            self.models['park_slopes'] = self._fit_park_slopes_model(
                weather_formula, random_slope_features, method, maxiter
            )
            print(f"      Converged: {self.models['park_slopes'].converged}")
        except Exception as e:
            print(f"      Failed: {e}")
            self.models['park_slopes'] = None

        # Create comparison table
        self._create_comparison_table()

        return self.models

    def _fit_null_model(self, method: str, maxiter: int) -> MixedLMResults:
        """Fit intercept-only model with team and season variance components."""
        # Use away_team as primary grouping, season as variance component
        model = smf.mixedlm(
            f"{self.target} ~ 1",
            data=self._df_train,
            groups="away_team",
            vc_formula={"season": "0 + C(season)"}
        )
        return model.fit(method=method, maxiter=maxiter)

    def _fit_fixed_weather_model(
        self, weather_formula: str, method: str, maxiter: int
    ) -> MixedLMResults:
        """Fit model with weather as fixed effects."""
        model = smf.mixedlm(
            f"{self.target} ~ {weather_formula}",
            data=self._df_train,
            groups="away_team",
            vc_formula={"season": "0 + C(season)"}
        )
        return model.fit(method=method, maxiter=maxiter)

    def _fit_park_intercept_model(
        self, weather_formula: str, method: str, maxiter: int
    ) -> MixedLMResults:
        """Fit model with park random intercept and team/season variance components."""
        # Use home_team as the primary grouping for random intercepts
        # Add away_team and season as variance components
        model = smf.mixedlm(
            f"{self.target} ~ {weather_formula}",
            data=self._df_train,
            groups="home_team",  # Park random intercepts
            re_formula="~1",  # Random intercept for parks
            vc_formula={
                "away_team": "0 + C(away_team)",
                "season": "0 + C(season)"
            }
        )
        return model.fit(method=method, maxiter=maxiter)

    def _fit_park_slopes_model(
        self,
        weather_formula: str,
        random_slope_features: List[str],
        method: str,
        maxiter: int
    ) -> MixedLMResults:
        """Fit model with park-specific weather slopes."""
        # Create random effects formula for slopes
        re_formula = "~" + " + ".join(random_slope_features)

        model = smf.mixedlm(
            f"{self.target} ~ {weather_formula}",
            data=self._df_train,
            groups="home_team",
            re_formula=re_formula,
            vc_formula={
                "away_team": "0 + C(away_team)",
                "season": "0 + C(season)"
            }
        )
        return model.fit(method=method, maxiter=maxiter)

    def _create_comparison_table(self):
        """Create model comparison table with fit statistics."""
        rows = []
        for name, result in self.models.items():
            if result is not None:
                rows.append({
                    'Model': name,
                    'LogLik': result.llf,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'n_params': result.df_modelwc,
                    'Converged': result.converged
                })
        self.comparison_df = pd.DataFrame(rows)
        if len(self.comparison_df) > 0:
            self.comparison_df = self.comparison_df.sort_values('AIC')

    def get_comparison_table(self) -> pd.DataFrame:
        """Return model comparison table sorted by AIC."""
        return self.comparison_df

    def get_fixed_effects(
        self,
        model_name: str = 'park_intercept',
        apply_bonferroni: bool = True
    ) -> pd.DataFrame:
        """
        Extract fixed effects (average weather effects across all parks).

        Parameters
        ----------
        model_name : str
            Which model to extract from
        apply_bonferroni : bool
            Whether to apply Bonferroni correction to p-values for multiple
            comparisons. With 6 weather tests, α = 0.05 / 6 = 0.008.
            Default True.

        Returns
        -------
        pd.DataFrame
            Fixed effects with coefficients, SE, z-values, p-values, and
            optionally Bonferroni-corrected significance
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        fe_df = pd.DataFrame({
            'Parameter': result.fe_params.index,
            'Coefficient': result.fe_params.values,
            'Std_Error': result.bse_fe.values,
            'z_value': result.tvalues.values[:len(result.fe_params)],
            'p_value': result.pvalues.values[:len(result.fe_params)]
        })

        if apply_bonferroni:
            # Exclude intercept from multiple comparison count
            n_tests = len(fe_df) - 1  # -1 for intercept
            if n_tests > 1:
                alpha_corrected = 0.05 / n_tests
                fe_df['bonferroni_alpha'] = alpha_corrected
                fe_df['significant_corrected'] = fe_df['p_value'] < alpha_corrected

                # Mark significance without assuming intercept position
                fe_df.loc[fe_df['Parameter'] == 'Intercept', 'significant_corrected'] = np.nan
                fe_df.loc[fe_df['Parameter'] == 'Intercept', 'bonferroni_alpha'] = np.nan
            else:
                fe_df['significant_corrected'] = fe_df['p_value'] < 0.05
                fe_df['bonferroni_alpha'] = 0.05

        return fe_df

    def get_variance_components(self, model_name: str = 'park_intercept') -> pd.DataFrame:
        """
        Extract variance components (how much variance from each source).

        Parameters
        ----------
        model_name : str
            Which model to extract from

        Returns
        -------
        pd.DataFrame
            Variance components with estimates
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        components = []

        # Group variance (primary random effect)
        group_label = 'Group (home_team)' if model_name in ['park_intercept', 'park_slopes'] else 'Group (away_team)'
        try:
            if hasattr(result.cov_re, 'iloc') and result.cov_re.shape[0] > 0:
                group_var = result.cov_re.iloc[0, 0]
            elif hasattr(result.cov_re, '__getitem__') and len(result.cov_re) > 0:
                group_var = result.cov_re[0, 0]
            else:
                # Try to get from random effects
                re = result.random_effects
                if re:
                    first_re = list(re.values())[0]
                    if hasattr(first_re, '__iter__'):
                        group_var = np.var([list(v.values())[0] if hasattr(v, 'values') else v for v in re.values()])
                    else:
                        group_var = np.var(list(re.values()))
                else:
                    group_var = 0.0
            components.append({
                'Source': group_label,
                'Variance': group_var,
                'Type': 'Random Intercept'
            })
        except Exception:
            components.append({
                'Source': group_label,
                'Variance': 0.0,
                'Type': 'Random Intercept'
            })

        # Variance components
        if hasattr(result, 'vcomp') and result.vcomp is not None:
            vc_names = list(result.vcomp.keys()) if isinstance(result.vcomp, dict) else []
            vc_values = list(result.vcomp.values()) if isinstance(result.vcomp, dict) else result.vcomp
            for i, val in enumerate(vc_values):
                name = vc_names[i] if i < len(vc_names) else f'VC_{i}'
                components.append({
                    'Source': name,
                    'Variance': val,
                    'Type': 'Variance Component'
                })

        # Residual variance
        components.append({
            'Source': 'Residual',
            'Variance': result.scale,
            'Type': 'Residual'
        })

        vc_df = pd.DataFrame(components)

        # Calculate percentage of total
        total_var = vc_df['Variance'].sum()
        if total_var > 0:
            vc_df['Pct_of_Total'] = (vc_df['Variance'] / total_var * 100).round(2)
        else:
            vc_df['Pct_of_Total'] = 0.0

        return vc_df

    def get_random_effects(self, model_name: str = 'park_intercept') -> pd.DataFrame:
        """
        Extract random effects (BLUPs) for each park.

        Parameters
        ----------
        model_name : str
            Which model to extract from (default: 'park_intercept' for stability)

        Returns
        -------
        pd.DataFrame
            Random effects for each group (Park and their intercept/slope effects)
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]
        re_dict = result.random_effects

        rows = []
        for group, effects in re_dict.items():
            row = {'Park': group}
            if isinstance(effects, pd.Series):
                # For park_intercept and park_slopes models, look for:
                # 1. 'Group' key (statsmodels default for random intercept)
                # 2. 'Intercept' key (explicit intercept term)
                # 3. First non-variance-component entry

                intercept_found = False
                slope_effects = {}

                for idx, val in effects.items():
                    idx_str = str(idx)

                    # Skip variance components (away_team, season)
                    if 'away_team' in idx_str or 'season' in idx_str:
                        continue

                    # Check for intercept-like keys
                    if idx_str == 'Group' or idx_str == 'Intercept' or 'home_team' in idx_str:
                        row['Intercept'] = val
                        intercept_found = True
                    else:
                        # This is a random slope effect
                        clean_col = idx_str.replace('[', '_').replace(']', '_')
                        slope_effects[clean_col] = val

                # If no explicit intercept found but we have slope effects,
                # the first entry in cov_re diagonal is the intercept variance
                if not intercept_found:
                    # For models with random slopes, the intercept is usually the first entry
                    non_vc_entries = [(idx, val) for idx, val in effects.items()
                                      if 'away_team' not in str(idx) and 'season' not in str(idx)]
                    if non_vc_entries:
                        # First non-VC entry is typically the intercept
                        row['Intercept'] = non_vc_entries[0][1]
                        intercept_found = True
                        # Rest are slopes
                        for idx, val in non_vc_entries[1:]:
                            clean_col = str(idx).replace('[', '_').replace(']', '_')
                            slope_effects[clean_col] = val

                # Add slope effects
                row.update(slope_effects)

                # Fallback: if still no intercept, use 0.0 with warning
                if not intercept_found:
                    row['Intercept'] = 0.0
                    warnings.warn(f"Could not find intercept for park {group}, using 0.0")

            else:
                # Simple scalar random effect
                row['Intercept'] = effects
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('Park').reset_index(drop=True)

        # Ensure Intercept column exists
        if 'Intercept' not in df.columns:
            df['Intercept'] = 0.0

        return df

    def compute_r_squared(self, model_name: str = 'park_intercept') -> Dict[str, float]:
        """
        Compute marginal and conditional R² for mixed models.

        Marginal R²: Variance explained by fixed effects only
        Conditional R²: Variance explained by fixed + random effects

        Uses Nakagawa & Schielzeth (2013) method.

        Parameters
        ----------
        model_name : str
            Which model to use

        Returns
        -------
        Dict with 'marginal_r2' and 'conditional_r2'
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]

        # Get variance components
        var_fixed = np.var(result.fittedvalues)
        var_residual = result.scale

        # Random effects variance
        var_random = 0
        if hasattr(result, 'cov_re'):
            cov_re = result.cov_re
            if hasattr(cov_re, 'values'):
                var_random += np.trace(cov_re.values)
            else:
                var_random += np.trace(cov_re)

        if hasattr(result, 'vcomp') and result.vcomp is not None:
            if isinstance(result.vcomp, dict):
                var_random += sum(result.vcomp.values())
            else:
                var_random += np.sum(result.vcomp)

        total_var = var_fixed + var_random + var_residual

        marginal_r2 = var_fixed / total_var
        conditional_r2 = (var_fixed + var_random) / total_var

        return {
            'marginal_r2': marginal_r2,
            'conditional_r2': conditional_r2,
            'var_fixed': var_fixed,
            'var_random': var_random,
            'var_residual': var_residual
        }

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str = 'park_intercept',
        include_random: bool = True
    ) -> np.ndarray:
        """
        Generate predictions from a fitted model.

        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        model_name : str
            Which model to use (default: 'park_intercept' for stability)
        include_random : bool
            Whether to include random effects in predictions

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not fitted or failed")

        result = self.models[model_name]

        # Ensure string types for grouping columns
        df_pred = df.copy()
        for col in ['home_team', 'away_team', 'season']:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].astype(str)

        if include_random:
            return result.predict(df_pred)
        else:
            # Fixed effects only - manually compute using coefficients
            # Get fixed effect parameters
            fe_params = result.fe_params

            # Build prediction using fixed effects only
            # Start with intercept
            y_pred = np.full(len(df_pred), fe_params.get('Intercept', 0.0))

            # Add contributions from weather features
            for feature in self.weather_features:
                if feature in df_pred.columns and feature in fe_params.index:
                    y_pred += df_pred[feature].values * fe_params[feature]

            # Add is_night contribution if present
            if 'is_night' in df_pred.columns and 'is_night' in fe_params.index:
                y_pred += df_pred['is_night'].values * fe_params['is_night']

            return y_pred

    def evaluate(
        self,
        df_test: pd.DataFrame,
        model_name: str = 'park_intercept'
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test data
        model_name : str
            Which model to use

        Returns
        -------
        Dict with RMSE, MAE, and R² metrics
        """
        y_true = df_test[self.target].values
        y_pred = self.predict(df_test, model_name)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'n_samples': len(y_true)
        }

    def predict_raw(
        self,
        df: pd.DataFrame,
        model_name: str = 'park_intercept',
        include_random: bool = True
    ) -> np.ndarray:
        """
        Generate predictions on raw scale (for deviation models).

        For deviation models, this adds back the expected values to convert
        deviation predictions to raw scale predictions.

        For raw models, this is equivalent to predict().

        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on (must contain expected_col if deviation model)
        model_name : str
            Which model to use
        include_random : bool
            Whether to include random effects in predictions

        Returns
        -------
        np.ndarray
            Predicted values on raw scale
        """
        deviation_pred = self.predict(df, model_name, include_random)

        if self.target_type == 'deviation':
            if self.expected_col is None:
                raise ValueError("expected_col not set for deviation model")
            if self.expected_col not in df.columns:
                raise ValueError(f"Expected column '{self.expected_col}' not in data")

            expected_values = df[self.expected_col].values
            return deviation_pred + expected_values
        else:
            # For raw models, just return the prediction
            return deviation_pred

    def evaluate_on_raw_scale(
        self,
        df_test: pd.DataFrame,
        model_name: str = 'park_intercept'
    ) -> Dict[str, float]:
        """
        Evaluate model on test data, using raw scale for fair comparison.

        For deviation models, this converts predictions back to raw scale
        before computing metrics. This allows fair comparison with raw models.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test data
        model_name : str
            Which model to use

        Returns
        -------
        Dict with RMSE, MAE, and R² metrics on raw scale
        """
        if self.target_type == 'deviation':
            if self.raw_target is None:
                raise ValueError("raw_target not set for deviation model")
            if self.raw_target not in df_test.columns:
                raise ValueError(f"Raw target column '{self.raw_target}' not in data")

            y_true = df_test[self.raw_target].values
            y_pred = self.predict_raw(df_test, model_name)
        else:
            # For raw models, use standard evaluation
            y_true = df_test[self.target].values
            y_pred = self.predict(df_test, model_name)

        # Filter out NaN values (from missing expected values)
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'n_samples': len(y_true),
            'scale': 'raw'
        }

    def summary(self, model_name: str = 'park_intercept') -> str:
        """Get model summary."""
        if model_name not in self.models or self.models[model_name] is None:
            return f"Model '{model_name}' not fitted or failed"
        return self.models[model_name].summary().as_text()


def fit_mixed_effects_models(
    me_data: Dict[str, Any],
    random_slope_features: List[str] = None,
    use_enhanced_slopes: bool = None
) -> Tuple[MixedEffectsModelFitter, MixedEffectsModelFitter]:
    """
    Convenience function to fit mixed-effects models for both targets.

    Parameters
    ----------
    me_data : Dict
        Output from prepare_mixed_effects_data()
    random_slope_features : List[str], optional
        Weather features for random slopes. If None, uses defaults based on
        whether enhanced features are enabled.
    use_enhanced_slopes : bool, optional
        Whether to use enhanced random slopes (air_density instead of wspd_mph).
        If None, auto-detects from me_data.

    Returns
    -------
    Tuple of (strikeouts_fitter, runs_fitter)
    """
    # Determine if using enhanced features
    is_enhanced = me_data.get('use_enhanced_features', False)
    if use_enhanced_slopes is None:
        use_enhanced_slopes = is_enhanced

    # Set default random slopes based on feature set
    if random_slope_features is None:
        if use_enhanced_slopes and 'air_density' in me_data['weather_features']:
            random_slope_features = ENHANCED_RANDOM_SLOPES
        else:
            random_slope_features = BASIC_RANDOM_SLOPES

    # Filter to only include features that exist in the data
    weather_features = me_data['weather_features']
    random_slope_features = [f for f in random_slope_features if f in weather_features]

    print(f"\nFitting models with:")
    print(f"  Weather features: {weather_features}")
    print(f"  Random slope features: {random_slope_features}")

    # Fit strikeouts model
    k_fitter = MixedEffectsModelFitter(
        target=me_data['target_strikeouts'],
        weather_features=weather_features
    )
    k_fitter.fit_model_hierarchy(
        me_data['df_train'],
        random_slope_features=random_slope_features
    )

    # Fit runs model
    runs_fitter = MixedEffectsModelFitter(
        target=me_data['target_runs'],
        weather_features=weather_features
    )
    runs_fitter.fit_model_hierarchy(
        me_data['df_train'],
        random_slope_features=random_slope_features
    )

    return k_fitter, runs_fitter


def fit_deviation_models(
    me_data: Dict[str, Any],
    random_slope_features: List[str] = None,
    use_enhanced_slopes: bool = None
) -> Tuple[MixedEffectsModelFitter, MixedEffectsModelFitter]:
    """
    Fit mixed-effects models using deviation targets.

    Parameters
    ----------
    me_data : Dict
        Output from prepare_deviation_data()
    random_slope_features : List[str], optional
        Weather features for random slopes
    use_enhanced_slopes : bool, optional
        Whether to use enhanced random slopes

    Returns
    -------
    Tuple of (strikeouts_fitter, runs_fitter) using deviation targets
    """
    # Verify this is deviation data
    if 'deviation_columns' not in me_data:
        raise ValueError("me_data must come from prepare_deviation_data()")

    # Determine if using enhanced features
    is_enhanced = me_data.get('use_enhanced_features', False)
    if use_enhanced_slopes is None:
        use_enhanced_slopes = is_enhanced

    # Set default random slopes based on feature set
    if random_slope_features is None:
        if use_enhanced_slopes and 'air_density' in me_data['weather_features']:
            random_slope_features = ENHANCED_RANDOM_SLOPES
        else:
            random_slope_features = BASIC_RANDOM_SLOPES

    # Filter to only include features that exist in the data
    weather_features = me_data['weather_features']
    random_slope_features = [f for f in random_slope_features if f in weather_features]

    print(f"\nFitting DEVIATION models with:")
    print(f"  Weather features: {weather_features}")
    print(f"  Random slope features: {random_slope_features}")
    print(f"  Target type: deviation")

    # Filter training data to only valid deviations
    df_train = me_data['df_train'].dropna(subset=['deviation_away_k', 'deviation_away_runs']).copy()
    print(f"  Training samples (with valid deviations): {len(df_train)}")

    # Fit strikeouts deviation model
    k_fitter = MixedEffectsModelFitter(
        target='deviation_away_k',
        weather_features=weather_features,
        target_type='deviation',
        expected_col='expected_away_k',
        raw_target='away_bat_k'
    )
    k_fitter.fit_model_hierarchy(
        df_train,
        random_slope_features=random_slope_features
    )

    # Fit runs deviation model
    runs_fitter = MixedEffectsModelFitter(
        target='deviation_away_runs',
        weather_features=weather_features,
        target_type='deviation',
        expected_col='expected_away_runs',
        raw_target='away_runs_scored'
    )
    runs_fitter.fit_model_hierarchy(
        df_train,
        random_slope_features=random_slope_features
    )

    return k_fitter, runs_fitter


def time_series_cross_validate(
    df: pd.DataFrame,
    target: str,
    weather_features: List[str],
    n_splits: int = 5,
    min_train_seasons: int = 2,
    model_type: str = 'park_intercept',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform time-series cross-validation for mixed-effects models.

    Uses expanding window approach where each fold uses all prior data
    for training and one future season for testing. This prevents
    temporal leakage and provides honest performance estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with season column
    target : str
        Target variable name
    weather_features : List[str]
        Weather features to use
    n_splits : int
        Number of CV folds (default: 5)
    min_train_seasons : int
        Minimum seasons required for training (default: 2)
    model_type : str
        Which model to evaluate ('park_intercept' or 'park_slopes')
    verbose : bool
        Whether to print progress

    Returns
    -------
    Dict containing:
        - fold_metrics: List of metrics for each fold
        - mean_rmse: Average RMSE across folds
        - std_rmse: Standard deviation of RMSE
        - mean_r2: Average R² across folds
        - fold_details: Detailed info about each fold
    """
    # Sort by date to ensure temporal ordering
    df = df.sort_values('game_date').copy()

    # Get unique seasons sorted
    seasons = sorted(df['season'].astype(int).unique())

    if len(seasons) < min_train_seasons + 1:
        raise ValueError(f"Need at least {min_train_seasons + 1} seasons, got {len(seasons)}")

    # Determine test seasons (use last n_splits seasons)
    n_available = len(seasons) - min_train_seasons
    actual_splits = min(n_splits, n_available)

    if actual_splits < n_splits:
        warnings.warn(f"Only {actual_splits} splits possible with {len(seasons)} seasons")

    test_seasons = seasons[-actual_splits:]

    fold_metrics = []
    fold_details = []

    for i, test_season in enumerate(test_seasons):
        train_seasons = [s for s in seasons if s < test_season]

        if len(train_seasons) < min_train_seasons:
            continue

        if verbose:
            print(f"\n{'='*50}")
            print(f"CV Fold {i+1}/{actual_splits}")
            print(f"Train: {train_seasons}")
            print(f"Test: {test_season}")
            print(f"{'='*50}")

        # Split data
        train_mask = df['season'].astype(int).isin(train_seasons)
        test_mask = df['season'].astype(int) == test_season

        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        if verbose:
            print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

        # Fit model
        try:
            fitter = MixedEffectsModelFitter(
                target=target,
                weather_features=weather_features
            )

            # Suppress verbose output during CV
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Only fit the model type we're evaluating
                if model_type == 'park_intercept':
                    fitter.models[model_type] = fitter._fit_park_intercept_model(
                        ' + '.join(weather_features) + ' + is_night',
                        'lbfgs', 200
                    )
                else:
                    fitter.models['park_intercept'] = fitter._fit_park_intercept_model(
                        ' + '.join(weather_features) + ' + is_night',
                        'lbfgs', 200
                    )

                fitter._df_train = df_train

            # Evaluate on test
            metrics = fitter.evaluate(df_test, 'park_intercept')

            fold_metrics.append({
                'fold': i + 1,
                'train_seasons': train_seasons,
                'test_season': test_season,
                'n_train': len(df_train),
                'n_test': len(df_test),
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'converged': fitter.models['park_intercept'].converged
            })

            if verbose:
                print(f"RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")

        except Exception as e:
            if verbose:
                print(f"Fold {i+1} failed: {e}")
            fold_metrics.append({
                'fold': i + 1,
                'train_seasons': train_seasons,
                'test_season': test_season,
                'error': str(e)
            })

    # Compute summary statistics
    successful_folds = [f for f in fold_metrics if 'RMSE' in f]

    if len(successful_folds) == 0:
        return {
            'fold_metrics': fold_metrics,
            'error': 'All folds failed'
        }

    rmse_values = [f['RMSE'] for f in successful_folds]
    r2_values = [f['R2'] for f in successful_folds]

    results = {
        'fold_metrics': fold_metrics,
        'mean_rmse': np.mean(rmse_values),
        'std_rmse': np.std(rmse_values),
        'mean_r2': np.mean(r2_values),
        'std_r2': np.std(r2_values),
        'n_successful_folds': len(successful_folds),
        'n_total_folds': len(fold_metrics)
    }

    if verbose:
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successful folds: {results['n_successful_folds']}/{results['n_total_folds']}")
        print(f"Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        print(f"Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        print(f"{'='*60}\n")

    return results


def compare_basic_vs_enhanced(
    df: pd.DataFrame,
    test_start_season: int = 2023
) -> Dict[str, Any]:
    """
    Compare basic weather features vs enhanced physics-based features.

    This function fits models with both feature sets and compares their
    performance to demonstrate the value of the enhanced features.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from load_all_team_data()
    test_start_season : int
        First season for test set

    Returns
    -------
    Dict containing comparison results
    """
    from .data_prep import prepare_mixed_effects_data

    print("=" * 70)
    print("COMPARING BASIC vs ENHANCED WEATHER FEATURES")
    print("=" * 70)

    # Prepare data with basic features
    print("\n[1/4] Preparing data with BASIC features...")
    me_data_basic = prepare_mixed_effects_data(
        df,
        use_enhanced_features=False,
        test_start_season=test_start_season
    )

    # Prepare data with enhanced features
    print("\n[2/4] Preparing data with ENHANCED features...")
    me_data_enhanced = prepare_mixed_effects_data(
        df,
        use_enhanced_features=True,
        include_interactions=True,
        test_start_season=test_start_season
    )

    # Fit basic models
    print("\n[3/4] Fitting BASIC models...")
    k_basic, runs_basic = fit_mixed_effects_models(me_data_basic)

    # Fit enhanced models
    print("\n[4/4] Fitting ENHANCED models...")
    k_enhanced, runs_enhanced = fit_mixed_effects_models(me_data_enhanced)

    # Compare results
    results = {
        'basic': {
            'strikeouts': {},
            'runs': {}
        },
        'enhanced': {
            'strikeouts': {},
            'runs': {}
        }
    }

    # Get R² metrics
    for model_name in ['park_intercept', 'park_slopes']:
        try:
            r2_basic_k = k_basic.compute_r_squared(model_name)
            results['basic']['strikeouts'][model_name] = r2_basic_k
        except Exception:
            pass

        try:
            r2_enhanced_k = k_enhanced.compute_r_squared(model_name)
            results['enhanced']['strikeouts'][model_name] = r2_enhanced_k
        except Exception:
            pass

        try:
            r2_basic_r = runs_basic.compute_r_squared(model_name)
            results['basic']['runs'][model_name] = r2_basic_r
        except Exception:
            pass

        try:
            r2_enhanced_r = runs_enhanced.compute_r_squared(model_name)
            results['enhanced']['runs'][model_name] = r2_enhanced_r
        except Exception:
            pass

    # Test set evaluation
    for fitter, data, label in [
        (k_basic, me_data_basic, 'basic'),
        (k_enhanced, me_data_enhanced, 'enhanced')
    ]:
        try:
            test_metrics = fitter.evaluate(data['df_test'], 'park_slopes')
            results[label]['strikeouts']['test_metrics'] = test_metrics
        except Exception:
            try:
                test_metrics = fitter.evaluate(data['df_test'], 'park_intercept')
                results[label]['strikeouts']['test_metrics'] = test_metrics
            except Exception:
                pass

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\nStrikeouts Model - Marginal R² (weather effects only):")
    for feature_set in ['basic', 'enhanced']:
        for model_name in ['park_intercept', 'park_slopes']:
            if model_name in results[feature_set]['strikeouts']:
                r2 = results[feature_set]['strikeouts'][model_name].get('marginal_r2', 'N/A')
                if isinstance(r2, float):
                    print(f"  {feature_set:10s} {model_name:15s}: {r2:.4f}")

    print("\nRuns Model - Marginal R² (weather effects only):")
    for feature_set in ['basic', 'enhanced']:
        for model_name in ['park_intercept', 'park_slopes']:
            if model_name in results[feature_set]['runs']:
                r2 = results[feature_set]['runs'][model_name].get('marginal_r2', 'N/A')
                if isinstance(r2, float):
                    print(f"  {feature_set:10s} {model_name:15s}: {r2:.4f}")

    results['models'] = {
        'basic': {'strikeouts': k_basic, 'runs': runs_basic},
        'enhanced': {'strikeouts': k_enhanced, 'runs': runs_enhanced}
    }
    results['data'] = {
        'basic': me_data_basic,
        'enhanced': me_data_enhanced
    }

    return results


if __name__ == '__main__':
    from pathlib import Path
    from data_prep import load_all_team_data, prepare_mixed_effects_data
    import argparse

    parser = argparse.ArgumentParser(description='Fit mixed-effects models for weather analysis')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced physics-based features')
    parser.add_argument('--compare', action='store_true',
                        help='Compare basic vs enhanced features')
    args = parser.parse_args()

    # Load and prepare data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    # Run comparison if requested
    if args.compare:
        results = compare_basic_vs_enhanced(df)
        print("\nComparison complete. Results stored in 'results' dict.")
    else:
        # Regular model fitting
        print("\nPreparing mixed-effects data...")
        me_data = prepare_mixed_effects_data(
            df,
            use_enhanced_features=args.enhanced,
            include_interactions=args.enhanced
        )

        print("\nFitting mixed-effects models...")
        k_fitter, runs_fitter = fit_mixed_effects_models(me_data)

        print("\n" + "="*60)
        print("STRIKEOUTS MODEL COMPARISON")
        print("="*60)
        print(k_fitter.get_comparison_table())

        print("\n" + "="*60)
        print("RUNS MODEL COMPARISON")
        print("="*60)
        print(runs_fitter.get_comparison_table())

        # Show fixed effects for best model
        print("\n" + "="*60)
        print("STRIKEOUTS - Fixed Effects (park_slopes model)")
        print("="*60)
        try:
            print(k_fitter.get_fixed_effects('park_slopes'))
        except Exception as e:
            print(f"Could not get fixed effects: {e}")
            # Try a simpler model
            print("\nTrying park_intercept model instead:")
            print(k_fitter.get_fixed_effects('park_intercept'))

        # Variance decomposition
        print("\n" + "="*60)
        print("STRIKEOUTS - Variance Components")
        print("="*60)
        try:
            print(k_fitter.get_variance_components('park_slopes'))
        except Exception as e:
            print(f"Could not get variance components: {e}")
            print(k_fitter.get_variance_components('park_intercept'))

        # R² metrics
        print("\n" + "="*60)
        print("STRIKEOUTS - R² Metrics")
        print("="*60)
        try:
            r2_metrics = k_fitter.compute_r_squared('park_slopes')
            print(f"Marginal R² (fixed effects):     {r2_metrics['marginal_r2']:.4f}")
            print(f"Conditional R² (fixed + random): {r2_metrics['conditional_r2']:.4f}")
        except Exception as e:
            print(f"Could not compute R²: {e}")

        # Test set evaluation
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        for fitter, name in [(k_fitter, 'Strikeouts'), (runs_fitter, 'Runs')]:
            print(f"\n{name}:")
            for model_name in ['fixed_weather', 'park_intercept', 'park_slopes']:
                try:
                    metrics = fitter.evaluate(me_data['df_test'], model_name)
                    print(f"  {model_name}: RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.4f}")
                except Exception as e:
                    print(f"  {model_name}: Failed - {e}")
