# Mixed-Effects Model: Input/Output Specification

This document specifies the exact inputs and outputs at each stage of the mixed-effects modeling pipeline.

---

## Stage 1: Raw Data Loading

**Function**: `load_all_team_data(data_dir)`

### Input
- **30 CSV files** named `{team}_data_{year}_day_night.csv`
- Each file contains game-level data for one team's home games

### Raw CSV Columns (48 total)

| Column | Type | Description |
|--------|------|-------------|
| `game_pk` | int | Unique game identifier |
| `game_date` | date | Date of game |
| `season` | int | Year (2015-2024) |
| `away_team` | str | Away team abbreviation (e.g., "SD", "TEX") |
| `game_start` | datetime | Game start timestamp |
| `start_hour` | int | Hour game started |
| `home_runs_scored` | int | Runs scored by home team |
| `away_runs_scored` | int | **TARGET**: Runs scored by away team |
| `total_runs` | int | Combined runs |
| `home_runs_hit` | int | Home runs hit (total) |
| `strikeouts` | int | Total strikeouts |
| `walks` | int | Total walks |
| `hits` | int | Total hits |
| `total_pitches` | int | Pitch count |
| `avg_exit_velocity` | float | Average exit velocity |
| `n_barrels` | int | Number of barrels |
| `n_bbe` | int | Batted ball events |
| `barrel_rate` | float | Barrel rate |
| `hr_h_ratio` | float | HR to hit ratio |
| `home_bat_hr` | int | Home team HRs |
| `home_bat_k` | int | Home team strikeouts |
| `home_bat_bb` | int | Home team walks |
| `home_bat_h` | int | Home team hits |
| `home_bat_pitches` | int | Pitches seen by home |
| `home_bat_exit_velo` | float | Home exit velocity |
| `home_bat_bbe` | int | Home batted balls |
| `home_bat_hr_h_ratio` | float | Home HR/H ratio |
| `away_bat_hr` | int | Away team HRs |
| `away_bat_k` | int | **TARGET**: Away team strikeouts |
| `away_bat_bb` | int | Away team walks |
| `away_bat_h` | int | Away team hits |
| `away_bat_pitches` | int | Pitches seen by away |
| `away_bat_exit_velo` | float | Away exit velocity |
| `away_bat_bbe` | int | Away batted balls |
| `away_bat_hr_h_ratio` | float | Away HR/H ratio |
| `temp_f` | float | **FEATURE**: Temperature (°F) |
| `temp_c` | float | Temperature (°C) |
| `rhum` | float | **FEATURE**: Relative humidity (%) |
| `pres` | float | Atmospheric pressure |
| `prcp` | float | Precipitation |
| `wspd` | float | Wind speed (m/s) |
| `wspd_mph` | float | **FEATURE**: Wind speed (mph) |
| `wdir` | float | Wind direction (degrees) |
| `wind_dir_bucket` | str | Wind direction category |
| `wind_cf` | float | **FEATURE**: Wind component toward CF |
| `wind_lcf` | float | **FEATURE**: Wind component toward LCF |
| `wind_rcf` | float | **FEATURE**: Wind component toward RCF |
| `day_night` | str | "day" or "night" |

### Output
- **DataFrame**: Combined data from all 30 teams
- **Shape**: ~10,000+ rows × 48 columns
- **Added column**: `home_team` (derived from filename, e.g., "SF", "NYY")

---

## Stage 2: Mixed-Effects Data Preparation

**Function**: `prepare_mixed_effects_data(df, standardize_weather=True, test_start_season=2023)`

### Input
- Combined DataFrame from Stage 1

### Processing Steps

1. **Drop rows with missing values** in required columns
2. **Create binary `is_night`**: `1` if `day_night == 'night'`, else `0`
3. **Convert grouping variables to string type** (statsmodels compatibility)
4. **Split by season**: Train (season < 2023), Test (season ≥ 2023)
5. **Standardize weather features** (z-score normalization, fit on train only)

### Output Dictionary

```python
{
    'df_train': pd.DataFrame,      # Training data
    'df_test': pd.DataFrame,       # Test data
    'weather_scaler': StandardScaler,  # Fitted scaler
    'group_info': {
        'home_teams': ['ARI', 'ATL', ..., 'WSH'],  # 30 parks
        'away_teams': ['ARI', 'ATL', ..., 'WSH'],  # 29-30 teams
        'seasons': ['2015', '2016', ..., '2022'],  # Training seasons
        'n_home_teams': 30,
        'n_away_teams': 29,
        'n_seasons': 8
    },
    'weather_features': ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'wind_lcf', 'wind_rcf'],
    'target_strikeouts': 'away_bat_k',
    'target_runs': 'away_runs_scored'
}
```

### Columns Used in Model

| Column | Role | Values |
|--------|------|--------|
| `temp_f` | Fixed effect (continuous) | Standardized (mean=0, std=1) |
| `rhum` | Fixed effect (continuous) | Standardized |
| `wspd_mph` | Fixed effect (continuous) | Standardized |
| `wind_cf` | Fixed effect (continuous) | Standardized |
| `wind_lcf` | Fixed effect (continuous) | Standardized |
| `wind_rcf` | Fixed effect (continuous) | Standardized |
| `is_night` | Fixed effect (binary) | 0 or 1 |
| `home_team` | Random effect grouping | 30 categories (e.g., "SF") |
| `away_team` | Variance component | 29 categories |
| `season` | Variance component | 8 categories ("2015"-"2022") |
| `away_bat_k` | Target variable | Integer (0-20+) |
| `away_runs_scored` | Target variable | Integer (0-20+) |

---

## Stage 3: Model Fitting (4-Model Hierarchy)

**Class**: `MixedEffectsModelFitter`

### Model 1: Null Model

**Formula**:
```
away_bat_k ~ 1
```

**Input to statsmodels**:
- `endog` (y): `away_bat_k` — shape (n_train,)
- `exog` (X): Intercept only — shape (n_train, 1)
- `groups`: `away_team` — categorical with 29 levels
- `vc_formula`: `{"season": "0 + C(season)"}` — 8 season dummies

**Output**:
- `MixedLMResults` object containing:
  - `llf`: Log-likelihood (float)
  - `aic`: Akaike Information Criterion (float)
  - `bic`: Bayesian Information Criterion (float)
  - `fe_params`: Fixed effects (just intercept) — shape (1,)
  - `random_effects`: Dict of {team: intercept_deviation}
  - `vcomp`: Variance components {season: variance}
  - `scale`: Residual variance (float)

---

### Model 2: Fixed Weather

**Formula**:
```
away_bat_k ~ temp_f + rhum + wspd_mph + wind_cf + is_night
```

**Input to statsmodels**:
- `endog` (y): `away_bat_k` — shape (n_train,)
- `exog` (X): Design matrix — shape (n_train, 6)
  ```
  [Intercept, temp_f, rhum, wspd_mph, wind_cf, is_night]
  ```
- `groups`: `away_team` — 29 levels
- `vc_formula`: `{"season": "0 + C(season)"}` — 8 dummies

**Output**:
- `fe_params`: Fixed effects — shape (6,)
  ```
  [β_intercept, β_temp_f, β_rhum, β_wspd_mph, β_wind_cf, β_is_night]
  ```
- `bse_fe`: Standard errors — shape (6,)
- `tvalues`: t-statistics — shape (6,)
- `pvalues`: p-values — shape (6,)

---

### Model 3: Park Intercept (Primary Model)

**Formula**:
```
away_bat_k ~ temp_f + rhum + wspd_mph + wind_cf + is_night + (1 | home_team)
```

**Input to statsmodels**:
- `endog` (y): `away_bat_k` — shape (n_train,)
- `exog` (X): Design matrix — shape (n_train, 6)
- `groups`: `home_team` — **30 parks** (primary grouping)
- `re_formula`: `"~1"` — random intercept per park
- `vc_formula`:
  ```python
  {
      "away_team": "0 + C(away_team)",  # 29 team dummies
      "season": "0 + C(season)"          # 8 season dummies
  }
  ```

**Output**:

| Output | Shape | Description |
|--------|-------|-------------|
| `fe_params` | (6,) | Population-average weather effects |
| `bse_fe` | (6,) | Standard errors of fixed effects |
| `pvalues` | (6,) | p-values for significance testing |
| `random_effects` | dict | {park: Series([intercept_deviation])} |
| `cov_re` | (1, 1) | Variance of park random intercepts |
| `vcomp` | dict | {"away_team": var, "season": var} |
| `scale` | float | Residual variance |
| `fittedvalues` | (n_train,) | Predicted values |
| `converged` | bool | Whether optimization converged |

**Example `fe_params`**:
```
Intercept     8.12   # Average strikeouts (league-wide)
temp_f       -0.10   # 1 SD warmer → -0.10 K's
rhum         -0.13   # 1 SD more humid → -0.13 K's
wspd_mph      0.02   # 1 SD faster wind → +0.02 K's (n.s.)
wind_cf       0.05   # Wind toward CF → +0.05 K's
is_night     -0.21   # Night game → -0.21 K's
```

**Example `random_effects`**:
```python
{
    'SF': Series({'Group': -0.28}),   # Oracle Park: 0.28 fewer K's than average
    'COL': Series({'Group': -0.85}),  # Coors Field: 0.85 fewer K's (altitude)
    'HOU': Series({'Group': +0.72}),  # Minute Maid: 0.72 more K's
    ...
}
```

---

### Model 4: Park Slopes

**Formula**:
```
away_bat_k ~ temp_f + rhum + wspd_mph + wind_cf + is_night +
             (1 + temp_f + wspd_mph | home_team)
```

**Input to statsmodels**:
- Same as Model 3, plus:
- `re_formula`: `"~ temp_f + wspd_mph"` — random slopes

**Output** (additional):
- `cov_re`: (3, 3) covariance matrix of random effects
  ```
              Intercept  temp_f  wspd_mph
  Intercept      0.15    0.02     0.01
  temp_f         0.02    0.03     0.00
  wspd_mph       0.01    0.00     0.02
  ```
- `random_effects`: Dict with per-park intercept AND slopes
  ```python
  {
      'SF': Series({
          'Group': -0.28,      # Park intercept
          'temp_f': +0.05,     # SF-specific temp effect
          'wspd_mph': -0.03    # SF-specific wind effect
      }),
      ...
  }
  ```

---

## Stage 4: Variance Decomposition

**Method**: `get_variance_components(model_name)`

### Input
- Fitted `MixedLMResults` object

### Output

| Source | Value | Calculation |
|--------|-------|-------------|
| Group (home_team) | `cov_re[0,0]` | Variance of park intercepts |
| away_team | `vcomp['away_team']` | Team batting variance |
| season | `vcomp['season']` | Year-to-year variance |
| Residual | `scale` | Unexplained variance |
| **Total** | Sum of above | |
| Pct_of_Total | `(var / total) * 100` | Percentage breakdown |

**Example Output**:
```
          Source   Variance  Pct_of_Total
0  Group (home_team)   0.52          5.2%
1        away_team    0.15          1.5%
2           season    0.41          4.1%
3         Residual    8.92         89.2%
```

---

## Stage 5: R² Computation

**Method**: `compute_r_squared(model_name)`

### Input
- Fitted model results

### Output

```python
{
    'marginal_r2': 0.08,      # Variance explained by fixed effects
    'conditional_r2': 0.18,   # Variance explained by fixed + random
    'var_fixed': 0.82,        # Variance of predictions (fixed only)
    'var_random': 1.08,       # Variance from random effects
    'var_residual': 8.92      # Residual variance
}
```

**Formulas**:
```
Marginal R² = var_fixed / (var_fixed + var_random + var_residual)
Conditional R² = (var_fixed + var_random) / (var_fixed + var_random + var_residual)
```

---

## Stage 6: Prediction

**Method**: `predict(df, model_name, include_random=True)`

### Input
- `df`: DataFrame with columns: `temp_f`, `rhum`, `wspd_mph`, `wind_cf`, `is_night`, `home_team`, `away_team`, `season`

### Output
- `np.ndarray` of shape (n_samples,) with predicted strikeouts

### Calculation
```
ŷ = β₀ + β₁·temp_f + β₂·rhum + β₃·wspd_mph + β₄·wind_cf + β₅·is_night + b_park
```

Where:
- `β₀...β₅` are fixed effects (same for all parks)
- `b_park` is the park-specific random effect (different for each park)

---

## Stage 7: Test Set Evaluation

**Method**: `evaluate(df_test, model_name)`

### Input
- `df_test`: Test DataFrame (2023+ games)

### Output

```python
{
    'RMSE': 2.45,     # Root Mean Squared Error
    'MAE': 1.89,      # Mean Absolute Error
    'R2': 0.08,       # Test set R²
    'n_samples': 2500 # Number of test games
}
```

---

## Summary: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA (30 CSVs)                            │
│  48 columns × ~350 games each = ~10,500 total rows                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEATURE EXTRACTION                              │
│  Keep: temp_f, rhum, wspd_mph, wind_cf, is_night, home_team,           │
│        away_team, season, away_bat_k, away_runs_scored                  │
│  Shape: (10,000, 10)                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRAIN/TEST SPLIT                               │
│  Train: season ≤ 2022 (~7,500 games)                                    │
│  Test:  season ≥ 2023 (~2,500 games)                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STANDARDIZATION                                 │
│  Weather features: z = (x - μ_train) / σ_train                          │
│  Grouping vars: Convert to object dtype                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MIXED-EFFECTS MODEL FIT                            │
│  Fixed effects (X):  [1, temp_f, rhum, wspd_mph, wind_cf, is_night]    │
│  Random effects:     (1 | home_team)                                    │
│  Variance components: away_team (29), season (8)                        │
│  Target (y):         away_bat_k                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODEL OUTPUTS                                 │
│  • fe_params (6): Population-average weather effects                    │
│  • random_effects (30): Park-specific intercept deviations              │
│  • vcomp (2): Team and season variance                                  │
│  • scale (1): Residual variance                                         │
│  • R² metrics: marginal=0.08, conditional=0.18                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Concrete Example: Single Prediction

**Input row**:
```
temp_f: 65°F → standardized: (65 - 70) / 12 = -0.42
rhum: 80% → standardized: (80 - 65) / 15 = +1.00
wspd_mph: 10 mph → standardized: (10 - 8) / 5 = +0.40
wind_cf: 5 → standardized: (5 - 3) / 8 = +0.25
is_night: 1 (night game)
home_team: "SF"
```

**Calculation**:
```
ŷ = 8.12                    # Intercept
  + (-0.10) × (-0.42)       # temp_f effect = +0.042
  + (-0.13) × (+1.00)       # rhum effect = -0.130
  + (0.02) × (+0.40)        # wspd_mph effect = +0.008
  + (0.05) × (+0.25)        # wind_cf effect = +0.0125
  + (-0.21) × (1)           # is_night effect = -0.210
  + (-0.28)                 # SF park effect

ŷ = 8.12 + 0.042 - 0.130 + 0.008 + 0.0125 - 0.210 - 0.28
ŷ = 7.56 strikeouts predicted
```
