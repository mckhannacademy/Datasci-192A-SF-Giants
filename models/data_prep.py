"""
Data preparation module for away team regression models.
Loads all team CSV files, combines them, and prepares features for modeling.

Enhanced with physics-based weather features (air density, heat index) and
stadium characteristics (elevation, retractable roof status).

================================================================================
WEATHER FEATURE DOCUMENTATION
================================================================================

This module prepares weather features for mixed-effects models that predict
away team strikeouts and runs. Understanding what each feature measures is
critical for interpreting model coefficients.

BASIC WEATHER FEATURES:
------------------------------------------------------------------------
| Feature          | Description                      | Unit / Range     |
|------------------|----------------------------------|------------------|
| temp_f           | Air temperature                  | Fahrenheit       |
| rhum             | Relative humidity                | 0-100 (percent)  |
| wspd_mph         | Wind speed magnitude             | mph              |
| wind_cf          | Wind component toward CF         | -1 to +1         |
| wind_lcf         | Wind component toward LCF        | -1 to +1         |
| wind_rcf         | Wind component toward RCF        | -1 to +1         |
| air_density      | Air density (computed)           | kg/m³            |
| wind_out_impact  | wspd_mph × wind_cf (interaction) | mph (signed)     |
------------------------------------------------------------------------

HOW WIND DIRECTION IS TREATED:
Wind is decomposed into directional components rather than being a single
speed+direction pair:
- wspd_mph = overall wind speed (how hard it's blowing)
- wind_cf, wind_lcf, wind_rcf = how much wind helps/hurts ball flight

Positive values mean wind is blowing OUT (helps home runs)
Negative values mean wind is blowing IN (suppresses home runs)

WHY INCLUDE BOTH wind_cf AND wind_out_impact?
- wind_cf alone assumes linear effect: 20 mph wind = 4x effect of 5 mph wind
- wind_out_impact = wspd_mph × wind_cf captures that strong winds matter MORE
- A light breeze blowing out has minimal effect
- A strong wind blowing out significantly helps home runs (non-linear)

AIR DENSITY - THE KEY PHYSICS DRIVER:
Lower air density = ball travels farther (less air resistance)
Air density is affected by:
1. Temperature: Hot air is less dense
2. Humidity: Moist air is slightly less dense (counterintuitive!)
3. Pressure: Lower pressure = less dense (altitude effect)
4. Altitude: Coors Field (~5280 ft) has ~17% lower air density

ENHANCED FEATURES (optional):
------------------------------------------------------------------------
| Feature          | Description                      | Why Useful       |
|------------------|----------------------------------|------------------|
| heat_index       | Apparent temperature             | Player fatigue   |
| elevation_ft     | Stadium elevation                | Baseline density |
| has_roof         | Retractable roof indicator       | Weather nullified|
| density_ratio    | Actual/expected density          | Weather anomalies|
------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
import warnings


# ============================================================================
# PHYSICS-BASED WEATHER FEATURE FUNCTIONS
# ============================================================================

def compute_air_density(temp_f: np.ndarray, rhum: np.ndarray, pres: np.ndarray = None) -> np.ndarray:
    """
    Compute air density from temperature, humidity, and pressure.

    Air density affects ball flight - lower density = ball travels further.
    At Coors Field (5280 ft), air density is ~17% lower than sea level.

    Parameters
    ----------
    temp_f : np.ndarray
        Temperature in Fahrenheit
    rhum : np.ndarray
        Relative humidity (0-100)
    pres : np.ndarray, optional
        Pressure in hPa. If None, uses standard sea level pressure (1013.25 hPa)

    Returns
    -------
    np.ndarray
        Air density in kg/m³
    """
    # Convert temperature to Kelvin
    temp_k = (temp_f - 32) * 5/9 + 273.15

    # Use standard pressure if not provided
    if pres is None:
        pres = np.full_like(temp_f, 1013.25)

    # Saturation vapor pressure (Tetens formula)
    temp_c = (temp_f - 32) * 5/9
    e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))  # hPa

    # Actual vapor pressure
    e = (rhum / 100) * e_sat  # hPa

    # Air density formula (simplified ideal gas law with humidity correction)
    # Moist air is less dense than dry air
    R_d = 287.05  # J/(kg·K) - gas constant for dry air
    R_v = 461.5   # J/(kg·K) - gas constant for water vapor

    # Partial pressure of dry air
    p_d = (pres - e) * 100  # Convert to Pa

    # Density of dry air component
    rho_d = p_d / (R_d * temp_k)

    # Density of water vapor component
    rho_v = (e * 100) / (R_v * temp_k)

    # Total air density
    rho = rho_d + rho_v

    return rho


def compute_heat_index(temp_f: np.ndarray, rhum: np.ndarray) -> np.ndarray:
    """
    Compute heat index (apparent temperature) from temperature and humidity.

    Heat index affects player fatigue and performance. High heat index
    can lead to reduced performance, especially late in games.

    Uses the Rothfusz regression equation (NWS method).

    Parameters
    ----------
    temp_f : np.ndarray
        Temperature in Fahrenheit
    rhum : np.ndarray
        Relative humidity (0-100)

    Returns
    -------
    np.ndarray
        Heat index in Fahrenheit
    """
    # Ensure arrays
    T = np.asarray(temp_f, dtype=float)
    R = np.asarray(rhum, dtype=float)

    # Initialize with simple formula for low temps
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))

    # Use Rothfusz regression for temps >= 80°F
    mask = ((HI + T) / 2) >= 80

    if np.any(mask):
        T_m = T[mask]
        R_m = R[mask]

        HI_full = (
            -42.379
            + 2.04901523 * T_m
            + 10.14333127 * R_m
            - 0.22475541 * T_m * R_m
            - 0.00683783 * T_m**2
            - 0.05481717 * R_m**2
            + 0.00122874 * T_m**2 * R_m
            + 0.00085282 * T_m * R_m**2
            - 0.00000199 * T_m**2 * R_m**2
        )

        # Adjustments for extreme conditions
        # Low humidity adjustment
        low_rh = R_m < 13
        high_temp = (T_m > 80) & (T_m < 112)
        adj_mask = low_rh & high_temp
        if np.any(adj_mask):
            adj = ((13 - R_m[adj_mask]) / 4) * np.sqrt((17 - np.abs(T_m[adj_mask] - 95)) / 17)
            HI_full[adj_mask] -= adj

        # High humidity adjustment
        high_rh = R_m > 85
        temp_range = (T_m > 80) & (T_m < 87)
        adj_mask2 = high_rh & temp_range
        if np.any(adj_mask2):
            adj = ((R_m[adj_mask2] - 85) / 10) * ((87 - T_m[adj_mask2]) / 5)
            HI_full[adj_mask2] += adj

        HI[mask] = HI_full

    return HI


def compute_altitude_adjusted_density(
    air_density: np.ndarray,
    elevation_ft: np.ndarray
) -> np.ndarray:
    """
    Adjust air density for elevation effects.

    This creates a normalized air density that accounts for the
    base elevation of each stadium.

    Parameters
    ----------
    air_density : np.ndarray
        Computed air density
    elevation_ft : np.ndarray
        Stadium elevation in feet

    Returns
    -------
    np.ndarray
        Elevation-normalized air density factor
    """
    # Air density decreases approximately 3% per 1000 ft
    # Standard sea level density is ~1.225 kg/m³
    SEA_LEVEL_DENSITY = 1.225

    # Barometric formula approximation for density ratio
    # density_ratio = exp(-elevation / H) where H ≈ 29,000 ft for density
    H = 29000  # Scale height in feet

    expected_density = SEA_LEVEL_DENSITY * np.exp(-elevation_ft / H)

    # Ratio of actual to expected (captures weather deviation from typical)
    density_ratio = air_density / expected_density

    return density_ratio

# Mapping from file name prefix to standard team abbreviation (park code)
TEAM_FILE_TO_ABBREV = {
    'angels': 'LAA',
    'astros': 'HOU',
    'athletics': 'OAK',
    'blue_jays': 'TOR',
    'braves': 'ATL',
    'brewers': 'MIL',
    'cardinals': 'STL',
    'cubs': 'CHC',
    'dbacks': 'ARI',
    'dodgers': 'LAD',
    'fenway': 'BOS',  # Red Sox
    'giants': 'SF',
    'guardians': 'CLE',
    'mariners': 'SEA',
    'marlins': 'MIA',
    'mets': 'NYM',
    'nationals': 'WSH',
    'orioles': 'BAL',
    'padres': 'SD',
    'phillies': 'PHI',
    'pirates': 'PIT',
    'rangers': 'TEX',
    'rays': 'TB',
    'reds': 'CIN',
    'rockies': 'COL',
    'royals': 'KC',
    'tigers': 'DET',
    'twins': 'MIN',
    'white_sox': 'CWS',
    'yankee': 'NYY',
}

# Weather features to use (simplified - removed wind_lcf/wind_rcf due to r>0.94 with wind_cf)
# Multicollinearity fix: only keep wind_cf as the representative wind direction feature
WEATHER_FEATURES_BASIC = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']

# Enhanced weather features (physics-based)
WEATHER_FEATURES_ENHANCED = [
    'temp_f', 'rhum', 'wspd_mph', 'wind_cf',
    'air_density', 'heat_index'
]

# Full feature set including stadium characteristics
WEATHER_FEATURES_FULL = [
    'temp_f', 'rhum', 'wspd_mph', 'wind_cf',
    'air_density', 'heat_index',
    'elevation_ft', 'has_roof'
]

# Default to basic for backwards compatibility
WEATHER_FEATURES = WEATHER_FEATURES_BASIC

# Target variables
TARGET_STRIKEOUTS = 'away_bat_k'
TARGET_RUNS = 'away_runs_scored'


def load_stadium_parameters(params_path: str = None) -> pd.DataFrame:
    """
    Load stadium parameters including elevation and roof status.

    Parameters
    ----------
    params_path : str, optional
        Path to team_parameters.csv. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Stadium parameters indexed by team_code
    """
    if params_path is None:
        # Try common locations
        possible_paths = [
            Path(__file__).parent.parent / 'analysis' / 'team_parameters.csv',
            Path('analysis/team_parameters.csv'),
            Path('../analysis/team_parameters.csv'),
        ]
        for p in possible_paths:
            if p.exists():
                params_path = p
                break
        if params_path is None:
            raise FileNotFoundError("Could not find team_parameters.csv")

    params_df = pd.read_csv(params_path)

    # Ensure required columns exist
    required_cols = ['team_code', 'elevation_ft', 'has_roof']
    missing = [c for c in required_cols if c not in params_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in team_parameters.csv: {missing}")

    return params_df.set_index('team_code')


def add_enhanced_weather_features(
    df: pd.DataFrame,
    stadium_params: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Add physics-based weather features to the dataset.

    Computes:
    - air_density: Air density based on temp, humidity, and pressure
    - heat_index: Apparent temperature accounting for humidity
    - elevation_ft: Stadium elevation (merged from params)
    - has_roof: Whether stadium has retractable roof (merged from params)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with weather columns (temp_f, rhum) and home_team
    stadium_params : pd.DataFrame, optional
        Stadium parameters. If None, loads from default location.

    Returns
    -------
    pd.DataFrame
        Dataset with enhanced features added
    """
    data = df.copy()

    # Convert string columns to object dtype (required for statsmodels compatibility)
    # This fixes issues with pandas StringDtype not being understood by patsy
    for col in ['home_team', 'away_team', 'season']:
        if col in data.columns:
            data[col] = data[col].astype(str).astype('object')

    # Check for required columns
    if 'temp_f' not in data.columns or 'rhum' not in data.columns:
        raise ValueError("Dataset must contain 'temp_f' and 'rhum' columns")

    # Compute air density
    pres = data['pres'].values if 'pres' in data.columns else None
    data['air_density'] = compute_air_density(
        data['temp_f'].values,
        data['rhum'].values,
        pres
    )

    # Compute heat index
    data['heat_index'] = compute_heat_index(
        data['temp_f'].values,
        data['rhum'].values
    )

    # Load and merge stadium parameters
    if stadium_params is None:
        try:
            stadium_params = load_stadium_parameters()
        except FileNotFoundError:
            warnings.warn("Could not load stadium parameters. Elevation/roof features not added.")
            return data

    # Merge stadium parameters
    if 'home_team' in data.columns:
        # Handle team code mapping (OAK vs ATH, ARI vs AZ)
        team_mapping = {
            'OAK': 'ATH',  # Oakland Athletics historical code
            'ARI': 'AZ',   # Arizona Diamondbacks
        }

        # Create lookup column with standardized codes
        lookup_team = data['home_team'].replace(team_mapping)

        # Merge elevation
        if 'elevation_ft' in stadium_params.columns:
            data['elevation_ft'] = lookup_team.map(stadium_params['elevation_ft'])

        # Merge roof status
        if 'has_roof' in stadium_params.columns:
            data['has_roof'] = lookup_team.map(stadium_params['has_roof']).fillna(0).astype(int)

        # Compute altitude-adjusted density if elevation is available
        if 'elevation_ft' in data.columns and not data['elevation_ft'].isna().all():
            data['density_ratio'] = compute_altitude_adjusted_density(
                data['air_density'].values,
                data['elevation_ft'].fillna(0).values
            )

    return data


def load_all_team_data(data_dir: str = 'data') -> pd.DataFrame:
    """
    Load all team CSV files and combine into a single DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing team CSV files

    Returns
    -------
    pd.DataFrame
        Combined dataset with home_team column added
    """
    data_path = Path(data_dir)
    all_dfs = []

    for csv_file in data_path.glob('*_data_*_day_night.csv'):
        # Extract team name from filename (e.g., 'giants_data_2020_day_night.csv' -> 'giants')
        filename = csv_file.stem  # removes .csv
        team_prefix = filename.split('_data_')[0]

        if team_prefix not in TEAM_FILE_TO_ABBREV:
            warnings.warn(f"Unknown team prefix: {team_prefix} in file {csv_file}")
            continue

        team_abbrev = TEAM_FILE_TO_ABBREV[team_prefix]

        # Load the CSV
        df = pd.read_csv(csv_file)
        df['home_team'] = team_abbrev
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No team CSV files found in {data_dir}")

    # Combine all DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure game_date is datetime
    combined['game_date'] = pd.to_datetime(combined['game_date'])

    # Convert string columns to object dtype for statsmodels compatibility
    for col in ['home_team', 'away_team']:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).astype('object')

    # Sort by date
    combined = combined.sort_values('game_date').reset_index(drop=True)

    print(f"Loaded {len(combined)} games from {len(all_dfs)} team files")
    print(f"Date range: {combined['game_date'].min()} to {combined['game_date'].max()}")
    print(f"Seasons: {sorted(combined['season'].unique())}")

    return combined


def prepare_features(
    df: pd.DataFrame,
    include_interactions: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare feature matrix and target variables.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from load_all_team_data
    include_interactions : bool
        Whether to include weather × park interaction terms

    Returns
    -------
    Tuple containing:
        - X: Feature DataFrame (weather + day_night + park one-hot)
        - y_strikeouts: Away team strikeouts
        - y_runs: Away team runs
    """
    # Make a copy to avoid modifying original
    data = df.copy()

    # Drop rows with missing values in key columns
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = data.dropna(subset=required_cols)

    # Extract weather features
    X_weather = data[WEATHER_FEATURES].copy()

    # Encode day_night as binary (night = 1, day = 0)
    X_weather['is_night'] = (data['day_night'] == 'night').astype(int)

    # One-hot encode parks
    parks = np.array(data['home_team'].tolist()).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    park_encoded = encoder.fit_transform(parks)
    park_columns = [f'park_{cat}' for cat in encoder.categories_[0]]
    X_park = pd.DataFrame(park_encoded, columns=park_columns, index=data.index)

    # Combine features
    X = pd.concat([X_weather.reset_index(drop=True), X_park.reset_index(drop=True)], axis=1)

    # Add interaction terms if requested
    if include_interactions:
        interaction_data = {}
        for weather_feat in WEATHER_FEATURES:
            for park_col in park_columns:
                interaction_name = f'{weather_feat}_x_{park_col}'
                interaction_data[interaction_name] = X[weather_feat].values * X[park_col].values
        X = pd.concat([X, pd.DataFrame(interaction_data, index=X.index)], axis=1)

    # Extract targets
    y_strikeouts = data[TARGET_STRIKEOUTS].reset_index(drop=True)
    y_runs = data[TARGET_RUNS].reset_index(drop=True)

    # Store metadata
    X.attrs['park_columns'] = park_columns
    X.attrs['weather_features'] = WEATHER_FEATURES
    X.attrs['park_encoder'] = encoder

    return X, y_strikeouts, y_runs


def train_test_split_by_season(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_strikeouts: pd.Series,
    y_runs: pd.Series,
    test_start_season: int = 2023
) -> Dict[str, np.ndarray]:
    """
    Split data into train/test based on season.

    Parameters
    ----------
    df : pd.DataFrame
        Original combined dataset (for season column)
    X : pd.DataFrame
        Feature matrix
    y_strikeouts : pd.Series
        Strikeouts target
    y_runs : pd.Series
        Runs target
    test_start_season : int
        First season to include in test set

    Returns
    -------
    Dict with train/test splits for X and both targets
    """
    # Get the cleaned data's season column (aligned with X)
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = df.dropna(subset=required_cols).reset_index(drop=True)
    seasons = data['season']

    train_mask = seasons < test_start_season
    test_mask = seasons >= test_start_season

    splits = {
        'X_train': X[train_mask].values,
        'X_test': X[test_mask].values,
        'y_train_strikeouts': y_strikeouts[train_mask].values,
        'y_test_strikeouts': y_strikeouts[test_mask].values,
        'y_train_runs': y_runs[train_mask].values,
        'y_test_runs': y_runs[test_mask].values,
        'feature_names': X.columns.tolist(),
        'train_seasons': sorted(seasons[train_mask].unique()),
        'test_seasons': sorted(seasons[test_mask].unique()),
    }

    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(splits['X_train'])} games (seasons {splits['train_seasons']})")
    print(f"  Testing:  {len(splits['X_test'])} games (seasons {splits['test_seasons']})")

    return splits


def create_random_month_split(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Create train/test split by randomly assigning year-month units.

    Each year-month combination (e.g., "2021-06", "2022-08") is treated as a
    separate unit and randomly assigned to either train or test set. This tests
    whether the model generalizes across time rather than only forward in time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'game_date' column (datetime)
    test_size : float
        Proportion of month-units to assign to test set (default: 0.30 = 30%)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Tuple containing:
        - train_mask: Boolean array for training rows
        - test_mask: Boolean array for test rows
        - train_months: List of year-month strings in train set
        - test_months: List of year-month strings in test set

    Example
    -------
    >>> train_mask, test_mask, train_months, test_months = create_random_month_split(df)
    >>> print(f"Train months: {len(train_months)}, Test months: {len(test_months)}")
    """
    # Ensure game_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])

    # Create year-month identifier for each game
    year_months = df['game_date'].dt.to_period('M').astype(str)

    # Get unique year-month units
    unique_months = sorted(year_months.unique())
    n_months = len(unique_months)
    n_test = int(np.ceil(n_months * test_size))
    n_train = n_months - n_test

    # Randomly assign months to train/test
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(n_months)

    train_month_indices = shuffled_indices[:n_train]
    test_month_indices = shuffled_indices[n_train:]

    train_months = [unique_months[i] for i in sorted(train_month_indices)]
    test_months = [unique_months[i] for i in sorted(test_month_indices)]

    # Create boolean masks
    train_mask = year_months.isin(train_months).values
    test_mask = year_months.isin(test_months).values

    print(f"\nRandom Month Split (seed={random_state}):")
    print(f"  Total month-units: {n_months}")
    print(f"  Train months: {n_train} ({100*n_train/n_months:.1f}%)")
    print(f"  Test months: {n_test} ({100*n_test/n_months:.1f}%)")
    print(f"  Train games: {train_mask.sum()}")
    print(f"  Test games: {test_mask.sum()}")
    print(f"\n  Train months sample: {train_months[:5]}...")
    print(f"  Test months sample: {test_months[:5]}...")

    return train_mask, test_mask, train_months, test_months


def prepare_nn_data(
    df: pd.DataFrame,
    test_start_season: int = 2023
) -> Dict[str, np.ndarray]:
    """
    Prepare data specifically for neural network with park embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from load_all_team_data
    test_start_season : int
        First season to include in test set

    Returns
    -------
    Dict with separate arrays for weather features and park IDs
    """
    # Drop rows with missing values
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = df.dropna(subset=required_cols).reset_index(drop=True)

    # Weather features + day_night
    X_weather = data[WEATHER_FEATURES].copy()
    X_weather['is_night'] = (data['day_night'] == 'night').astype(int)

    # Create park ID mapping
    unique_parks = sorted(data['home_team'].unique())
    park_to_id = {park: i for i, park in enumerate(unique_parks)}
    park_ids = data['home_team'].map(park_to_id).values

    # Targets
    y_strikeouts = data[TARGET_STRIKEOUTS].values
    y_runs = data[TARGET_RUNS].values

    # Train/test split
    seasons = data['season']
    train_mask = seasons < test_start_season
    test_mask = seasons >= test_start_season

    # Standardize weather features (fit on train only)
    scaler = StandardScaler()
    X_weather_train = scaler.fit_transform(X_weather[train_mask])
    X_weather_test = scaler.transform(X_weather[test_mask])

    nn_data = {
        'X_weather_train': X_weather_train.astype(np.float32),
        'X_weather_test': X_weather_test.astype(np.float32),
        'park_ids_train': park_ids[train_mask].astype(np.int64),
        'park_ids_test': park_ids[test_mask].astype(np.int64),
        'y_train_strikeouts': y_strikeouts[train_mask].astype(np.float32),
        'y_test_strikeouts': y_strikeouts[test_mask].astype(np.float32),
        'y_train_runs': y_runs[train_mask].astype(np.float32),
        'y_test_runs': y_runs[test_mask].astype(np.float32),
        'park_to_id': park_to_id,
        'id_to_park': {v: k for k, v in park_to_id.items()},
        'n_parks': len(unique_parks),
        'n_weather_features': X_weather.shape[1],
        'weather_scaler': scaler,
        'weather_feature_names': X_weather.columns.tolist(),
    }

    print(f"\nNN Data Prepared:")
    print(f"  Number of parks: {nn_data['n_parks']}")
    print(f"  Weather features: {nn_data['n_weather_features']}")
    print(f"  Training samples: {len(nn_data['X_weather_train'])}")
    print(f"  Testing samples: {len(nn_data['X_weather_test'])}")

    return nn_data


def get_park_game_counts(df: pd.DataFrame, test_start_season: int = 2023) -> pd.DataFrame:
    """
    Get the number of games per park in train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset
    test_start_season : int
        First season in test set

    Returns
    -------
    pd.DataFrame
        Game counts per park
    """
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = df.dropna(subset=required_cols)

    train_counts = data[data['season'] < test_start_season].groupby('home_team').size()
    test_counts = data[data['season'] >= test_start_season].groupby('home_team').size()

    counts = pd.DataFrame({
        'train_games': train_counts,
        'test_games': test_counts
    }).fillna(0).astype(int)

    counts['total_games'] = counts['train_games'] + counts['test_games']
    counts = counts.sort_values('total_games', ascending=False)

    return counts


def prepare_mixed_effects_data(
    df: pd.DataFrame,
    standardize_weather: bool = True,
    test_start_season: int = 2023,
    use_enhanced_features: bool = False,
    include_interactions: bool = False,
    scaler_type: str = 'robust'
) -> Dict[str, any]:
    """
    Prepare data specifically for mixed-effects regression models.

    This function prepares data with proper categorical grouping variables
    for mixed-effects models that control for team quality and season trends
    while estimating park-specific weather effects.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from load_all_team_data
    standardize_weather : bool
        Whether to standardize weather features (helps model convergence)
    test_start_season : int
        First season to include in test set
    use_enhanced_features : bool
        Whether to compute and include physics-based features (air_density, heat_index)
        and stadium characteristics (elevation, has_roof)
    include_interactions : bool
        Whether to include dome × weather and elevation × air_density interactions
    scaler_type : str
        Type of scaler to use: 'robust' (default, uses median/IQR - less sensitive
        to outliers) or 'standard' (uses mean/std)

    Returns
    -------
    Dict containing:
        - df_train: Training DataFrame ready for mixed-effects fitting
        - df_test: Test DataFrame for evaluation
        - weather_scaler: Fitted scaler (if standardize_weather=True)
        - group_info: Dict with unique levels for each grouping variable
        - weather_features: List of weather feature names used
        - scaler_type: Type of scaler used
    """
    # Determine which features to use
    if use_enhanced_features:
        # Add enhanced features first
        data = add_enhanced_weather_features(df)
        weather_features = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'heat_index']

        # Add stadium features if available
        if 'elevation_ft' in data.columns:
            weather_features.append('elevation_ft')
        if 'has_roof' in data.columns:
            weather_features.append('has_roof')
    else:
        data = df.copy()
        weather_features = WEATHER_FEATURES_BASIC.copy()

        # Always compute air_density for physics-based ball flight modeling
        # Air density directly affects how far the ball travels - lower density = farther flight
        if 'air_density' not in data.columns:
            pres = data['pres'].values if 'pres' in data.columns else None
            data['air_density'] = compute_air_density(
                data['temp_f'].values,
                data['rhum'].values,
                pres
            )

    # NOTE: wind_out_impact (wspd_mph × wind_cf) was previously included but removed
    # Reason: Not statistically significant (p > 0.65) and adds noise rather than signal
    # The interaction was intended to capture non-linear effects of strong winds blowing out,
    # but empirical testing showed it doesn't improve model performance.
    # If you want to re-enable it for experimental purposes, uncomment below:
    # if 'wspd_mph' in data.columns and 'wind_cf' in data.columns:
    #     data['wind_out_impact'] = data['wspd_mph'] * data['wind_cf']
    #     if 'wind_out_impact' not in weather_features:
    #         weather_features.append('wind_out_impact')

    # Drop rows with missing values in key columns
    required_cols = weather_features + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night',
                                        'home_team', 'away_team', 'season']
    # Only require columns that exist in the data
    required_cols = [c for c in required_cols if c in data.columns]
    data = data.dropna(subset=required_cols).copy()

    # Create binary is_night indicator
    data['is_night'] = (data['day_night'] == 'night').astype(int)

    # Create interaction terms if requested
    if include_interactions and use_enhanced_features:
        # Dome × weather interactions (zero out weather effect when dome is closed)
        # Note: We don't know actual roof status per game, so this creates an
        # interaction that the model can use to estimate differential effects
        if 'has_roof' in data.columns:
            for feat in ['temp_f', 'rhum', 'wspd_mph', 'air_density']:
                if feat in data.columns:
                    interaction_name = f'{feat}_x_roof'
                    data[interaction_name] = data[feat] * data['has_roof']
                    weather_features.append(interaction_name)

        # Elevation × air_density interaction (altitude amplifies density effects)
        if 'elevation_ft' in data.columns and 'air_density' in data.columns:
            # Normalize elevation to 0-1 range for better coefficient interpretation
            data['elevation_norm'] = data['elevation_ft'] / 5280  # Normalize to Coors = 1
            data['elevation_x_density'] = data['elevation_norm'] * data['air_density']
            weather_features.extend(['elevation_norm', 'elevation_x_density'])

    # Ensure grouping variables are regular Python string objects (not pandas StringDtype)
    # This is important for statsmodels compatibility
    # Use pd.Series with dtype='object' to force object dtype
    data['home_team'] = pd.Series([str(x) for x in data['home_team'].values], dtype='object', index=data.index)
    data['away_team'] = pd.Series([str(x) for x in data['away_team'].values], dtype='object', index=data.index)
    data['season'] = pd.Series([str(x) for x in data['season'].values], dtype='object', index=data.index)

    # Train/test split by season
    train_mask = data['season'].astype(int) < test_start_season
    df_train = data[train_mask].copy().reset_index(drop=True)
    df_test = data[~train_mask].copy().reset_index(drop=True)

    # Standardize weather features if requested
    weather_scaler = None
    # Only standardize continuous features (not binary like has_roof)
    features_to_scale = [f for f in weather_features if f not in ['has_roof']]

    if standardize_weather:
        # Use RobustScaler by default (less sensitive to outliers ~11.6% in wind features)
        # RobustScaler uses median and IQR instead of mean and std
        if scaler_type == 'robust':
            weather_scaler = RobustScaler()
        elif scaler_type == 'standard':
            weather_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}. Use 'robust' or 'standard'.")

        df_train[features_to_scale] = weather_scaler.fit_transform(df_train[features_to_scale])
        df_test[features_to_scale] = weather_scaler.transform(df_test[features_to_scale])

    # Gather group information
    group_info = {
        'home_teams': sorted(data['home_team'].unique().tolist()),
        'away_teams': sorted(data['away_team'].unique().tolist()),
        'seasons': sorted(data['season'].unique().tolist()),
        'n_home_teams': data['home_team'].nunique(),
        'n_away_teams': data['away_team'].nunique(),
        'n_seasons': data['season'].nunique(),
    }

    # Game counts per group
    group_info['games_per_home_team'] = df_train.groupby('home_team').size().to_dict()
    group_info['games_per_away_team'] = df_train.groupby('away_team').size().to_dict()
    group_info['games_per_season'] = df_train.groupby('season').size().to_dict()

    print(f"\nMixed-Effects Data Prepared:")
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    print(f"  Home teams (parks): {group_info['n_home_teams']}")
    print(f"  Away teams: {group_info['n_away_teams']}")
    print(f"  Seasons: {group_info['n_seasons']}")
    print(f"  Weather standardized: {standardize_weather} (scaler: {scaler_type if standardize_weather else 'N/A'})")
    print(f"  Enhanced features: {use_enhanced_features}")
    print(f"  Weather features used: {weather_features}")

    return {
        'df_train': df_train,
        'df_test': df_test,
        'weather_scaler': weather_scaler,
        'scaler_type': scaler_type if standardize_weather else None,
        'group_info': group_info,
        'weather_features': weather_features,
        'features_scaled': features_to_scale,
        'target_strikeouts': TARGET_STRIKEOUTS,
        'target_runs': TARGET_RUNS,
        'use_enhanced_features': use_enhanced_features,
    }


# ============================================================================
# DEVIATION-BASED TARGET FUNCTIONS
# ============================================================================

def compute_team_season_averages(
    df: pd.DataFrame,
    method: str = 'expanding',
    min_games: int = 10
) -> pd.DataFrame:
    """
    Compute away team season averages for strikeouts and runs.

    These averages represent a team's baseline performance as an away team,
    which can be used to compute deviation targets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with away_team, season, away_bat_k, away_runs_scored columns
    method : str
        'expanding' (RECOMMENDED) - use only games before current date
            This prevents temporal leakage by only using past information.
        'loo' (leave-one-out) - compute average excluding current game
            WARNING: This causes temporal leakage as it uses future games
            in computing the average. Only use for ablation studies.
    min_games : int
        Minimum games required before computing meaningful average.
        Games before this threshold use the team's partial-season average.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - away_team_avg_k: Team's average strikeouts as away team
        - away_team_avg_runs: Team's average runs as away team
        - away_team_games: Number of games used for average

    Notes
    -----
    The 'expanding' method is recommended because it:
    1. Prevents temporal leakage (no future information used)
    2. More realistically models what would be known at prediction time
    3. Provides more conservative but honest performance estimates
    """
    data = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(data['game_date']):
        data['game_date'] = pd.to_datetime(data['game_date'])

    # Sort by date for expanding window
    data = data.sort_values(['away_team', 'season', 'game_date']).reset_index(drop=True)

    # Initialize output columns
    data['away_team_avg_k'] = np.nan
    data['away_team_avg_runs'] = np.nan
    data['away_team_games'] = 0

    if method == 'loo':
        # Leave-one-out: compute average excluding current game
        for (team, season), group in data.groupby(['away_team', 'season']):
            indices = group.index.tolist()
            n_games = len(indices)

            if n_games < 2:
                continue

            # Total sums for the group
            total_k = group['away_bat_k'].sum()
            total_runs = group['away_runs_scored'].sum()

            for idx in indices:
                # Exclude current game
                other_k = total_k - data.loc[idx, 'away_bat_k']
                other_runs = total_runs - data.loc[idx, 'away_runs_scored']
                other_games = n_games - 1

                if other_games >= min_games:
                    data.loc[idx, 'away_team_avg_k'] = other_k / other_games
                    data.loc[idx, 'away_team_avg_runs'] = other_runs / other_games
                    data.loc[idx, 'away_team_games'] = other_games
                elif other_games > 0:
                    # Use partial average but flag with lower game count
                    data.loc[idx, 'away_team_avg_k'] = other_k / other_games
                    data.loc[idx, 'away_team_avg_runs'] = other_runs / other_games
                    data.loc[idx, 'away_team_games'] = other_games

    elif method == 'expanding':
        # Expanding window: only use games before current date
        for (team, season), group in data.groupby(['away_team', 'season']):
            indices = group.index.tolist()
            dates = group['game_date'].tolist()
            k_values = group['away_bat_k'].tolist()
            runs_values = group['away_runs_scored'].tolist()

            cumsum_k = 0
            cumsum_runs = 0
            n_prior = 0

            for i, idx in enumerate(indices):
                if n_prior >= 1:
                    avg_k = cumsum_k / n_prior
                    avg_runs = cumsum_runs / n_prior
                    data.loc[idx, 'away_team_avg_k'] = avg_k
                    data.loc[idx, 'away_team_avg_runs'] = avg_runs
                    data.loc[idx, 'away_team_games'] = n_prior

                # Add current game to cumulative sums for next iteration
                if pd.notna(k_values[i]):
                    cumsum_k += k_values[i]
                if pd.notna(runs_values[i]):
                    cumsum_runs += runs_values[i]
                n_prior += 1
    else:
        raise ValueError(f"Unknown method: {method}. Use 'loo' or 'expanding'.")

    return data[['away_team_avg_k', 'away_team_avg_runs', 'away_team_games']]


def compute_park_adjustment_factors(
    league_avgs_path: str = None
) -> pd.DataFrame:
    """
    Compute park adjustment factors from league batting averages.

    Factor = park_avg / league_avg
    Example: COL strikeouts = 15.58, league = 17.01 → factor = 0.916

    Parameters
    ----------
    league_avgs_path : str, optional
        Path to league_batting_avgs_2021_2025.csv.
        If None, searches in default locations.

    Returns
    -------
    pd.DataFrame
        Park factors indexed by home_team with columns:
        - k_factor: Strikeout park factor
        - runs_factor: Runs park factor
    """
    if league_avgs_path is None:
        # Try common locations
        possible_paths = [
            Path(__file__).parent.parent / 'data' / 'league_batting_avgs_2021_2025.csv',
            Path('data/league_batting_avgs_2021_2025.csv'),
            Path('../data/league_batting_avgs_2021_2025.csv'),
        ]
        for p in possible_paths:
            if p.exists():
                league_avgs_path = p
                break
        if league_avgs_path is None:
            raise FileNotFoundError("Could not find league_batting_avgs_2021_2025.csv")

    league_df = pd.read_csv(league_avgs_path)

    # Extract league average row
    league_avg_row = league_df[league_df['home_team'] == 'LEAGUE AVG']
    if league_avg_row.empty:
        raise ValueError("LEAGUE AVG row not found in league_batting_avgs_2021_2025.csv")

    league_avg_k = league_avg_row['strikeouts'].values[0]
    league_avg_runs = league_avg_row['total_runs'].values[0]

    # Filter out league average row
    park_df = league_df[league_df['home_team'] != 'LEAGUE AVG'].copy()

    # Compute park factors
    park_df['k_factor'] = park_df['strikeouts'] / league_avg_k
    park_df['runs_factor'] = park_df['total_runs'] / league_avg_runs

    # Set index to home_team
    result = park_df.set_index('home_team')[['k_factor', 'runs_factor']].copy()

    # Store league averages as attributes (for reference)
    result.attrs['league_avg_k'] = league_avg_k
    result.attrs['league_avg_runs'] = league_avg_runs

    return result


def compute_deviation_targets(
    df: pd.DataFrame,
    method: str = 'loo',
    min_games: int = 10,
    league_avgs_path: str = None
) -> pd.DataFrame:
    """
    Compute deviation targets: actual - expected.

    Expected = team_season_avg * park_factor

    This creates targets that represent how much a team over/under-performs
    relative to their expected baseline at each park.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required columns for team averages and park identification
    method : str
        Method for computing team averages ('loo' or 'expanding')
    min_games : int
        Minimum games required for team averages
    league_avgs_path : str, optional
        Path to league batting averages CSV

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - away_team_avg_k, away_team_avg_runs, away_team_games
        - k_factor, runs_factor (park factors)
        - expected_away_k, expected_away_runs
        - deviation_away_k, deviation_away_runs
    """
    data = df.copy()

    # Step 1: Compute team season averages
    team_avgs = compute_team_season_averages(data, method=method, min_games=min_games)
    data = pd.concat([data, team_avgs], axis=1)

    # Step 2: Compute park factors
    park_factors = compute_park_adjustment_factors(league_avgs_path)

    # Map park factors to each game based on home_team
    # Handle potential team code mismatches (OAK vs ATH, ARI vs AZ)
    team_mapping = {
        'OAK': 'ATH',  # Oakland Athletics historical code
        'ARI': 'AZ',   # Arizona Diamondbacks
    }

    # Create lookup column
    lookup_team = data['home_team'].replace(team_mapping)

    # Map factors
    data['k_factor'] = lookup_team.map(park_factors['k_factor'])
    data['runs_factor'] = lookup_team.map(park_factors['runs_factor'])

    # Handle missing park factors (use 1.0 as neutral)
    data['k_factor'] = data['k_factor'].fillna(1.0)
    data['runs_factor'] = data['runs_factor'].fillna(1.0)

    # Step 3: Compute expected values
    # Expected = team_avg * park_factor
    data['expected_away_k'] = data['away_team_avg_k'] * data['k_factor']
    data['expected_away_runs'] = data['away_team_avg_runs'] * data['runs_factor']

    # Step 4: Compute deviations
    # Deviation = actual - expected
    data['deviation_away_k'] = data['away_bat_k'] - data['expected_away_k']
    data['deviation_away_runs'] = data['away_runs_scored'] - data['expected_away_runs']

    return data


def prepare_deviation_data(
    df: pd.DataFrame,
    standardize_weather: bool = True,
    test_start_season: int = 2023,
    use_enhanced_features: bool = False,
    include_interactions: bool = False,
    deviation_method: str = 'expanding',
    min_games: int = 10
) -> Dict[str, any]:
    """
    Prepare data for deviation-based mixed-effects models.

    This is an extension of prepare_mixed_effects_data() that also computes
    deviation targets for comparison with raw targets.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from load_all_team_data
    standardize_weather : bool
        Whether to standardize weather features
    test_start_season : int
        First season to include in test set
    use_enhanced_features : bool
        Whether to compute physics-based features
    include_interactions : bool
        Whether to include dome × weather and elevation × air_density interactions
    deviation_method : str
        Method for computing team averages ('loo' or 'expanding')
    min_games : int
        Minimum games required for meaningful team averages

    Returns
    -------
    Dict containing all fields from prepare_mixed_effects_data() plus:
        - deviation_columns: List of deviation target column names
        - expected_columns: List of expected value column names
        - deviation_method: Method used for computing deviations
        - n_valid_deviations: Number of games with valid deviation targets
    """
    # First compute deviation targets on the raw data
    data_with_deviations = compute_deviation_targets(
        df,
        method=deviation_method,
        min_games=min_games
    )

    # Now prepare for mixed effects (this handles enhanced features, standardization, etc.)
    me_data = prepare_mixed_effects_data(
        data_with_deviations,
        standardize_weather=standardize_weather,
        test_start_season=test_start_season,
        use_enhanced_features=use_enhanced_features,
        include_interactions=include_interactions
    )

    # Add deviation-specific information
    deviation_cols = ['deviation_away_k', 'deviation_away_runs']
    expected_cols = ['expected_away_k', 'expected_away_runs']
    team_avg_cols = ['away_team_avg_k', 'away_team_avg_runs', 'away_team_games']
    park_factor_cols = ['k_factor', 'runs_factor']

    me_data['deviation_columns'] = deviation_cols
    me_data['expected_columns'] = expected_cols
    me_data['team_avg_columns'] = team_avg_cols
    me_data['park_factor_columns'] = park_factor_cols
    me_data['deviation_method'] = deviation_method
    me_data['min_games'] = min_games

    # Count valid deviations (where team averages are available)
    n_valid_train = me_data['df_train']['deviation_away_k'].notna().sum()
    n_valid_test = me_data['df_test']['deviation_away_k'].notna().sum()
    me_data['n_valid_deviations'] = {'train': n_valid_train, 'test': n_valid_test}

    # Add deviation targets to me_data for convenience
    me_data['target_deviation_k'] = 'deviation_away_k'
    me_data['target_deviation_runs'] = 'deviation_away_runs'

    print(f"\nDeviation Data Summary:")
    print(f"  Method: {deviation_method}")
    print(f"  Min games for average: {min_games}")
    print(f"  Valid deviations (train): {n_valid_train} / {len(me_data['df_train'])}")
    print(f"  Valid deviations (test): {n_valid_test} / {len(me_data['df_test'])}")

    # Show deviation statistics
    if n_valid_train > 0:
        dev_k = me_data['df_train']['deviation_away_k'].dropna()
        dev_runs = me_data['df_train']['deviation_away_runs'].dropna()
        print(f"\n  Deviation Stats (train):")
        print(f"    K deviation: mean={dev_k.mean():.3f}, std={dev_k.std():.3f}")
        print(f"    Runs deviation: mean={dev_runs.mean():.3f}, std={dev_runs.std():.3f}")

    return me_data


# ============================================================================
# PARK-WEATHER INTERACTION FUNCTIONS
# ============================================================================

def prepare_park_weather_interactions(
    df: pd.DataFrame,
    weather_features: List[str] = ['temp_f', 'wspd_mph', 'wind_cf'],
    standardize: bool = True,
    test_start_season: int = 2023,
    scaler_type: str = 'robust',
    split_method: str = 'season',
    test_size: float = 0.30,
    random_state: int = 42,
    include_roof_interactions: bool = False
) -> Dict[str, any]:
    """
    Prepare data for Ridge/Lasso regression with explicit park × weather interactions.

    This function creates a feature matrix with:
    - Main weather effects (3 features): temp_f, wspd_mph, wind_cf
    - Park indicator dummies (30 features): park_ARI, ..., park_WSH
    - Park × weather interactions (90 features): temp_f_x_ARI, ..., wind_cf_x_WSH
    - Day/night indicator (1 feature): is_night

    Total: ~124 columns (may vary based on parks present in data)

    The interaction terms allow each park to have its own weather sensitivity,
    enabling insights like:
    - Oracle Park has high wind_cf sensitivity (marine layer)
    - Coors Field has high temp_f sensitivity (altitude)
    - Domed stadiums have near-zero weather sensitivity

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset from load_all_team_data with weather columns and home_team
    weather_features : List[str]
        Weather features to include in interactions.
        Default: ['temp_f', 'wspd_mph', 'wind_cf'] (3 features × 30 parks = 90 interactions)
    standardize : bool
        Whether to standardize weather features before creating interactions.
        Recommended: True for Ridge/Lasso regularization to work properly.
    test_start_season : int
        First season to include in test set (default: 2023). Only used if split_method='season'.
    scaler_type : str
        Type of scaler: 'robust' (median/IQR) or 'standard' (mean/std)
    split_method : str
        Method for train/test split:
        - 'season': Split by season (train < test_start_season, test >= test_start_season)
        - 'random_month': Randomly assign year-month units to train/test
    test_size : float
        Proportion of data for test set (default: 0.30). Only used if split_method='random_month'.
    random_state : int
        Random seed for reproducibility (default: 42). Only used if split_method='random_month'.
    include_roof_interactions : bool
        Whether to include has_roof × weather interactions. When True, adds:
        - has_roof: Binary indicator (1 if stadium has retractable/fixed roof)
        - has_roof × temp_f, has_roof × wspd_mph, has_roof × wind_cf
        This allows the model to learn that roofed stadiums have dampened weather effects.

    Returns
    -------
    Dict containing:
        - df_train: Training DataFrame with all features
        - df_test: Test DataFrame with all features
        - X_train: Feature matrix (numpy array) for training
        - X_test: Feature matrix (numpy array) for testing
        - y_train_strikeouts: Training strikeout targets
        - y_test_strikeouts: Test strikeout targets
        - y_train_runs: Training runs targets
        - y_test_runs: Test runs targets
        - feature_names: List of all feature column names
        - weather_features: Weather features used
        - park_dummies: List of park dummy column names
        - interaction_features: List of interaction column names
        - weather_scaler: Fitted scaler object (if standardize=True)
        - scaling_params: Dict with center/scale for each weather feature
        - parks: List of unique park codes

    Example
    -------
    >>> from data_prep import load_all_team_data, prepare_park_weather_interactions
    >>> df = load_all_team_data('data')
    >>> data = prepare_park_weather_interactions(df)
    >>> print(f"Features: {len(data['feature_names'])}")
    >>> print(f"Interactions: {len(data['interaction_features'])}")
    """
    # Make a copy to avoid modifying original
    data = df.copy()

    # Ensure required columns exist
    required_cols = weather_features + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team', 'season']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing values in key columns
    data = data.dropna(subset=required_cols).copy()

    # Convert grouping variables to object dtype for consistency
    data['home_team'] = data['home_team'].astype(str).astype('object')
    data['season'] = data['season'].astype(int)

    # Create is_night indicator
    data['is_night'] = (data['day_night'] == 'night').astype(int)

    # Train/test split based on method
    if split_method == 'season':
        # Original method: split by season
        train_mask = data['season'] < test_start_season
        test_mask = ~train_mask
        split_info = {
            'method': 'season',
            'test_start_season': test_start_season,
            'train_seasons': sorted(data.loc[train_mask, 'season'].unique()),
            'test_seasons': sorted(data.loc[test_mask, 'season'].unique()),
        }
        print(f"\nSeason-based split: train < {test_start_season}, test >= {test_start_season}")
    elif split_method == 'random_month':
        # New method: random assignment of year-month units
        train_mask, test_mask, train_months, test_months = create_random_month_split(
            data, test_size=test_size, random_state=random_state
        )
        split_info = {
            'method': 'random_month',
            'test_size': test_size,
            'random_state': random_state,
            'train_months': train_months,
            'test_months': test_months,
        }
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'season' or 'random_month'.")

    df_train = data[train_mask].copy().reset_index(drop=True)
    df_test = data[test_mask].copy().reset_index(drop=True)

    # Get unique parks
    parks = sorted(df_train['home_team'].unique())
    print(f"Found {len(parks)} unique parks in training data")

    # Add has_roof indicator if requested
    roof_features = []
    roof_interaction_features = []
    if include_roof_interactions:
        # Load stadium parameters to get has_roof
        try:
            stadium_params = load_stadium_parameters()
            # Map has_roof to each game based on home_team
            # Handle team code differences (AZ vs ARI, ATH vs OAK)
            team_mapping = {'ARI': 'AZ', 'OAK': 'ATH'}

            for df_split in [df_train, df_test]:
                lookup_team = df_split['home_team'].replace(team_mapping)
                df_split['has_roof'] = lookup_team.map(stadium_params['has_roof']).fillna(0).astype(int)

            roof_features = ['has_roof']

            # Create has_roof × weather interactions (will be added after scaling)
            print(f"Added has_roof indicator. Parks with roofs: {[p for p in parks if p in ['TB', 'MIA', 'HOU', 'ARI', 'TOR', 'MIL', 'SEA', 'TEX']]}")
        except FileNotFoundError:
            warnings.warn("Could not load stadium parameters. has_roof feature not added.")
            include_roof_interactions = False

    # Standardize weather features if requested
    weather_scaler = None
    scaling_params = {}

    if standardize:
        if scaler_type == 'robust':
            weather_scaler = RobustScaler()
        elif scaler_type == 'standard':
            weather_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        # Fit on training data only
        df_train[weather_features] = weather_scaler.fit_transform(df_train[weather_features])
        df_test[weather_features] = weather_scaler.transform(df_test[weather_features])

        # Store scaling parameters
        if scaler_type == 'robust':
            for i, feat in enumerate(weather_features):
                scaling_params[feat] = {
                    'center': weather_scaler.center_[i],
                    'scale': weather_scaler.scale_[i]
                }
        else:
            for i, feat in enumerate(weather_features):
                scaling_params[feat] = {
                    'center': weather_scaler.mean_[i],
                    'scale': weather_scaler.scale_[i]
                }

    # Create park dummies (one-hot encoding, keeping all categories for interaction terms)
    # Use pd.concat to avoid DataFrame fragmentation warnings
    park_dummies = [f'park_{p}' for p in parks]
    train_park_dummies = {f'park_{park}': (df_train['home_team'] == park).astype(int) for park in parks}
    test_park_dummies = {f'park_{park}': (df_test['home_team'] == park).astype(int) for park in parks}

    # Create interaction terms: weather_feature × park_dummy
    interaction_features = []
    train_interactions = {}
    test_interactions = {}

    for weather_feat in weather_features:
        for park in parks:
            interaction_name = f'{weather_feat}_x_{park}'
            interaction_features.append(interaction_name)
            train_interactions[interaction_name] = df_train[weather_feat].values * train_park_dummies[f'park_{park}'].values
            test_interactions[interaction_name] = df_test[weather_feat].values * test_park_dummies[f'park_{park}'].values

    # Create has_roof × weather interactions if requested
    # This allows the model to learn that roofed stadiums have dampened weather effects
    if include_roof_interactions and 'has_roof' in df_train.columns:
        for weather_feat in weather_features:
            interaction_name = f'has_roof_x_{weather_feat}'
            roof_interaction_features.append(interaction_name)
            train_interactions[interaction_name] = df_train['has_roof'].values * df_train[weather_feat].values
            test_interactions[interaction_name] = df_test['has_roof'].values * df_test[weather_feat].values

    # Concatenate all new columns at once to avoid fragmentation
    df_train = pd.concat([df_train, pd.DataFrame(train_park_dummies, index=df_train.index), pd.DataFrame(train_interactions, index=df_train.index)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(test_park_dummies, index=df_test.index), pd.DataFrame(test_interactions, index=df_test.index)], axis=1)

    # Define feature columns in order: weather, is_night, has_roof (if included), park dummies, interactions, roof interactions
    feature_names = weather_features + ['is_night'] + roof_features + park_dummies + interaction_features + roof_interaction_features

    # Create feature matrices
    X_train = df_train[feature_names].values
    X_test = df_test[feature_names].values

    # Extract targets
    y_train_strikeouts = df_train[TARGET_STRIKEOUTS].values
    y_test_strikeouts = df_test[TARGET_STRIKEOUTS].values
    y_train_runs = df_train[TARGET_RUNS].values
    y_test_runs = df_test[TARGET_RUNS].values

    print(f"\nPark-Weather Interaction Data Prepared:")
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    print(f"  Weather features: {len(weather_features)} ({weather_features})")
    print(f"  Park dummies: {len(park_dummies)}")
    print(f"  Park-weather interactions: {len(interaction_features)}")
    if roof_interaction_features:
        print(f"  Roof-weather interactions: {len(roof_interaction_features)} ({roof_interaction_features})")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Standardized: {standardize} (scaler: {scaler_type if standardize else 'N/A'})")

    return {
        'df_train': df_train,
        'df_test': df_test,
        'X_train': X_train,
        'X_test': X_test,
        'y_train_strikeouts': y_train_strikeouts,
        'y_test_strikeouts': y_test_strikeouts,
        'y_train_runs': y_train_runs,
        'y_test_runs': y_test_runs,
        'feature_names': feature_names,
        'weather_features': weather_features,
        'park_dummies': park_dummies,
        'interaction_features': interaction_features,
        'roof_features': roof_features,
        'roof_interaction_features': roof_interaction_features,
        'weather_scaler': weather_scaler,
        'scaling_params': scaling_params,
        'scaler_type': scaler_type if standardize else None,
        'parks': parks,
        'target_strikeouts': TARGET_STRIKEOUTS,
        'target_runs': TARGET_RUNS,
        'split_info': split_info,
    }


if __name__ == '__main__':
    # Test the data loading
    import os

    # Determine the data directory path
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\n" + "="*50)
    print("Preparing features with interactions...")
    X, y_k, y_runs = prepare_features(df, include_interactions=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Weather features: {WEATHER_FEATURES}")
    print(f"Sample features: {X.columns[:10].tolist()} ...")

    print("\n" + "="*50)
    print("Creating train/test split...")
    splits = train_test_split_by_season(df, X, y_k, y_runs)

    print("\n" + "="*50)
    print("Preparing NN data...")
    nn_data = prepare_nn_data(df)

    print("\n" + "="*50)
    print("Preparing mixed-effects data (basic features)...")
    me_data = prepare_mixed_effects_data(df, use_enhanced_features=False)

    print("\n" + "="*50)
    print("Preparing mixed-effects data (ENHANCED features)...")
    me_data_enhanced = prepare_mixed_effects_data(
        df,
        use_enhanced_features=True,
        include_interactions=True
    )

    # Show sample of enhanced features
    print("\nEnhanced feature sample:")
    enhanced_cols = ['home_team', 'temp_f', 'rhum', 'air_density', 'heat_index',
                     'elevation_ft', 'has_roof']
    available_cols = [c for c in enhanced_cols if c in me_data_enhanced['df_train'].columns]
    print(me_data_enhanced['df_train'][available_cols].head(10))

    # Show stadium parameter summary
    print("\n" + "="*50)
    print("Stadium parameters loaded:")
    try:
        params = load_stadium_parameters()
        print(params[['elevation_ft', 'has_roof']].sort_values('elevation_ft', ascending=False))
    except Exception as e:
        print(f"Could not load stadium parameters: {e}")

    print("\n" + "="*50)
    print("Park game counts:")
    counts = get_park_game_counts(df)
    print(counts)

    # Test deviation-based target functions
    print("\n" + "="*50)
    print("Testing DEVIATION-BASED TARGET FUNCTIONS...")

    print("\n[1] Computing park adjustment factors...")
    try:
        park_factors = compute_park_adjustment_factors()
        print(f"  Found {len(park_factors)} parks")
        print(f"  League avg K: {park_factors.attrs.get('league_avg_k', 'N/A')}")
        print(f"  League avg Runs: {park_factors.attrs.get('league_avg_runs', 'N/A')}")
        print("\n  Park factors (sample):")
        print(park_factors.head(10))
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[2] Computing team season averages (LOO method)...")
    try:
        team_avgs = compute_team_season_averages(df, method='loo', min_games=10)
        valid_count = team_avgs['away_team_avg_k'].notna().sum()
        print(f"  Valid team averages: {valid_count} / {len(team_avgs)}")
        print(f"  Avg K (mean): {team_avgs['away_team_avg_k'].mean():.2f}")
        print(f"  Avg Runs (mean): {team_avgs['away_team_avg_runs'].mean():.2f}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[3] Computing deviation targets...")
    try:
        df_with_dev = compute_deviation_targets(df, method='loo', min_games=10)
        valid_dev = df_with_dev['deviation_away_k'].notna().sum()
        print(f"  Valid deviations: {valid_dev} / {len(df_with_dev)}")
        print(f"  Deviation K stats:")
        print(f"    Mean: {df_with_dev['deviation_away_k'].mean():.3f} (should be ~0)")
        print(f"    Std: {df_with_dev['deviation_away_k'].std():.3f}")
        print(f"  Deviation Runs stats:")
        print(f"    Mean: {df_with_dev['deviation_away_runs'].mean():.3f} (should be ~0)")
        print(f"    Std: {df_with_dev['deviation_away_runs'].std():.3f}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[4] Preparing deviation data for mixed-effects models...")
    try:
        dev_data = prepare_deviation_data(df, use_enhanced_features=False)
        print(f"  Deviation columns: {dev_data['deviation_columns']}")
        print(f"  Expected columns: {dev_data['expected_columns']}")
    except Exception as e:
        print(f"  Error: {e}")
