"""
Data preparation module for away team regression models.

Loads team CSV files, combines them, and prepares features for modeling.
Includes physics-based weather features (air density, heat index) and
stadium characteristics (elevation, retractable roof status).

Weather Features:
- temp_f: Air temperature (°F)
- rhum: Relative humidity (0-100%)
- wspd_mph: Wind speed (mph)
- wind_cf: Wind component toward center field (-1 to +1, positive = blowing out)
- air_density: Computed from temp/humidity/pressure (kg/m³) - affects ball flight
- heat_index: Apparent temperature for player fatigue
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
import warnings


def compute_air_density(temp_f: np.ndarray, rhum: np.ndarray, pres: np.ndarray = None) -> np.ndarray:
    """Compute air density from temperature, humidity, and pressure (kg/m³)."""
    temp_k = (temp_f - 32) * 5/9 + 273.15
    if pres is None:
        pres = np.full_like(temp_f, 1013.25)

    temp_c = (temp_f - 32) * 5/9
    e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
    e = (rhum / 100) * e_sat
    p_d = (pres - e) * 100

    R_d, R_v = 287.05, 461.5
    return p_d / (R_d * temp_k) + (e * 100) / (R_v * temp_k)


def compute_heat_index(temp_f: np.ndarray, rhum: np.ndarray) -> np.ndarray:
    """Compute heat index (apparent temperature) using Rothfusz regression."""
    T = np.asarray(temp_f, dtype=float)
    R = np.asarray(rhum, dtype=float)

    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))
    mask = ((HI + T) / 2) >= 80

    if np.any(mask):
        T_m, R_m = T[mask], R[mask]
        HI_full = (-42.379 + 2.04901523 * T_m + 10.14333127 * R_m - 0.22475541 * T_m * R_m
                   - 0.00683783 * T_m**2 - 0.05481717 * R_m**2 + 0.00122874 * T_m**2 * R_m
                   + 0.00085282 * T_m * R_m**2 - 0.00000199 * T_m**2 * R_m**2)

        adj_mask = (R_m < 13) & (T_m > 80) & (T_m < 112)
        if np.any(adj_mask):
            HI_full[adj_mask] -= ((13 - R_m[adj_mask]) / 4) * np.sqrt((17 - np.abs(T_m[adj_mask] - 95)) / 17)

        adj_mask2 = (R_m > 85) & (T_m > 80) & (T_m < 87)
        if np.any(adj_mask2):
            HI_full[adj_mask2] += ((R_m[adj_mask2] - 85) / 10) * ((87 - T_m[adj_mask2]) / 5)

        HI[mask] = HI_full
    return HI


def compute_altitude_adjusted_density(air_density: np.ndarray, elevation_ft: np.ndarray) -> np.ndarray:
    """Compute density ratio (actual/expected based on elevation)."""
    expected_density = 1.225 * np.exp(-elevation_ft / 29000)
    return air_density / expected_density


TEAM_FILE_TO_ABBREV = {
    'angels': 'LAA', 'astros': 'HOU', 'athletics': 'OAK', 'blue_jays': 'TOR',
    'braves': 'ATL', 'brewers': 'MIL', 'cardinals': 'STL', 'cubs': 'CHC',
    'dbacks': 'ARI', 'dodgers': 'LAD', 'fenway': 'BOS', 'giants': 'SF',
    'guardians': 'CLE', 'mariners': 'SEA', 'marlins': 'MIA', 'mets': 'NYM',
    'nationals': 'WSH', 'orioles': 'BAL', 'padres': 'SD', 'phillies': 'PHI',
    'pirates': 'PIT', 'rangers': 'TEX', 'rays': 'TB', 'reds': 'CIN',
    'rockies': 'COL', 'royals': 'KC', 'tigers': 'DET', 'twins': 'MIN',
    'white_sox': 'CWS', 'yankee': 'NYY',
}

# Team name (from elevation.csv) to team code mapping
TEAM_NAME_TO_CODE = {
    'Rockies': 'COL', 'Athletics': 'OAK', 'Tigers': 'DET', 'Dodgers': 'LAD',
    'Blue Jays': 'TOR', 'Red Sox': 'BOS', 'Orioles': 'BAL', 'D-backs': 'ARI',
    'Phillies': 'PHI', 'Rays': 'TB', 'Nationals': 'WSH', 'Twins': 'MIN',
    'Braves': 'ATL', 'Angels': 'LAA', 'Reds': 'CIN', 'Yankees': 'NYY',
    'Giants': 'SF', 'Mets': 'NYM', 'Cubs': 'CHC', 'White Sox': 'CWS',
    'Brewers': 'MIL', 'Marlins': 'MIA', 'Astros': 'HOU', 'Royals': 'KC',
    'Cardinals': 'STL', 'Pirates': 'PIT', 'Guardians': 'CLE', 'Padres': 'SD',
    'Rangers': 'TEX', 'Mariners': 'SEA',
}

# Pacific coast parks with marine layer influence (fog/cool moist air)
MARINE_LAYER_PARKS = ['SF', 'OAK', 'SD', 'LAA', 'LAD', 'SEA']

WEATHER_FEATURES_BASIC = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']
WEATHER_FEATURES_ENHANCED = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'heat_index']
WEATHER_FEATURES_FULL = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'heat_index', 'elevation_ft', 'has_roof']
WEATHER_FEATURES = WEATHER_FEATURES_BASIC
TARGET_STRIKEOUTS = 'away_bat_k'
TARGET_RUNS = 'away_runs_scored'


def load_stadium_parameters(params_path: str = None) -> pd.DataFrame:
    """Load stadium parameters (elevation, roof status) from CSV."""
    if params_path is None:
        for p in [Path(__file__).parent.parent / 'analysis' / 'team_parameters.csv',
                  Path('analysis/team_parameters.csv'), Path('../analysis/team_parameters.csv')]:
            if p.exists():
                params_path = p
                break
        if params_path is None:
            raise FileNotFoundError("Could not find team_parameters.csv")

    params_df = pd.read_csv(params_path)
    missing = [c for c in ['team_code', 'elevation_ft', 'has_roof'] if c not in params_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return params_df.set_index('team_code')


def load_elevation_data(elevation_path: str = None) -> Dict[str, float]:
    """Load elevation data from elevation.csv and return dict mapping team_code -> elevation_ft.

    Uses data/elevation.csv as the authoritative source for stadium elevations.
    """
    if elevation_path is None:
        for p in [Path(__file__).parent.parent / 'data' / 'elevation.csv',
                  Path('data/elevation.csv'), Path('../data/elevation.csv')]:
            if p.exists():
                elevation_path = p
                break
        if elevation_path is None:
            raise FileNotFoundError("Could not find elevation.csv")

    df = pd.read_csv(elevation_path)
    elevation_map = {}
    for _, row in df.iterrows():
        team_name = row['Team']
        if team_name in TEAM_NAME_TO_CODE:
            elevation_map[TEAM_NAME_TO_CODE[team_name]] = row['Elevation (Feet)']

    # Store league average for reference
    if 'League' in df['Team'].values:
        league_row = df[df['Team'] == 'League']
        elevation_map['_league_avg'] = league_row['Elevation (Feet)'].values[0]

    return elevation_map


def add_enhanced_weather_features(df: pd.DataFrame, stadium_params: pd.DataFrame = None) -> pd.DataFrame:
    """Add physics-based weather features (air_density, heat_index, elevation, roof)."""
    data = df.copy()

    for col in ['home_team', 'away_team', 'season']:
        if col in data.columns:
            data[col] = data[col].astype(str).astype('object')

    if 'temp_f' not in data.columns or 'rhum' not in data.columns:
        raise ValueError("Dataset must contain 'temp_f' and 'rhum' columns")

    pres = data['pres'].values if 'pres' in data.columns else None
    data['air_density'] = compute_air_density(data['temp_f'].values, data['rhum'].values, pres)
    data['heat_index'] = compute_heat_index(data['temp_f'].values, data['rhum'].values)

    if stadium_params is None:
        try:
            stadium_params = load_stadium_parameters()
        except FileNotFoundError:
            warnings.warn("Could not load stadium parameters.")
            return data

    if 'home_team' in data.columns:
        lookup_team = data['home_team'].replace({'OAK': 'ATH', 'ARI': 'AZ'})
        if 'elevation_ft' in stadium_params.columns:
            data['elevation_ft'] = lookup_team.map(stadium_params['elevation_ft'])
        if 'has_roof' in stadium_params.columns:
            data['has_roof'] = lookup_team.map(stadium_params['has_roof']).fillna(0).astype(int)
        if 'elevation_ft' in data.columns and not data['elevation_ft'].isna().all():
            data['density_ratio'] = compute_altitude_adjusted_density(
                data['air_density'].values, data['elevation_ft'].fillna(0).values)
    return data


def load_all_team_data(data_dir: str = 'data') -> pd.DataFrame:
    """Load all team CSV files and combine into a single DataFrame."""
    data_path = Path(data_dir)
    all_dfs = []

    for csv_file in data_path.glob('*_data_*_day_night.csv'):
        team_prefix = csv_file.stem.split('_data_')[0]
        if team_prefix not in TEAM_FILE_TO_ABBREV:
            warnings.warn(f"Unknown team prefix: {team_prefix}")
            continue
        df = pd.read_csv(csv_file)
        df['home_team'] = TEAM_FILE_TO_ABBREV[team_prefix]
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No team CSV files found in {data_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['game_date'] = pd.to_datetime(combined['game_date'])
    for col in ['home_team', 'away_team']:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).astype('object')
    combined = combined.sort_values('game_date').reset_index(drop=True)

    print(f"Loaded {len(combined)} games from {len(all_dfs)} team files")
    print(f"Date range: {combined['game_date'].min()} to {combined['game_date'].max()}")
    return combined


def prepare_features(df: pd.DataFrame, include_interactions: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare feature matrix and target variables."""
    data = df.copy()
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = data.dropna(subset=required_cols)

    X_weather = data[WEATHER_FEATURES].copy()
    X_weather['is_night'] = (data['day_night'] == 'night').astype(int)

    parks = np.array(data['home_team'].tolist()).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    park_encoded = encoder.fit_transform(parks)
    park_columns = [f'park_{cat}' for cat in encoder.categories_[0]]
    X_park = pd.DataFrame(park_encoded, columns=park_columns, index=data.index)

    X = pd.concat([X_weather.reset_index(drop=True), X_park.reset_index(drop=True)], axis=1)

    if include_interactions:
        interaction_data = {f'{wf}_x_{pc}': X[wf].values * X[pc].values
                           for wf in WEATHER_FEATURES for pc in park_columns}
        X = pd.concat([X, pd.DataFrame(interaction_data, index=X.index)], axis=1)

    y_strikeouts = data[TARGET_STRIKEOUTS].reset_index(drop=True)
    y_runs = data[TARGET_RUNS].reset_index(drop=True)
    X.attrs['park_columns'] = park_columns
    X.attrs['weather_features'] = WEATHER_FEATURES
    X.attrs['park_encoder'] = encoder
    return X, y_strikeouts, y_runs


def train_test_split_by_season(df: pd.DataFrame, X: pd.DataFrame, y_strikeouts: pd.Series,
                                y_runs: pd.Series, test_start_season: int = 2023) -> Dict[str, np.ndarray]:
    """Split data into train/test based on season."""
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = df.dropna(subset=required_cols).reset_index(drop=True)
    seasons = data['season']

    train_mask = seasons < test_start_season
    test_mask = seasons >= test_start_season

    splits = {
        'X_train': X[train_mask].values, 'X_test': X[test_mask].values,
        'y_train_strikeouts': y_strikeouts[train_mask].values, 'y_test_strikeouts': y_strikeouts[test_mask].values,
        'y_train_runs': y_runs[train_mask].values, 'y_test_runs': y_runs[test_mask].values,
        'feature_names': X.columns.tolist(),
        'train_seasons': sorted(seasons[train_mask].unique()),
        'test_seasons': sorted(seasons[test_mask].unique()),
    }
    print(f"Train: {len(splits['X_train'])} games, Test: {len(splits['X_test'])} games")
    return splits


def create_random_month_split(df: pd.DataFrame, test_size: float = 0.30,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Create train/test split by randomly assigning year-month units."""
    if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])

    year_months = df['game_date'].dt.to_period('M').astype(str)
    unique_months = sorted(year_months.unique())
    n_months = len(unique_months)
    n_test = int(np.ceil(n_months * test_size))

    np.random.seed(random_state)
    shuffled = np.random.permutation(n_months)
    train_months = [unique_months[i] for i in sorted(shuffled[:-n_test])]
    test_months = [unique_months[i] for i in sorted(shuffled[-n_test:])]

    train_mask = year_months.isin(train_months).values
    test_mask = year_months.isin(test_months).values
    print(f"Random Month Split: {len(train_months)} train months, {len(test_months)} test months")
    return train_mask, test_mask, train_months, test_months


def prepare_nn_data(df: pd.DataFrame, test_start_season: int = 2023) -> Dict[str, np.ndarray]:
    """Prepare data for neural network with park embeddings."""
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = df.dropna(subset=required_cols).reset_index(drop=True)

    X_weather = data[WEATHER_FEATURES].copy()
    X_weather['is_night'] = (data['day_night'] == 'night').astype(int)

    unique_parks = sorted(data['home_team'].unique())
    park_to_id = {park: i for i, park in enumerate(unique_parks)}
    park_ids = data['home_team'].map(park_to_id).values

    seasons = data['season']
    train_mask = seasons < test_start_season

    scaler = StandardScaler()
    X_weather_train = scaler.fit_transform(X_weather[train_mask])
    X_weather_test = scaler.transform(X_weather[~train_mask])

    return {
        'X_weather_train': X_weather_train.astype(np.float32),
        'X_weather_test': X_weather_test.astype(np.float32),
        'park_ids_train': park_ids[train_mask].astype(np.int64),
        'park_ids_test': park_ids[~train_mask].astype(np.int64),
        'y_train_strikeouts': data.loc[train_mask, TARGET_STRIKEOUTS].values.astype(np.float32),
        'y_test_strikeouts': data.loc[~train_mask, TARGET_STRIKEOUTS].values.astype(np.float32),
        'y_train_runs': data.loc[train_mask, TARGET_RUNS].values.astype(np.float32),
        'y_test_runs': data.loc[~train_mask, TARGET_RUNS].values.astype(np.float32),
        'park_to_id': park_to_id, 'id_to_park': {v: k for k, v in park_to_id.items()},
        'n_parks': len(unique_parks), 'n_weather_features': X_weather.shape[1],
        'weather_scaler': scaler, 'weather_feature_names': X_weather.columns.tolist(),
    }


def get_park_game_counts(df: pd.DataFrame, test_start_season: int = 2023) -> pd.DataFrame:
    """Get game counts per park in train and test sets."""
    required_cols = WEATHER_FEATURES + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team']
    data = df.dropna(subset=required_cols)
    train_counts = data[data['season'] < test_start_season].groupby('home_team').size()
    test_counts = data[data['season'] >= test_start_season].groupby('home_team').size()
    counts = pd.DataFrame({'train_games': train_counts, 'test_games': test_counts}).fillna(0).astype(int)
    counts['total_games'] = counts['train_games'] + counts['test_games']
    return counts.sort_values('total_games', ascending=False)


def prepare_mixed_effects_data(df: pd.DataFrame, standardize_weather: bool = True,
                                test_start_season: int = 2023, use_enhanced_features: bool = False,
                                include_interactions: bool = False, scaler_type: str = 'robust') -> Dict:
    """Prepare data for mixed-effects regression models."""
    if use_enhanced_features:
        data = add_enhanced_weather_features(df)
        weather_features = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'heat_index']
        if 'elevation_ft' in data.columns:
            weather_features.append('elevation_ft')
        if 'has_roof' in data.columns:
            weather_features.append('has_roof')
    else:
        data = df.copy()
        weather_features = WEATHER_FEATURES_BASIC.copy()
        if 'air_density' not in data.columns:
            pres = data['pres'].values if 'pres' in data.columns else None
            data['air_density'] = compute_air_density(data['temp_f'].values, data['rhum'].values, pres)

    required_cols = [c for c in weather_features + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night',
                     'home_team', 'away_team', 'season'] if c in data.columns]
    data = data.dropna(subset=required_cols).copy()
    data['is_night'] = (data['day_night'] == 'night').astype(int)

    if include_interactions and use_enhanced_features:
        if 'has_roof' in data.columns:
            for feat in ['temp_f', 'rhum', 'wspd_mph', 'air_density']:
                if feat in data.columns:
                    data[f'{feat}_x_roof'] = data[feat] * data['has_roof']
                    weather_features.append(f'{feat}_x_roof')
        if 'elevation_ft' in data.columns and 'air_density' in data.columns:
            data['elevation_norm'] = data['elevation_ft'] / 5280
            data['elevation_x_density'] = data['elevation_norm'] * data['air_density']
            weather_features.extend(['elevation_norm', 'elevation_x_density'])

    for col in ['home_team', 'away_team', 'season']:
        data[col] = pd.Series([str(x) for x in data[col].values], dtype='object', index=data.index)

    train_mask = data['season'].astype(int) < test_start_season
    df_train = data[train_mask].copy().reset_index(drop=True)
    df_test = data[~train_mask].copy().reset_index(drop=True)

    weather_scaler = None
    features_to_scale = [f for f in weather_features if f not in ['has_roof']]

    if standardize_weather:
        weather_scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        df_train[features_to_scale] = weather_scaler.fit_transform(df_train[features_to_scale])
        df_test[features_to_scale] = weather_scaler.transform(df_test[features_to_scale])

    group_info = {
        'home_teams': sorted(data['home_team'].unique().tolist()),
        'away_teams': sorted(data['away_team'].unique().tolist()),
        'seasons': sorted(data['season'].unique().tolist()),
        'n_home_teams': data['home_team'].nunique(),
        'n_away_teams': data['away_team'].nunique(),
        'n_seasons': data['season'].nunique(),
        'games_per_home_team': df_train.groupby('home_team').size().to_dict(),
        'games_per_away_team': df_train.groupby('away_team').size().to_dict(),
        'games_per_season': df_train.groupby('season').size().to_dict(),
    }

    print(f"Mixed-Effects Data: {len(df_train)} train, {len(df_test)} test, {group_info['n_home_teams']} parks")
    return {
        'df_train': df_train, 'df_test': df_test, 'weather_scaler': weather_scaler,
        'scaler_type': scaler_type if standardize_weather else None, 'group_info': group_info,
        'weather_features': weather_features, 'features_scaled': features_to_scale,
        'target_strikeouts': TARGET_STRIKEOUTS, 'target_runs': TARGET_RUNS,
        'use_enhanced_features': use_enhanced_features,
    }


def compute_team_season_averages(df: pd.DataFrame, method: str = 'expanding',
                                  min_games: int = 10) -> pd.DataFrame:
    """Compute away team season averages for strikeouts and runs."""
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['game_date']):
        data['game_date'] = pd.to_datetime(data['game_date'])
    data = data.sort_values(['away_team', 'season', 'game_date']).reset_index(drop=True)
    data['away_team_avg_k'] = np.nan
    data['away_team_avg_runs'] = np.nan
    data['away_team_games'] = 0

    if method == 'loo':
        for (team, season), group in data.groupby(['away_team', 'season']):
            indices = group.index.tolist()
            if len(indices) < 2:
                continue
            total_k, total_runs = group['away_bat_k'].sum(), group['away_runs_scored'].sum()
            for idx in indices:
                other_k = total_k - data.loc[idx, 'away_bat_k']
                other_runs = total_runs - data.loc[idx, 'away_runs_scored']
                other_games = len(indices) - 1
                if other_games > 0:
                    data.loc[idx, 'away_team_avg_k'] = other_k / other_games
                    data.loc[idx, 'away_team_avg_runs'] = other_runs / other_games
                    data.loc[idx, 'away_team_games'] = other_games
    elif method == 'expanding':
        for (team, season), group in data.groupby(['away_team', 'season']):
            indices = group.index.tolist()
            k_values, runs_values = group['away_bat_k'].tolist(), group['away_runs_scored'].tolist()
            cumsum_k, cumsum_runs, n_prior = 0, 0, 0
            for i, idx in enumerate(indices):
                if n_prior >= 1:
                    data.loc[idx, 'away_team_avg_k'] = cumsum_k / n_prior
                    data.loc[idx, 'away_team_avg_runs'] = cumsum_runs / n_prior
                    data.loc[idx, 'away_team_games'] = n_prior
                if pd.notna(k_values[i]):
                    cumsum_k += k_values[i]
                if pd.notna(runs_values[i]):
                    cumsum_runs += runs_values[i]
                n_prior += 1
    else:
        raise ValueError(f"Unknown method: {method}")

    return data[['away_team_avg_k', 'away_team_avg_runs', 'away_team_games']]


def compute_park_adjustment_factors(league_avgs_path: str = None) -> pd.DataFrame:
    """Compute park adjustment factors from league batting averages."""
    if league_avgs_path is None:
        for p in [Path(__file__).parent.parent / 'data' / 'league_batting_avgs_2021_2025.csv',
                  Path('data/league_batting_avgs_2021_2025.csv')]:
            if p.exists():
                league_avgs_path = p
                break
        if league_avgs_path is None:
            raise FileNotFoundError("Could not find league_batting_avgs_2021_2025.csv")

    league_df = pd.read_csv(league_avgs_path)
    league_avg_row = league_df[league_df['home_team'] == 'LEAGUE AVG']
    if league_avg_row.empty:
        raise ValueError("LEAGUE AVG row not found")

    league_avg_k = league_avg_row['strikeouts'].values[0]
    league_avg_runs = league_avg_row['total_runs'].values[0]

    park_df = league_df[league_df['home_team'] != 'LEAGUE AVG'].copy()
    park_df['k_factor'] = park_df['strikeouts'] / league_avg_k
    park_df['runs_factor'] = park_df['total_runs'] / league_avg_runs

    result = park_df.set_index('home_team')[['k_factor', 'runs_factor']].copy()
    result.attrs['league_avg_k'] = league_avg_k
    result.attrs['league_avg_runs'] = league_avg_runs
    return result


def compute_deviation_targets(df: pd.DataFrame, method: str = 'loo',
                               min_games: int = 10, league_avgs_path: str = None) -> pd.DataFrame:
    """Compute deviation targets: actual - expected (team_avg * park_factor)."""
    data = df.copy()
    team_avgs = compute_team_season_averages(data, method=method, min_games=min_games)
    data = pd.concat([data, team_avgs], axis=1)

    park_factors = compute_park_adjustment_factors(league_avgs_path)
    lookup_team = data['home_team'].replace({'OAK': 'ATH', 'ARI': 'AZ'})
    data['k_factor'] = lookup_team.map(park_factors['k_factor']).fillna(1.0)
    data['runs_factor'] = lookup_team.map(park_factors['runs_factor']).fillna(1.0)

    data['expected_away_k'] = data['away_team_avg_k'] * data['k_factor']
    data['expected_away_runs'] = data['away_team_avg_runs'] * data['runs_factor']
    data['deviation_away_k'] = data['away_bat_k'] - data['expected_away_k']
    data['deviation_away_runs'] = data['away_runs_scored'] - data['expected_away_runs']
    return data


def prepare_deviation_data(df: pd.DataFrame, standardize_weather: bool = True,
                            test_start_season: int = 2023, use_enhanced_features: bool = False,
                            include_interactions: bool = False, deviation_method: str = 'expanding',
                            min_games: int = 10) -> Dict:
    """Prepare data for deviation-based mixed-effects models."""
    data_with_deviations = compute_deviation_targets(df, method=deviation_method, min_games=min_games)
    me_data = prepare_mixed_effects_data(data_with_deviations, standardize_weather, test_start_season,
                                          use_enhanced_features, include_interactions)

    me_data['deviation_columns'] = ['deviation_away_k', 'deviation_away_runs']
    me_data['expected_columns'] = ['expected_away_k', 'expected_away_runs']
    me_data['team_avg_columns'] = ['away_team_avg_k', 'away_team_avg_runs', 'away_team_games']
    me_data['park_factor_columns'] = ['k_factor', 'runs_factor']
    me_data['deviation_method'] = deviation_method
    me_data['min_games'] = min_games
    me_data['n_valid_deviations'] = {
        'train': me_data['df_train']['deviation_away_k'].notna().sum(),
        'test': me_data['df_test']['deviation_away_k'].notna().sum()
    }
    me_data['target_deviation_k'] = 'deviation_away_k'
    me_data['target_deviation_runs'] = 'deviation_away_runs'

    print(f"Deviation Data: {me_data['n_valid_deviations']['train']} valid train, {me_data['n_valid_deviations']['test']} valid test")
    return me_data


def prepare_park_weather_interactions(df: pd.DataFrame, weather_features: List[str] = ['temp_f', 'wspd_mph', 'wind_cf'],
                                        standardize: bool = True, test_start_season: int = 2023,
                                        scaler_type: str = 'robust', split_method: str = 'season',
                                        test_size: float = 0.30, random_state: int = 42,
                                        include_roof_interactions: bool = False,
                                        include_elevation: bool = False,
                                        include_marine_layer: bool = False) -> Dict:
    """Prepare data for Ridge/Lasso regression with explicit park × weather interactions."""
    data = df.copy()
    required_cols = weather_features + [TARGET_STRIKEOUTS, TARGET_RUNS, 'day_night', 'home_team', 'season']
    data = data.dropna(subset=required_cols).copy()
    data['home_team'] = data['home_team'].astype(str).astype('object')
    data['season'] = data['season'].astype(int)
    data['is_night'] = (data['day_night'] == 'night').astype(int)

    if split_method == 'season':
        train_mask = data['season'] < test_start_season
        test_mask = ~train_mask
        split_info = {'method': 'season', 'test_start_season': test_start_season}
    elif split_method == 'random_month':
        train_mask, test_mask, train_months, test_months = create_random_month_split(data, test_size, random_state)
        split_info = {'method': 'random_month', 'train_months': train_months, 'test_months': test_months}
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    df_train = data[train_mask].copy().reset_index(drop=True)
    df_test = data[test_mask].copy().reset_index(drop=True)
    parks = sorted(df_train['home_team'].unique())

    roof_features, roof_interaction_features = [], []
    if include_roof_interactions:
        try:
            stadium_params = load_stadium_parameters()
            for df_split in [df_train, df_test]:
                lookup_team = df_split['home_team'].replace({'ARI': 'AZ', 'OAK': 'ATH'})
                df_split['has_roof'] = lookup_team.map(stadium_params['has_roof']).fillna(0).astype(int)
            roof_features = ['has_roof']
        except FileNotFoundError:
            warnings.warn("Could not load stadium parameters.")
            include_roof_interactions = False

    # Add elevation feature (v6)
    elevation_features = []
    elevation_map = None
    if include_elevation:
        try:
            elevation_map = load_elevation_data()
            for df_split in [df_train, df_test]:
                df_split['elevation_ft'] = df_split['home_team'].map(elevation_map).fillna(
                    elevation_map.get('_league_avg', 512.6))
            elevation_features = ['elevation_ft']
        except FileNotFoundError:
            warnings.warn("Could not load elevation data.")
            include_elevation = False

    # Add marine layer features (v6)
    marine_layer_features = []
    if include_marine_layer:
        for df_split in [df_train, df_test]:
            df_split['is_marine_layer'] = df_split['home_team'].isin(MARINE_LAYER_PARKS).astype(int)
            df_split['is_day'] = 1 - df_split['is_night']
            # Marine layer × day/night interactions
            df_split['marine_layer_day'] = df_split['is_marine_layer'] * df_split['is_day']
            df_split['marine_layer_night'] = df_split['is_marine_layer'] * df_split['is_night']
        marine_layer_features = ['marine_layer_day', 'marine_layer_night']

    weather_scaler, scaling_params = None, {}
    features_to_scale = weather_features + elevation_features  # Elevation is scaled like weather features
    if standardize:
        weather_scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        df_train[features_to_scale] = weather_scaler.fit_transform(df_train[features_to_scale])
        df_test[features_to_scale] = weather_scaler.transform(df_test[features_to_scale])
        attr = 'center_' if scaler_type == 'robust' else 'mean_'
        for i, feat in enumerate(features_to_scale):
            scaling_params[feat] = {'center': getattr(weather_scaler, attr)[i], 'scale': weather_scaler.scale_[i]}

    park_dummies = [f'park_{p}' for p in parks]
    train_park = {f'park_{p}': (df_train['home_team'] == p).astype(int) for p in parks}
    test_park = {f'park_{p}': (df_test['home_team'] == p).astype(int) for p in parks}

    interaction_features = []
    train_interactions, test_interactions = {}, {}
    for wf in weather_features:
        for p in parks:
            name = f'{wf}_x_{p}'
            interaction_features.append(name)
            train_interactions[name] = df_train[wf].values * train_park[f'park_{p}'].values
            test_interactions[name] = df_test[wf].values * test_park[f'park_{p}'].values

    if include_roof_interactions and 'has_roof' in df_train.columns:
        for wf in weather_features:
            name = f'has_roof_x_{wf}'
            roof_interaction_features.append(name)
            train_interactions[name] = df_train['has_roof'].values * df_train[wf].values
            test_interactions[name] = df_test['has_roof'].values * df_test[wf].values

    df_train = pd.concat([df_train, pd.DataFrame(train_park, index=df_train.index),
                          pd.DataFrame(train_interactions, index=df_train.index)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(test_park, index=df_test.index),
                         pd.DataFrame(test_interactions, index=df_test.index)], axis=1)

    # Build feature list: weather + elevation + is_night + marine_layer + roof + park_dummies + interactions
    feature_names = (weather_features + elevation_features + ['is_night'] +
                     marine_layer_features + roof_features + park_dummies +
                     interaction_features + roof_interaction_features)

    print(f"Park-Weather Interactions: {len(df_train)} train, {len(df_test)} test, {len(feature_names)} features")
    return {
        'df_train': df_train, 'df_test': df_test,
        'X_train': df_train[feature_names].values, 'X_test': df_test[feature_names].values,
        'y_train_strikeouts': df_train[TARGET_STRIKEOUTS].values,
        'y_test_strikeouts': df_test[TARGET_STRIKEOUTS].values,
        'y_train_runs': df_train[TARGET_RUNS].values, 'y_test_runs': df_test[TARGET_RUNS].values,
        'feature_names': feature_names, 'weather_features': weather_features,
        'elevation_features': elevation_features, 'marine_layer_features': marine_layer_features,
        'park_dummies': park_dummies, 'interaction_features': interaction_features,
        'roof_features': roof_features, 'roof_interaction_features': roof_interaction_features,
        'weather_scaler': weather_scaler, 'scaling_params': scaling_params,
        'scaler_type': scaler_type if standardize else None, 'parks': parks,
        'target_strikeouts': TARGET_STRIKEOUTS, 'target_runs': TARGET_RUNS, 'split_info': split_info,
        'elevation_map': elevation_map if include_elevation else None,
        'marine_layer_parks': MARINE_LAYER_PARKS if include_marine_layer else None,
    }


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\nPreparing features with interactions...")
    X, y_k, y_runs = prepare_features(df, include_interactions=True)
    print(f"Feature matrix shape: {X.shape}")

    print("\nCreating train/test split...")
    splits = train_test_split_by_season(df, X, y_k, y_runs)

    print("\nPreparing mixed-effects data (basic)...")
    me_data = prepare_mixed_effects_data(df, use_enhanced_features=False)

    print("\nPreparing mixed-effects data (enhanced)...")
    me_data_enhanced = prepare_mixed_effects_data(df, use_enhanced_features=True, include_interactions=True)

    print("\nPark game counts:")
    print(get_park_game_counts(df))

    print("\nTesting deviation targets...")
    try:
        dev_data = prepare_deviation_data(df, use_enhanced_features=False)
        print(f"Deviation columns: {dev_data['deviation_columns']}")
    except Exception as e:
        print(f"Error: {e}")
