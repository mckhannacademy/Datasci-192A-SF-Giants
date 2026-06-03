# Retraining Instructions for Model v6

This guide explains how to retrain the v6 weather prediction model with new or updated data.

## Prerequisites

- Python 3.x
- Required packages: `pandas`, `numpy`, `scikit-learn`, `joblib`
- Working directory: Repository root (`Datasci-192A-SF-Giants/`)

## Data File Requirements

### Location
All team data files should be placed in the `data/` directory.

### Naming Convention
```
{team}_data_{start_year}_day_night.csv
```

Examples:
- `sf_data_2019_day_night.csv`
- `lad_data_2019_day_night.csv`

### Required Columns (48 total)

Key columns that must be present:

| Column | Description |
|--------|-------------|
| `game_pk` | Unique game identifier |
| `game_date` | Date of the game |
| `season` | Year of the season |
| `away_team` | Away team abbreviation |
| `home_team` | Home team abbreviation |
| `temp_f` | Temperature in Fahrenheit |
| `rhum` | Relative humidity (%) |
| `pres` | Atmospheric pressure |
| `wspd_mph` | Wind speed in mph |
| `wind_cf` | Wind correction factor |
| `day_night` | Game time indicator (values: `"day"` or `"night"`) |
| `away_bat_k` | Away team strikeouts (target for strikeouts model) |
| `away_runs_scored` | Away team runs scored (target for runs model) |

## Reference Files

These files are required for training but should **not be modified** unless adding new teams/stadiums:

| File | Purpose |
|------|---------|
| `data/elevation.csv` | Stadium elevations (feet above sea level) |
| `analysis/team_parameters.csv` | Stadium metadata (roof status, coordinates) |
| `models/data_prep.py` | Team name mappings (`TEAM_FILE_TO_ABBREV` dictionary) |

## Configuration Parameters

Configuration is set in `models/v6/train_model_v6.py` at lines 72-82:

| Parameter | Default | Options |
|-----------|---------|---------|
| `weather_features` | `['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']` | Any subset of available weather columns |
| `split_method` | `'random_month'` | `'random_month'`, `'season'` |
| `test_size` | `0.30` | 0.0 - 1.0 (fraction for test set) |
| `random_state` | `42` | Any integer (for reproducibility) |
| `scaler_type` | `'robust'` | `'robust'`, `'standard'` |
| `include_elevation` | `True` | `True`, `False` |
| `include_marine_layer` | `True` | `True`, `False` |

### Parameter Details

- **split_method**:
  - `'random_month'`: Randomly assigns entire months to train/test (prevents data leakage from games in same series)
  - `'season'`: Splits by season year

- **scaler_type**:
  - `'robust'`: Uses median and IQR, better for outliers
  - `'standard'`: Uses mean and standard deviation

## Retraining Steps

```bash
# Step 1: Navigate to repository root
cd /path/to/Datasci-192A-SF-Giants

# Step 2: Run training script
python models/v6/train_model_v6.py
```

The script will:
1. Load all team data files from `data/`
2. Merge with elevation and stadium metadata
3. Calculate derived features (air density, marine layer)
4. Train Ridge regression models for strikeouts and runs
5. Export parameters for the JavaScript dashboard

## Output Files

| File | Size | Purpose |
|------|------|---------|
| `models/v6/dashboard_params_v6.json` | ~33KB | Model parameters for browser-side predictions |

The JSON file contains:
- Model coefficients and intercepts
- Scaler parameters (center, scale)
- Feature names and ordering
- Marine layer lookup table
- Team mappings

## Verification

After training completes:

1. **Check output file exists and is updated**:
   ```bash
   ls -la models/v6/dashboard_params_v6.json
   ```

2. **Review R² scores in training output**:
   - Expected R²: ~4% for strikeouts, ~2% for runs
   - These low values are expected - weather explains limited variance in baseball outcomes
   - The model captures real but small weather effects

3. **Validate JSON output**:
   ```bash
   python -c "import json; json.load(open('models/v6/dashboard_params_v6.json'))"
   ```

## Troubleshooting

### Missing Data Files
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/sf_data_2019_day_night.csv'
```
**Solution**: Ensure all team data files are in `data/` with correct naming convention.

### Missing Reference Files
```
FileNotFoundError: ... 'data/elevation.csv'
```
**Solution**: Verify `data/elevation.csv` and `analysis/team_parameters.csv` exist.

### Low R² Values
R² values of 2-5% are normal and expected. Weather is one of many factors affecting game outcomes. If R² drops significantly below these ranges, check:
- Data quality and completeness
- Feature scaling issues
- Data file integrity

### Memory Issues
For large datasets, the script may require significant memory. Solutions:
- Close other applications
- Process fewer teams at once
- Increase system swap space

### Team Name Mismatches
```
KeyError: 'TEAM_NAME'
```
**Solution**: Check `models/data_prep.py` and ensure the `TEAM_FILE_TO_ABBREV` dictionary includes all teams in your data files.

## Adding New Teams/Stadiums

To add a new team:

1. **Create data file**: Add `{team}_data_{year}_day_night.csv` to `data/`

2. **Update elevation data**: Add row to `data/elevation.csv`:
   ```csv
   Team,Stadium,Elevation_ft
   NEW,New Stadium Name,elevation_value
   ```

3. **Update team parameters**: Add row to `analysis/team_parameters.csv` with:
   - Team abbreviation
   - Stadium coordinates (latitude, longitude)
   - Roof status (`roof` column: 0=open, 1=retractable, 2=dome)
   - Coastal flag for marine layer

4. **Update team mappings**: Add entry to `TEAM_FILE_TO_ABBREV` in `models/data_prep.py`:
   ```python
   TEAM_FILE_TO_ABBREV = {
       # ... existing teams ...
       'new': 'NEW',  # Add new team
   }
   ```

5. **Retrain the model** following the steps above.
