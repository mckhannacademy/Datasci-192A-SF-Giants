# Model v6: Elevation and Marine Layer Features

## Overview

Model v6 extends v5 by adding:
1. **Elevation as a fixed effect** - Stadium altitude affects ball flight beyond what air density captures
2. **Marine layer binary variable** with day/night interaction - Pacific coast fog effects

## New Features

### Elevation (`elevation_ft`)

Elevation is added as a scaled weather feature. It captures altitude-specific effects that go beyond air density:

| Park | Elevation (ft) | Notable |
|------|---------------|---------|
| COL (Coors Field) | 5,280 | Highest - famous for offense |
| ATL (Truist Park) | 1,050 | Above average |
| ARI (Chase Field) | 1,090 | High elevation |
| SF (Oracle Park) | 65 | Near sea level |
| OAK (Oakland Coliseum) | -21 | Below sea level |

**Why elevation matters beyond air density:**
- Air density accounts for the physics of ball flight at different altitudes
- Elevation as a fixed effect captures additional altitude-related factors:
  - Pitcher fatigue at altitude
  - Player acclimation effects
  - Humidity patterns specific to altitude bands

### Marine Layer

The marine layer is a Pacific coast phenomenon — cool, moist air (fog) that forms over the ocean and moves inland. It significantly affects:
- Temperature (cooling effect)
- Humidity (increases)
- Ball flight (denser, wetter air)
- Visibility

**Marine Layer Parks:**
- SF (Oracle Park) - Famous McCovey Cove fog
- OAK (Oakland Coliseum) - Same bay system as SF
- SD (Petco Park) - Coastal fog common
- LAA (Angel Stadium) - Marine influence
- LAD (Dodger Stadium) - Marine influence
- SEA (T-Mobile Park) - Pacific Northwest fog

**Day/Night Interaction:**
- `marine_layer_day` - Day games, fog typically burns off
- `marine_layer_night` - Night games, fog rolls in (stronger effect expected)

## Files

| File | Description |
|------|-------------|
| `train_model_v6.py` | Training script with v6 features |
| `explainer_v6.py` | Prediction explainer with component breakdown |
| `dashboard_model_v6.js` | JavaScript implementation for client-side |
| `dashboard_params_v6.json` | Model coefficients (generated) |
| `README.md` | This file |

## Usage

### Training the Model

```bash
cd models/v6
python train_model_v6.py
```

This will:
1. Load game data with weather features
2. Add elevation from `data/elevation.csv`
3. Create marine layer × day/night interaction features
4. Train Ridge regression models for strikeouts and runs
5. Export `dashboard_params_v6.json`

### Python Explainer

```python
from models.v6.explainer_v6 import PredictionExplainerV6

explainer = PredictionExplainerV6()

# Component breakdown
result = explainer.explain_components(
    park='SF',
    weather={'temp_f': 55, 'rhum': 85, 'wspd_mph': 12, 'wind_cf': -5, 'is_night': True},
    target='runs'
)
print(result['components'])

# Elevation-specific analysis
elev = explainer.get_elevation_contribution(park='COL', weather={...}, target='runs')
print(elev['interpretation'])

# Marine layer analysis
ml = explainer.get_marine_layer_contribution(park='SF', weather={...}, target='runs')
print(ml['interpretation'])
```

### JavaScript Dashboard

```javascript
import { PredictionModel } from './dashboard_model_v6.js';

// Load params from dashboard_params_v6.json
const model = new PredictionModel(paramsJson);

// Make prediction
const result = model.predict('SF', {
    temp_f: 55,
    rhum: 85,
    wspd_mph: 12,
    wind_cf: -5,
    is_night: true
}, 'runs');

console.log(result.prediction);
console.log(result.components);
```

## Model Structure

```
y = Intercept
    + β_temp × temp_std
    + β_rhum × rhum_std
    + β_wspd × wspd_std
    + β_wind_cf × wind_cf_std
    + β_air_density × air_density_std
    + β_elevation × elevation_std           # NEW in v6
    + β_is_night × is_night
    + β_marine_day × (is_marine × is_day)   # NEW in v6
    + β_marine_night × (is_marine × is_night)  # NEW in v6
    + park_effect[park]
    + Σ (β_weather_x_park × weather_std × park_dummy)
```

## Component Breakdown

The v6 explainer groups contributions into three categories:

### 1. Weather Effects
- Temperature
- Humidity
- Wind speed
- Wind direction (to CF)
- Air density
- **Elevation** (new)

### 2. Time Effects
- Day/Night
- **Marine layer** (new)

### 3. Park Effects
- Base park effect
- Park × weather interactions

## Expected Coefficient Signs

| Feature | Strikeouts | Runs | Rationale |
|---------|-----------|------|-----------|
| `elevation_ft` | - or neutral | + | Higher elevation → more offense |
| `marine_layer_day` | + or neutral | - or neutral | Fog burns off during day |
| `marine_layer_night` | + | - | Fog rolls in, suppresses offense |

## Data Sources

- **Elevation**: `data/elevation.csv` (authoritative)
- **Weather**: Individual team CSV files in `data/`
- **League averages**: 512.6 ft (mean elevation across all parks)

## Changes from v5

1. Added `elevation_ft` to weather features (scaled with other weather features)
2. Added `marine_layer_day` and `marine_layer_night` binary interaction features
3. Updated `data_prep.py` with `include_elevation` and `include_marine_layer` options
4. New `explain_components()` method for grouped contribution breakdown
5. New `get_elevation_contribution()` and `get_marine_layer_contribution()` methods
6. **Removed roof/dome dampening** - We don't have data on whether roofs are open or closed, so weather effects are always applied as modeled for all parks including dome stadiums

## Verification

After training, check:

1. **Elevation coefficient** - Higher elevation should correlate with more runs
2. **Marine layer night** - Should have negative effect on runs (fog suppresses offense)
3. **Component totals** - Should sum to the prediction value
