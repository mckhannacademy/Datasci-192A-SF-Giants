Group Project Repository for Datasci 192A

# models

Regression models for predicting weather effects on MLB game outcomes (strikeouts and runs). Models estimate park-specific weather sensitivities using Ridge regression with interaction terms.

## Directory Structure

```
models/
├── __init__.py                      # Package exports
├── data_prep.py                     # Data loading and feature engineering
├── ridge_lasso_model.py             # Ridge/Lasso regression models
├── nn_embedding_model.py            # Neural network with park embeddings
├── mixed_effects_model.py           # Mixed-effects models (statsmodels)
├── park_effects.py                  # Park effect extraction and visualization
├── park_weather_interaction_model.py # Park × weather interactions
├── predict.py                       # Inference CLI
├── explainer.py                     # Prediction explanations
├── evaluate.py                      # Metrics and diagnostics
├── v6/                              # Current model version
│   ├── train_model_v6.py            # v6 training script
│   ├── explainer_v6.py              # v6 explainer with elevation/marine layer
│   ├── dashboard_model_v6.js        # JavaScript prediction client
│   └── dashboard_params_v6.json     # Serialized model parameters
├── dashboard_model_v*.js            # JavaScript clients (v4, v5)
├── dashboard_params_v*.json         # Model parameters (v4, v5)
├── baseline_stats.json              # League-average baselines
└── *.joblib, *.pt                   # Saved sklearn/torch models
```


## Core Modules

### data_prep.py
Data loading and feature engineering.

```python
from models import load_all_team_data, prepare_features, WEATHER_FEATURES

df = load_all_team_data()
X, y = prepare_features(df, target='away_strikeouts')
```

Key exports:
- `load_all_team_data()` - Load game data from all teams
- `prepare_features()` - Create feature matrix with weather variables
- `compute_air_density()` - Physics-based air density calculation
- `add_enhanced_weather_features()` - Add derived features
- `WEATHER_FEATURES`, `WEATHER_FEATURES_BASIC`, `WEATHER_FEATURES_ENHANCED`

### ridge_lasso_model.py
Ridge and Lasso regression with park interactions.

```python
from models import ParkWeatherRegressor

model = ParkWeatherRegressor(alpha=1.0)
model.fit(X_train, y_train, parks_train)
predictions = model.predict(X_test, parks_test)
```

### predict.py
CLI and API for inference.

```bash
# CLI usage
python -m models.predict --park SF --temp 68 --humidity 75 --wind_speed 12 --wind_cf 0.5 --night
```

```python
# Python API
from models.predict import predict_game

result = predict_game('SF', temp_f=68, rhum=75, wspd_mph=12, wind_cf=0.5, is_night=True)
print(result['strikeouts'], result['runs'])
```

### explainer.py
Prediction explanations for waterfall charts.

```python
from models.explainer import PredictionExplainer

explainer = PredictionExplainer()
result = explainer.explain_for_dashboard(park='SF', temp_f=85, rhum=60, wspd_mph=8, wind_cf=0.3)
```

### evaluate.py
Model evaluation and diagnostics.

```python
from models import compute_metrics, compare_models, plot_residuals

metrics = compute_metrics(y_true, y_pred)
comparison = compare_models({'ridge': ridge_preds, 'lasso': lasso_preds}, y_true)
```

### park_effects.py
Park-specific coefficient extraction and visualization.

```python
from models import extract_park_specific_coefficients, plot_weather_effects_heatmap

coeffs = extract_park_specific_coefficients(model)
plot_weather_effects_heatmap(coeffs, feature='temp_f')
```

## Dashboard Integration

The trained model exports to JSON for browser-side predictions:

1. **Training** produces `dashboard_params_v6.json` with:
   - Scaling parameters (median, IQR)
   - Fixed effect coefficients
   - Park-specific interaction coefficients

2. **JavaScript client** (`dashboard_model_v6.js`) loads params and computes predictions:
   ```javascript
   import { predictGame, explainPrediction } from './v6/dashboard_model_v6.js';

   const result = predictGame('SF', { temp_f: 72, rhum: 65, wspd_mph: 10, wind_cf: 0.2 });
   ```

3. **Explainer** provides component breakdowns for waterfall visualizations.

## Dependencies

Required:
- numpy
- pandas
- scikit-learn
- joblib

Optional:
- torch (for neural network models)
- statsmodels (for mixed-effects models)
- matplotlib (for visualization)
