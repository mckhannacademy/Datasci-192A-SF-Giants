#!/usr/bin/env python3
"""
Train Model v5: Full Weather Feature Set

Adds rhum (humidity) and air_density to the model, which were missing in v4.

Full feature set:
- temp_f: Air temperature (°F)
- rhum: Relative humidity (%)
- wspd_mph: Wind speed (mph)
- wind_cf: Wind component toward center field
- air_density: Computed from temp/humidity/pressure (kg/m³)
- is_night: Night game indicator
- has_roof: Retractable/dome roof indicator
"""

import json
from datetime import datetime
from pathlib import Path

from data_prep import (
    load_all_team_data,
    add_enhanced_weather_features,
    prepare_park_weather_interactions,
)
from park_weather_interaction_model import ParkWeatherInteractionModel


def train_v5_models(data_dir: Path = None, output_path: Path = None):
    """Train v5 models with full weather feature set."""

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    if output_path is None:
        output_path = Path(__file__).parent / 'dashboard_params_v5.json'

    print("=" * 60)
    print("TRAINING MODEL V5 WITH FULL WEATHER FEATURES")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    df = load_all_team_data(data_dir)

    # Add enhanced features (air_density, heat_index, etc.)
    print("\n[2/5] Adding enhanced weather features...")
    df = add_enhanced_weather_features(df)

    # Verify features are present
    required_features = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    print(f"   Features available: {required_features}")
    print(f"   air_density range: {df['air_density'].min():.4f} to {df['air_density'].max():.4f}")
    print(f"   rhum range: {df['rhum'].min():.1f}% to {df['rhum'].max():.1f}%")

    # Prepare data with FULL feature set
    print("\n[3/5] Preparing park-weather interactions with full features...")
    data = prepare_park_weather_interactions(
        df,
        weather_features=['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density'],
        split_method='random_month',
        test_size=0.30,
        random_state=42,
        scaler_type='robust',
        include_roof_interactions=True
    )

    print(f"   Train samples: {len(data['X_train'])}")
    print(f"   Test samples: {len(data['X_test'])}")
    print(f"   Total features: {len(data['feature_names'])}")
    print(f"   Weather features: {data['weather_features']}")
    print(f"   Parks: {len(data['parks'])}")

    # Fit strikeouts model
    print("\n[4/5] Fitting models...")
    print("\n   --- Strikeouts Model ---")
    k_model = ParkWeatherInteractionModel(target='strikeouts', model_type='ridge')
    k_model.fit(data)

    print("\n   --- Runs Model ---")
    runs_model = ParkWeatherInteractionModel(target='runs', model_type='ridge')
    runs_model.fit(data)

    # Build dashboard params
    print("\n[5/5] Exporting dashboard parameters...")

    # Initialize params with metadata
    params = {
        '_metadata': {
            'version': '5.0.0',
            'model_type': 'park_weather_interaction_with_roof',
            'generated_at': datetime.now().strftime('%Y-%m-%d'),
            'split_method': data['split_info']['method'],
            'train_test_split': '70/30 random month units',
            'random_state': 42,
            'scaler_type': data['scaler_type'],
            'features_used': data['weather_features'] + ['is_night', 'has_roof'],
            'interaction_features': data['weather_features'],
            'has_roof_description': 'Binary indicator (1 = retractable/dome roof, 0 = open-air). Weather effects are dampened for roofed stadiums.',
            'roof_dampening_factor': 0.05,
            'roofed_stadiums': ['ARI', 'HOU', 'MIA', 'MIL', 'SEA', 'TB', 'TEX', 'TOR'],
            'new_in_v5': 'Added rhum (humidity) and air_density features that were missing in v4.',
        },
        'scaling': data['scaling_params'],
    }

    # Export strikeouts model
    params = k_model.export_to_dashboard_params(existing_params=params)

    # Export runs model
    params = runs_model.export_to_dashboard_params(existing_params=params)

    # Add interpretations
    params['_metadata']['performance_notes'] = {
        'random_month_test_r2_strikeouts': round(k_model.metrics_['test_r2'], 4),
        'random_month_test_r2_runs': round(runs_model.metrics_['test_r2'], 4),
        'interpretation': f"Model explains ~{k_model.metrics_['test_r2']*100:.0f}% of strikeout variance and ~{runs_model.metrics_['test_r2']*100:.0f}% of run variance from weather. Use for directional effects only."
    }

    # Add fixed effects interpretation
    scaling = data['scaling_params']
    params['strikeouts']['fixed_effects_interpretation'] = {
        'temp_f': f"Per 1 SD ({scaling['temp_f']['scale']:.1f}°F) increase in temp, strikeouts change by {params['strikeouts']['fixed_effects']['temp_f']:.2f}.",
        'rhum': f"Per 1 SD ({scaling['rhum']['scale']:.1f}%) increase in humidity, strikeouts change by {params['strikeouts']['fixed_effects']['rhum']:.2f}.",
        'wspd_mph': f"Per 1 SD ({scaling['wspd_mph']['scale']:.1f} mph) increase in wind speed, strikeouts change by {params['strikeouts']['fixed_effects']['wspd_mph']:.2f}.",
        'wind_cf': f"Per 1 SD ({scaling['wind_cf']['scale']:.2f}) increase in wind toward CF, strikeouts change by {params['strikeouts']['fixed_effects']['wind_cf']:.2f}.",
        'air_density': f"Per 1 SD ({scaling['air_density']['scale']:.4f} kg/m³) increase in air density, strikeouts change by {params['strikeouts']['fixed_effects']['air_density']:.2f}.",
        'is_night': f"Night games have {params['strikeouts']['fixed_effects']['is_night']:.2f} fewer/more strikeouts than day games.",
    }

    params['runs']['fixed_effects_interpretation'] = {
        'temp_f': f"Per 1 SD ({scaling['temp_f']['scale']:.1f}°F) increase in temp, runs change by {params['runs']['fixed_effects']['temp_f']:.2f}.",
        'rhum': f"Per 1 SD ({scaling['rhum']['scale']:.1f}%) increase in humidity, runs change by {params['runs']['fixed_effects']['rhum']:.2f}.",
        'wspd_mph': f"Per 1 SD ({scaling['wspd_mph']['scale']:.1f} mph) increase in wind speed, runs change by {params['runs']['fixed_effects']['wspd_mph']:.2f}.",
        'wind_cf': f"Per 1 SD ({scaling['wind_cf']['scale']:.2f}) increase in wind toward CF, runs change by {params['runs']['fixed_effects']['wind_cf']:.2f}.",
        'air_density': f"Per 1 SD ({scaling['air_density']['scale']:.4f} kg/m³) increase in air density, runs change by {params['runs']['fixed_effects']['air_density']:.2f}.",
        'is_night': f"Night games have {params['runs']['fixed_effects']['is_night']:.2f} fewer/more runs than day games.",
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\n   Dashboard params saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL V5 TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nStrikeouts Model:")
    print(f"  - Train R²: {k_model.metrics_['train_r2']:.4f}")
    print(f"  - Test R²:  {k_model.metrics_['test_r2']:.4f}")
    print(f"  - RMSE:     {k_model.metrics_['test_rmse']:.3f}")

    print(f"\nRuns Model:")
    print(f"  - Train R²: {runs_model.metrics_['train_r2']:.4f}")
    print(f"  - Test R²:  {runs_model.metrics_['test_r2']:.4f}")
    print(f"  - RMSE:     {runs_model.metrics_['test_rmse']:.3f}")

    print(f"\nFeatures in model: {params['_metadata']['features_used']}")
    print(f"\nScaling parameters:")
    for feat, scale in params['scaling'].items():
        print(f"  {feat}: center={scale['center']:.4f}, scale={scale['scale']:.4f}")

    print(f"\nStrikeouts fixed effects:")
    for feat, coef in params['strikeouts']['fixed_effects'].items():
        print(f"  {feat}: {coef:.4f}")

    print(f"\nRuns fixed effects:")
    for feat, coef in params['runs']['fixed_effects'].items():
        print(f"  {feat}: {coef:.4f}")

    return params, k_model, runs_model


if __name__ == '__main__':
    train_v5_models()
