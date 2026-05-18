#!/usr/bin/env python3
"""
Train Model v6: Elevation and Marine Layer Features

Builds on v5 by adding:
- elevation_ft: Stadium elevation as a fixed effect
- marine_layer_day/marine_layer_night: Pacific coast fog interaction with day/night

Full feature set:
- temp_f: Air temperature (°F)
- rhum: Relative humidity (%)
- wspd_mph: Wind speed (mph)
- wind_cf: Wind component toward center field
- air_density: Computed from temp/humidity/pressure (kg/m³)
- elevation_ft: Stadium elevation (feet) - captures altitude effects beyond air density
- is_night: Night game indicator
- marine_layer_day: Pacific coast marine layer × day game
- marine_layer_night: Pacific coast marine layer × night game

Marine layer parks: SF, OAK, SD, LAA, LAD, SEA
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep import (
    load_all_team_data,
    add_enhanced_weather_features,
    prepare_park_weather_interactions,
    MARINE_LAYER_PARKS,
)
from park_weather_interaction_model import ParkWeatherInteractionModel


def train_v6_models(data_dir: Path = None, output_path: Path = None):
    """Train v6 models with elevation and marine layer features."""

    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data'
    if output_path is None:
        output_path = Path(__file__).parent / 'dashboard_params_v6.json'

    print("=" * 60)
    print("TRAINING MODEL V6 WITH ELEVATION AND MARINE LAYER")
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

    # Prepare data with v6 features: elevation + marine layer
    print("\n[3/5] Preparing park-weather interactions with elevation and marine layer...")
    data = prepare_park_weather_interactions(
        df,
        weather_features=['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density'],
        split_method='random_month',
        test_size=0.30,
        random_state=42,
        scaler_type='robust',
        include_roof_interactions=True,
        include_elevation=True,
        include_marine_layer=True,
    )

    print(f"   Train samples: {len(data['X_train'])}")
    print(f"   Test samples: {len(data['X_test'])}")
    print(f"   Total features: {len(data['feature_names'])}")
    print(f"   Weather features: {data['weather_features']}")
    print(f"   Elevation features: {data['elevation_features']}")
    print(f"   Marine layer features: {data['marine_layer_features']}")
    print(f"   Marine layer parks: {data['marine_layer_parks']}")
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

    # Get elevation data for metadata
    elevation_map = data.get('elevation_map', {})
    elevation_by_park = {park: elevation_map.get(park, 512.6) for park in data['parks']}

    # Initialize params with metadata
    params = {
        '_metadata': {
            'version': '6.0.0',
            'model_type': 'park_weather_interaction_with_elevation_marine_layer',
            'generated_at': datetime.now().strftime('%Y-%m-%d'),
            'split_method': data['split_info']['method'],
            'train_test_split': '70/30 random month units',
            'random_state': 42,
            'scaler_type': data['scaler_type'],
            'features_used': (data['weather_features'] + data['elevation_features'] +
                              ['is_night'] + data['marine_layer_features']),
            'interaction_features': data['weather_features'],
            'marine_layer_parks': MARINE_LAYER_PARKS,
            'marine_layer_description': 'Pacific coast stadiums with marine layer (fog) influence. '
                                        'Marine layer typically suppresses offense, especially at night.',
            'elevation_description': 'Stadium elevation in feet. Higher elevation = thinner air = '
                                     'balls travel farther (more runs) but may not affect strikeouts as much.',
            'elevation_by_park': elevation_by_park,
            'roofed_stadiums': ['ARI', 'HOU', 'MIA', 'MIL', 'SEA', 'TB', 'TEX', 'TOR'],
            'new_in_v6': 'Added elevation_ft as fixed effect and marine_layer × day/night interactions.',
        },
        'scaling': data['scaling_params'],
    }

    # Export strikeouts model
    params = k_model.export_to_dashboard_params(existing_params=params)

    # Export runs model
    params = runs_model.export_to_dashboard_params(existing_params=params)

    # Add performance notes
    params['_metadata']['performance_notes'] = {
        'random_month_test_r2_strikeouts': round(k_model.metrics_['test_r2'], 4),
        'random_month_test_r2_runs': round(runs_model.metrics_['test_r2'], 4),
        'interpretation': (f"Model explains ~{k_model.metrics_['test_r2']*100:.0f}% of strikeout variance "
                           f"and ~{runs_model.metrics_['test_r2']*100:.0f}% of run variance from weather. "
                           "Use for directional effects only.")
    }

    # Update features_used to include v6 features (overriding what export_to_dashboard_params set)
    params['_metadata']['features_used'] = (data['weather_features'] + data['elevation_features'] +
                                             ['is_night'] + data['marine_layer_features'])

    # Add elevation and marine layer coefficients to fixed effects
    # These are captured through the model but we need to make them explicit
    feature_names = data['feature_names']
    for target, model in [('strikeouts', k_model), ('runs', runs_model)]:
        # Add elevation coefficient
        if 'elevation_ft' in feature_names:
            idx = feature_names.index('elevation_ft')
            params[target]['fixed_effects']['elevation_ft'] = round(model.model.coef_[idx], 4)

        # Add marine layer coefficients
        for ml_feat in ['marine_layer_day', 'marine_layer_night']:
            if ml_feat in feature_names:
                idx = feature_names.index(ml_feat)
                params[target]['fixed_effects'][ml_feat] = round(model.model.coef_[idx], 4)

    # Add fixed effects interpretation
    scaling = data['scaling_params']
    params['strikeouts']['fixed_effects_interpretation'] = {
        'temp_f': f"Per 1 SD ({scaling['temp_f']['scale']:.1f}°F) increase in temp, strikeouts change by {params['strikeouts']['fixed_effects']['temp_f']:.2f}.",
        'rhum': f"Per 1 SD ({scaling['rhum']['scale']:.1f}%) increase in humidity, strikeouts change by {params['strikeouts']['fixed_effects']['rhum']:.2f}.",
        'wspd_mph': f"Per 1 SD ({scaling['wspd_mph']['scale']:.1f} mph) increase in wind speed, strikeouts change by {params['strikeouts']['fixed_effects']['wspd_mph']:.2f}.",
        'wind_cf': f"Per 1 SD ({scaling['wind_cf']['scale']:.2f}) increase in wind toward CF, strikeouts change by {params['strikeouts']['fixed_effects']['wind_cf']:.2f}.",
        'air_density': f"Per 1 SD ({scaling['air_density']['scale']:.4f} kg/m³) increase in air density, strikeouts change by {params['strikeouts']['fixed_effects']['air_density']:.2f}.",
        'elevation_ft': f"Per 1 SD ({scaling['elevation_ft']['scale']:.0f} ft) increase in elevation, strikeouts change by {params['strikeouts']['fixed_effects'].get('elevation_ft', 0):.2f}.",
        'is_night': f"Night games have {params['strikeouts']['fixed_effects']['is_night']:.2f} fewer/more strikeouts than day games.",
        'marine_layer_day': f"Pacific coast day games have {params['strikeouts']['fixed_effects'].get('marine_layer_day', 0):.2f} adjustment to strikeouts.",
        'marine_layer_night': f"Pacific coast night games have {params['strikeouts']['fixed_effects'].get('marine_layer_night', 0):.2f} adjustment to strikeouts.",
    }

    params['runs']['fixed_effects_interpretation'] = {
        'temp_f': f"Per 1 SD ({scaling['temp_f']['scale']:.1f}°F) increase in temp, runs change by {params['runs']['fixed_effects']['temp_f']:.2f}.",
        'rhum': f"Per 1 SD ({scaling['rhum']['scale']:.1f}%) increase in humidity, runs change by {params['runs']['fixed_effects']['rhum']:.2f}.",
        'wspd_mph': f"Per 1 SD ({scaling['wspd_mph']['scale']:.1f} mph) increase in wind speed, runs change by {params['runs']['fixed_effects']['wspd_mph']:.2f}.",
        'wind_cf': f"Per 1 SD ({scaling['wind_cf']['scale']:.2f}) increase in wind toward CF, runs change by {params['runs']['fixed_effects']['wind_cf']:.2f}.",
        'air_density': f"Per 1 SD ({scaling['air_density']['scale']:.4f} kg/m³) increase in air density, runs change by {params['runs']['fixed_effects']['air_density']:.2f}.",
        'elevation_ft': f"Per 1 SD ({scaling['elevation_ft']['scale']:.0f} ft) increase in elevation, runs change by {params['runs']['fixed_effects'].get('elevation_ft', 0):.2f}.",
        'is_night': f"Night games have {params['runs']['fixed_effects']['is_night']:.2f} fewer/more runs than day games.",
        'marine_layer_day': f"Pacific coast day games have {params['runs']['fixed_effects'].get('marine_layer_day', 0):.2f} adjustment to runs.",
        'marine_layer_night': f"Pacific coast night games have {params['runs']['fixed_effects'].get('marine_layer_night', 0):.2f} adjustment to runs.",
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\n   Dashboard params saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL V6 TRAINING COMPLETE")
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

    print(f"\nNew v6 Features:")
    print(f"  Elevation effect on strikeouts: {params['strikeouts']['fixed_effects'].get('elevation_ft', 'N/A')}")
    print(f"  Elevation effect on runs: {params['runs']['fixed_effects'].get('elevation_ft', 'N/A')}")
    print(f"  Marine layer day (K): {params['strikeouts']['fixed_effects'].get('marine_layer_day', 'N/A')}")
    print(f"  Marine layer night (K): {params['strikeouts']['fixed_effects'].get('marine_layer_night', 'N/A')}")
    print(f"  Marine layer day (R): {params['runs']['fixed_effects'].get('marine_layer_day', 'N/A')}")
    print(f"  Marine layer night (R): {params['runs']['fixed_effects'].get('marine_layer_night', 'N/A')}")

    print(f"\nScaling parameters:")
    for feat, scale in params['scaling'].items():
        print(f"  {feat}: center={scale['center']:.4f}, scale={scale['scale']:.4f}")

    return params, k_model, runs_model


if __name__ == '__main__':
    train_v6_models()
