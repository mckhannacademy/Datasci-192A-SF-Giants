#!/usr/bin/env python3
"""
Compute baseline statistics for the explainer module.

This script reads league weather data and computes:
1. League-wide baseline statistics (mean values for each feature)
2. Park-specific baseline statistics

Output is saved to models/baseline_stats.json for use by the PredictionExplainer.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add models directory to path for imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root / 'models'))

from data_prep import compute_air_density


def compute_baselines(data_path: str = None, output_path: str = None) -> dict:
    """
    Compute league-wide and park-specific baseline statistics.

    Parameters
    ----------
    data_path : str, optional
        Path to league_weather_2021_2025.csv
    output_path : str, optional
        Path to output baseline_stats.json

    Returns
    -------
    dict
        Baseline statistics dictionary
    """
    # Default paths
    if data_path is None:
        data_path = repo_root / 'data' / 'league_weather_2021_2025.csv'
    if output_path is None:
        output_path = repo_root / 'models' / 'baseline_stats.json'

    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} games")

    # Features to compute baselines for
    features = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'is_night']

    # Compute air_density if not present
    if 'air_density' not in df.columns:
        print("Computing air_density from temperature, humidity, and pressure...")
        pres = df['pres'].values if 'pres' in df.columns else None
        df['air_density'] = compute_air_density(
            df['temp_f'].values,
            df['rhum'].values,
            pres
        )

    # Compute is_night from start_hour if not present
    if 'is_night' not in df.columns and 'start_hour' in df.columns:
        # Night games typically start at 6pm or later
        df['is_night'] = (df['start_hour'] >= 18).astype(int)
    elif 'is_night' not in df.columns:
        # Default to proportion estimate
        df['is_night'] = 0.55  # Approximate league average

    # Compute league-wide baselines
    print("\nComputing league-wide baselines...")
    league_wide = {}
    for feat in features:
        if feat in df.columns:
            values = df[feat].dropna()
            league_wide[feat] = {
                'mean': round(float(values.mean()), 4),
                'median': round(float(values.median()), 4),
                'std': round(float(values.std()), 4),
                'min': round(float(values.min()), 4),
                'max': round(float(values.max()), 4),
                'q25': round(float(values.quantile(0.25)), 4),
                'q75': round(float(values.quantile(0.75)), 4),
            }
            print(f"  {feat}: mean={league_wide[feat]['mean']:.2f}, "
                  f"std={league_wide[feat]['std']:.2f}")

    # Compute park-specific baselines
    print("\nComputing park-specific baselines...")
    park_specific = {}

    # Group by home_team
    for park, park_df in df.groupby('home_team'):
        park_stats = {}
        for feat in features:
            if feat in park_df.columns:
                values = park_df[feat].dropna()
                if len(values) > 0:
                    park_stats[feat] = {
                        'mean': round(float(values.mean()), 4),
                        'median': round(float(values.median()), 4),
                        'std': round(float(values.std()), 4),
                        'n_games': int(len(values)),
                    }
        park_specific[park] = park_stats
        print(f"  {park}: {len(park_df)} games, "
              f"avg temp={park_stats.get('temp_f', {}).get('mean', 'N/A'):.1f}°F")

    # Build output structure
    baselines = {
        '_metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'source_file': str(data_path),
            'n_games': len(df),
            'features': features,
            'date_range': {
                'min': str(df['game_date'].min()),
                'max': str(df['game_date'].max()),
            }
        },
        'league_wide': league_wide,
        'park_specific': park_specific,
    }

    # Save to JSON
    print(f"\nSaving baselines to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(baselines, f, indent=2)

    print("Done!")
    return baselines


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute baseline statistics for the explainer module'
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to league weather data CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output baseline_stats.json')

    args = parser.parse_args()

    compute_baselines(args.data, args.output)


if __name__ == '__main__':
    main()
