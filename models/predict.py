"""
Prediction inference script for weather-adjusted baseball performance.

This script loads model parameters from dashboard_params.json and generates
predictions for away team strikeouts and runs based on weather conditions.

Usage:
    # From command line:
    python predict.py --park SF --temp 68 --humidity 75 --wind_speed 12 --wind_cf 0.5 --night

    # As module:
    from predict import predict_game
    result = predict_game('SF', temp_f=68, rhum=75, wspd_mph=12, wind_cf=0.5, is_night=True)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse


def load_model_params(params_path: str = None) -> Dict:
    """
    Load model parameters from dashboard_params.json.

    Parameters
    ----------
    params_path : str, optional
        Path to dashboard_params.json. If None, uses default location.

    Returns
    -------
    Dict
        Model parameters including scaling, fixed effects, and park effects.
    """
    if params_path is None:
        # Try common locations
        possible_paths = [
            Path(__file__).parent / 'dashboard_params.json',
            Path('models/dashboard_params.json'),
            Path('dashboard_params.json'),
        ]
        for p in possible_paths:
            if p.exists():
                params_path = p
                break
        if params_path is None:
            raise FileNotFoundError("Could not find dashboard_params.json")

    with open(params_path, 'r') as f:
        params = json.load(f)

    return params


def standardize_features(
    params: Dict,
    temp_f: float,
    rhum: float,
    wspd_mph: float,
    wind_cf: float,
    air_density: float = None
) -> Dict[str, float]:
    """
    Standardize raw weather features using stored scaling parameters.

    Parameters
    ----------
    params : Dict
        Model parameters with scaling info
    temp_f : float
        Temperature in Fahrenheit
    rhum : float
        Relative humidity (0-100)
    wspd_mph : float
        Wind speed in mph
    wind_cf : float
        Wind component toward center field (-1 to +1)
    air_density : float, optional
        Air density in kg/m^3. If None, computes from temp/humidity.

    Returns
    -------
    Dict[str, float]
        Standardized feature values
    """
    scaling = params['scaling']

    # Compute air density if not provided
    if air_density is None:
        # Simplified air density calculation
        temp_k = (temp_f - 32) * 5/9 + 273.15
        temp_c = (temp_f - 32) * 5/9
        e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
        e = (rhum / 100) * e_sat
        p_d = (1013.25 - e) * 100
        R_d = 287.05
        R_v = 461.5
        rho_d = p_d / (R_d * temp_k)
        rho_v = (e * 100) / (R_v * temp_k)
        air_density = rho_d + rho_v

    # Standardize using RobustScaler parameters (center=median, scale=IQR)
    standardized = {
        'temp_f': (temp_f - scaling['temp_f']['center']) / scaling['temp_f']['scale'],
        'rhum': (rhum - scaling['rhum']['center']) / scaling['rhum']['scale'],
        'wspd_mph': (wspd_mph - scaling['wspd_mph']['center']) / scaling['wspd_mph']['scale'],
        'wind_cf': (wind_cf - scaling['wind_cf']['center']) / scaling['wind_cf']['scale'],
        'air_density': (air_density - scaling['air_density']['center']) / scaling['air_density']['scale'],
    }

    return standardized


def predict_game(
    park: str,
    temp_f: float,
    rhum: float,
    wspd_mph: float,
    wind_cf: float,
    is_night: bool = False,
    air_density: float = None,
    params: Dict = None
) -> Dict[str, float]:
    """
    Predict away team strikeouts and runs for a game.

    Parameters
    ----------
    park : str
        Home team/park code (e.g., 'SF', 'LAD', 'COL')
    temp_f : float
        Temperature in Fahrenheit
    rhum : float
        Relative humidity (0-100)
    wspd_mph : float
        Wind speed in mph
    wind_cf : float
        Wind component toward center field (-1 to +1)
        Positive = blowing out, negative = blowing in
    is_night : bool
        Whether it's a night game
    air_density : float, optional
        Air density in kg/m^3. If None, computes from temp/humidity.
    params : Dict, optional
        Model parameters. If None, loads from default location.

    Returns
    -------
    Dict containing:
        - strikeouts: Predicted away team strikeouts
        - runs: Predicted away team runs
        - confidence: Model confidence indicators
        - raw_inputs: Raw input values
        - standardized_inputs: Standardized input values
    """
    if params is None:
        params = load_model_params()

    # Validate park code
    valid_parks = list(params['strikeouts']['park_effects'].keys())
    if park not in valid_parks:
        raise ValueError(f"Unknown park: {park}. Valid parks: {valid_parks}")

    # Standardize features
    std_features = standardize_features(
        params, temp_f, rhum, wspd_mph, wind_cf, air_density
    )

    # Compute predictions
    results = {}

    for target in ['strikeouts', 'runs']:
        fe = params[target]['fixed_effects']
        park_effect = params[target]['park_effects'].get(park, 0.0)

        # Prediction = Intercept + sum(coef * standardized_feature) + park_effect
        pred = fe['Intercept']
        pred += fe['temp_f'] * std_features['temp_f']
        pred += fe['rhum'] * std_features['rhum']
        pred += fe['wspd_mph'] * std_features['wspd_mph']
        pred += fe['wind_cf'] * std_features['wind_cf']
        pred += fe['air_density'] * std_features['air_density']
        pred += fe['is_night'] * (1 if is_night else 0)
        pred += park_effect

        results[target] = round(pred, 2)

    # Add model R^2 for confidence context
    results['model_confidence'] = {
        'strikeouts_r2': params['strikeouts']['model_r2']['marginal'],
        'runs_r2': params['runs']['model_r2']['marginal'],
        'note': 'Low R^2 indicates weather explains only a small portion of variance'
    }

    # Include inputs for debugging
    results['raw_inputs'] = {
        'park': park,
        'temp_f': temp_f,
        'rhum': rhum,
        'wspd_mph': wspd_mph,
        'wind_cf': wind_cf,
        'is_night': is_night
    }
    results['standardized_inputs'] = std_features

    return results


def predict_batch(
    games: list,
    params: Dict = None
) -> list:
    """
    Make predictions for multiple games.

    Parameters
    ----------
    games : list
        List of game dictionaries with keys:
        park, temp_f, rhum, wspd_mph, wind_cf, is_night (optional)
    params : Dict, optional
        Model parameters

    Returns
    -------
    list
        List of prediction results
    """
    if params is None:
        params = load_model_params()

    results = []
    for game in games:
        try:
            pred = predict_game(
                park=game['park'],
                temp_f=game['temp_f'],
                rhum=game['rhum'],
                wspd_mph=game['wspd_mph'],
                wind_cf=game['wind_cf'],
                is_night=game.get('is_night', False),
                air_density=game.get('air_density'),
                params=params
            )
            results.append({'success': True, **pred})
        except Exception as e:
            results.append({'success': False, 'error': str(e), 'input': game})

    return results


def format_prediction(result: Dict) -> str:
    """
    Format prediction result for display.

    Parameters
    ----------
    result : Dict
        Prediction result from predict_game()

    Returns
    -------
    str
        Formatted prediction string
    """
    lines = [
        "=" * 50,
        "WEATHER-ADJUSTED PREDICTION",
        "=" * 50,
        "",
        f"Park: {result['raw_inputs']['park']}",
        f"Temperature: {result['raw_inputs']['temp_f']}°F",
        f"Humidity: {result['raw_inputs']['rhum']}%",
        f"Wind Speed: {result['raw_inputs']['wspd_mph']} mph",
        f"Wind Direction: {'Blowing out' if result['raw_inputs']['wind_cf'] > 0 else 'Blowing in'} "
        f"(cf={result['raw_inputs']['wind_cf']:.2f})",
        f"Night Game: {'Yes' if result['raw_inputs']['is_night'] else 'No'}",
        "",
        "-" * 50,
        "PREDICTIONS (Away Team)",
        "-" * 50,
        f"Strikeouts: {result['strikeouts']:.1f}",
        f"Runs: {result['runs']:.1f}",
        "",
        "-" * 50,
        "MODEL CONFIDENCE",
        "-" * 50,
        f"Strikeouts Model R²: {result['model_confidence']['strikeouts_r2']:.1%}",
        f"Runs Model R²: {result['model_confidence']['runs_r2']:.1%}",
        "",
        "Note: Low R² means weather explains a small portion of variance.",
        "Team quality and other factors have larger effects.",
        "=" * 50,
    ]
    return "\n".join(lines)


def main():
    """Command-line interface for predictions."""
    parser = argparse.ArgumentParser(
        description='Predict weather-adjusted baseball performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Day game at Oracle Park with mild conditions
  python predict.py --park SF --temp 65 --humidity 70 --wind_speed 8 --wind_cf 0.0

  # Night game at Coors Field with wind blowing out
  python predict.py --park COL --temp 75 --humidity 40 --wind_speed 15 --wind_cf 0.7 --night

  # Hot day game at Chase Field (dome)
  python predict.py --park ARI --temp 105 --humidity 20 --wind_speed 0 --wind_cf 0.0
        """
    )

    parser.add_argument('--park', required=True, type=str,
                        help='Home team/park code (e.g., SF, LAD, COL, NYY)')
    parser.add_argument('--temp', required=True, type=float,
                        help='Temperature in Fahrenheit')
    parser.add_argument('--humidity', required=True, type=float,
                        help='Relative humidity (0-100)')
    parser.add_argument('--wind_speed', required=True, type=float,
                        help='Wind speed in mph')
    parser.add_argument('--wind_cf', required=True, type=float,
                        help='Wind component toward CF (-1 to +1, positive=blowing out)')
    parser.add_argument('--night', action='store_true',
                        help='Night game flag')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON instead of formatted text')
    parser.add_argument('--params', type=str, default=None,
                        help='Path to dashboard_params.json')

    args = parser.parse_args()

    try:
        result = predict_game(
            park=args.park.upper(),
            temp_f=args.temp,
            rhum=args.humidity,
            wspd_mph=args.wind_speed,
            wind_cf=args.wind_cf,
            is_night=args.night,
            params=load_model_params(args.params) if args.params else None
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_prediction(result))

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
