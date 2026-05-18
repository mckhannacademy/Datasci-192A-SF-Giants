"""
Model v6 Prediction Explainer with Component Breakdown.

Extends the base explainer with:
- Component-level grouping (weather, time, park effects)
- Elevation-specific analysis
- Marine layer contribution tracking

Usage:
    from v6.explainer_v6 import PredictionExplainerV6
    explainer = PredictionExplainerV6()
    result = explainer.explain_components(park='SF', weather={...}, target='runs')
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainer import PredictionExplainer, PARK_NAMES, DOME_PARKS

# Marine layer parks - Pacific coast stadiums with fog influence
MARINE_LAYER_PARKS = ['SF', 'OAK', 'SD', 'LAA', 'LAD', 'SEA']


@dataclass
class ComponentBreakdown:
    """Grouped contribution breakdown for dashboard display."""
    weather_effects: Dict[str, float]
    time_effects: Dict[str, float]
    park_effects: Dict[str, float]
    intercept: float
    total_prediction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'weather_effects': {
                **{k: round(v, 4) for k, v in self.weather_effects.items()},
                'subtotal': round(sum(self.weather_effects.values()), 4)
            },
            'time_effects': {
                **{k: round(v, 4) for k, v in self.time_effects.items()},
                'subtotal': round(sum(self.time_effects.values()), 4)
            },
            'park_effects': {
                **{k: round(v, 4) for k, v in self.park_effects.items()},
                'subtotal': round(sum(self.park_effects.values()), 4)
            },
            'intercept': round(self.intercept, 4),
            'total_prediction': round(self.total_prediction, 4)
        }


class PredictionExplainerV6(PredictionExplainer):
    """
    Extended prediction explainer with component breakdown for v6 model.

    Adds support for:
    - Elevation as a weather effect
    - Marine layer × day/night interactions
    - Component grouping for cleaner dashboard display
    """

    def __init__(self, params_path: str = None, baseline_path: str = None):
        # Override default params path to prefer v6
        if params_path is None:
            v6_path = Path(__file__).parent / 'dashboard_params_v6.json'
            if v6_path.exists():
                params_path = str(v6_path)

        super().__init__(params_path, baseline_path)

        # Extract v6-specific metadata
        self.marine_layer_parks = self.params.get('_metadata', {}).get(
            'marine_layer_parks', MARINE_LAYER_PARKS)
        self.elevation_by_park = self.params.get('_metadata', {}).get(
            'elevation_by_park', {})

    def _get_elevation(self, park: str) -> float:
        """Get elevation for a park in feet."""
        return self.elevation_by_park.get(park, 512.6)  # League average default

    def is_marine_layer_park(self, park: str) -> bool:
        """Check if park is in the marine layer zone."""
        return park in self.marine_layer_parks

    def predict_v6(self, park: str, weather: Dict, target: str = 'strikeouts') -> tuple:
        """
        Make prediction using v6 model with elevation and marine layer.

        Returns:
            (prediction, raw_inputs, standardized_inputs, component_contributions)
        """
        if target not in self.params:
            raise ValueError(f"Unknown target: {target}. Use 'strikeouts' or 'runs'.")
        if park not in self.params[target]['park_effects']:
            raise ValueError(f"Unknown park: {park}")

        temp_f = weather.get('temp_f', 72)
        rhum = weather.get('rhum', 60)
        wspd_mph = weather.get('wspd_mph', 8)
        wind_cf = weather.get('wind_cf', 0)
        is_night = 1 if weather.get('is_night', False) else 0
        air_density = weather.get('air_density') or self._compute_air_density(temp_f, rhum)
        elevation_ft = weather.get('elevation_ft') or self._get_elevation(park)

        raw_inputs = {
            'temp_f': temp_f, 'rhum': rhum, 'wspd_mph': wspd_mph,
            'wind_cf': wind_cf, 'air_density': air_density,
            'elevation_ft': elevation_ft, 'is_night': is_night, 'park': park
        }

        # Standardize weather features
        standardized = {
            f: self._standardize_value(f, raw_inputs[f])
            for f in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'elevation_ft']
            if f in self.scaling
        }

        fe = self.params[target]['fixed_effects']

        # Calculate contributions
        contributions = {
            'intercept': fe['Intercept'],
            'temp_f': fe.get('temp_f', 0) * standardized.get('temp_f', 0),
            'rhum': fe.get('rhum', 0) * standardized.get('rhum', 0),
            'wspd_mph': fe.get('wspd_mph', 0) * standardized.get('wspd_mph', 0),
            'wind_cf': fe.get('wind_cf', 0) * standardized.get('wind_cf', 0),
            'air_density': fe.get('air_density', 0) * standardized.get('air_density', 0),
            'elevation': fe.get('elevation_ft', 0) * standardized.get('elevation_ft', 0),
            'is_night': fe.get('is_night', 0) * is_night,
            'park_effect': self.params[target]['park_effects'].get(park, 0.0),
        }

        # Marine layer effects
        is_marine = 1 if self.is_marine_layer_park(park) else 0
        is_day = 1 - is_night
        contributions['marine_layer_day'] = fe.get('marine_layer_day', 0) * is_marine * is_day
        contributions['marine_layer_night'] = fe.get('marine_layer_night', 0) * is_marine * is_night

        # Park-weather interactions
        park_weather_interactions = self.params[target].get('park_weather_interactions', {}).get(park, {})
        interaction_total = 0
        for wf in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']:
            if wf in park_weather_interactions and wf in standardized:
                interaction_total += park_weather_interactions[wf] * standardized[wf]
        contributions['park_weather_interactions'] = interaction_total

        prediction = sum(contributions.values())

        return prediction, raw_inputs, standardized, contributions

    def explain_components(self, park: str, weather: Dict, target: str = 'strikeouts') -> Dict[str, Any]:
        """
        Return prediction broken down by component groups.

        Groups:
        - weather_effects: temperature, humidity, wind, air_density, elevation
        - time_effects: day_night, marine_layer
        - park_effects: base_park_effect, park_weather_interactions

        Returns:
            Dict with 'components' key containing ComponentBreakdown
        """
        prediction, raw_inputs, standardized, contributions = self.predict_v6(park, weather, target)

        # Group contributions
        weather_effects = {
            'temperature': contributions['temp_f'],
            'humidity': contributions['rhum'],
            'wind_speed': contributions['wspd_mph'],
            'wind_direction': contributions['wind_cf'],
            'air_density': contributions['air_density'],
            'elevation': contributions['elevation'],
        }

        time_effects = {
            'day_night': contributions['is_night'],
            'marine_layer': contributions['marine_layer_day'] + contributions['marine_layer_night'],
        }

        park_effects = {
            'base_park_effect': contributions['park_effect'],
            'park_weather_interactions': contributions['park_weather_interactions'],
        }

        breakdown = ComponentBreakdown(
            weather_effects=weather_effects,
            time_effects=time_effects,
            park_effects=park_effects,
            intercept=contributions['intercept'],
            total_prediction=prediction
        )

        return {
            'target': target,
            'park': park,
            'park_name': PARK_NAMES.get(park, park),
            'components': breakdown.to_dict(),
            'raw_inputs': {k: round(v, 2) if isinstance(v, float) else v
                           for k, v in raw_inputs.items()},
            'is_marine_layer_park': self.is_marine_layer_park(park),
            'is_dome_park': self.is_dome_park(park),
        }

    def get_elevation_contribution(self, park: str, weather: Dict, target: str = 'strikeouts') -> Dict[str, Any]:
        """
        Get elevation-specific analysis.

        Returns detailed information about how elevation affects the prediction.
        """
        _, raw_inputs, standardized, contributions = self.predict_v6(park, weather, target)

        elevation_ft = raw_inputs['elevation_ft']
        league_avg_elevation = 512.6  # From elevation.csv
        elevation_deviation = elevation_ft - league_avg_elevation

        fe = self.params[target]['fixed_effects']
        elevation_coef = fe.get('elevation_ft', 0)

        return {
            'park': park,
            'park_name': PARK_NAMES.get(park, park),
            'target': target,
            'elevation_ft': elevation_ft,
            'league_avg_elevation': league_avg_elevation,
            'elevation_deviation': elevation_deviation,
            'standardized_elevation': standardized.get('elevation_ft', 0),
            'elevation_coefficient': elevation_coef,
            'elevation_contribution': contributions['elevation'],
            'interpretation': self._interpret_elevation(
                elevation_ft, contributions['elevation'], target
            ),
        }

    def _interpret_elevation(self, elevation_ft: float, contribution: float, target: str) -> str:
        """Generate human-readable interpretation of elevation effect."""
        target_unit = 'strikeouts' if target == 'strikeouts' else 'runs'

        if elevation_ft > 4000:
            location_desc = "high altitude"
        elif elevation_ft > 1000:
            location_desc = "elevated"
        elif elevation_ft < 100:
            location_desc = "near sea level"
        else:
            location_desc = "moderate elevation"

        if abs(contribution) < 0.05:
            effect_desc = f"minimal effect on {target_unit}"
        elif contribution > 0:
            effect_desc = f"adds {contribution:.2f} {target_unit}"
        else:
            effect_desc = f"reduces {target_unit} by {abs(contribution):.2f}"

        return f"At {elevation_ft:.0f} ft ({location_desc}), elevation {effect_desc}."

    def get_marine_layer_contribution(self, park: str, weather: Dict, target: str = 'strikeouts') -> Dict[str, Any]:
        """
        Get marine layer-specific analysis.

        Returns detailed information about how the marine layer affects the prediction.
        """
        _, raw_inputs, _, contributions = self.predict_v6(park, weather, target)

        is_night = raw_inputs['is_night'] == 1
        is_marine = self.is_marine_layer_park(park)

        fe = self.params[target]['fixed_effects']

        return {
            'park': park,
            'park_name': PARK_NAMES.get(park, park),
            'target': target,
            'is_marine_layer_park': is_marine,
            'is_night_game': is_night,
            'marine_layer_day_coef': fe.get('marine_layer_day', 0),
            'marine_layer_night_coef': fe.get('marine_layer_night', 0),
            'marine_layer_day_contribution': contributions['marine_layer_day'],
            'marine_layer_night_contribution': contributions['marine_layer_night'],
            'total_marine_layer_contribution': (
                contributions['marine_layer_day'] + contributions['marine_layer_night']
            ),
            'interpretation': self._interpret_marine_layer(
                park, is_night, contributions, target
            ),
        }

    def _interpret_marine_layer(self, park: str, is_night: bool, contributions: Dict, target: str) -> str:
        """Generate human-readable interpretation of marine layer effect."""
        target_unit = 'strikeouts' if target == 'strikeouts' else 'runs'

        if not self.is_marine_layer_park(park):
            return f"{PARK_NAMES.get(park, park)} is not in the marine layer zone - no fog effect."

        time_desc = "night" if is_night else "day"
        contribution = contributions['marine_layer_night'] if is_night else contributions['marine_layer_day']

        if abs(contribution) < 0.05:
            return f"Marine layer has minimal effect on {target_unit} for this {time_desc} game."
        elif contribution > 0:
            return f"Pacific coast fog conditions add {contribution:.2f} {target_unit} for this {time_desc} game."
        else:
            return f"Pacific coast fog conditions reduce {target_unit} by {abs(contribution):.2f} for this {time_desc} game."

    def explain_for_dashboard_v6(self, park: str, temp_f: float, humidity: float, wind_speed: float,
                                  wind_direction_cf: float, is_night: bool, target: str = 'strikeouts',
                                  baseline_type: str = 'league', dome_closed: bool = True) -> Dict[str, Any]:
        """
        Extended dashboard API with component breakdown.

        Adds v6-specific fields to the standard dashboard output.
        """
        # Get base dashboard output
        result = self.explain_for_dashboard(
            park=park, temp_f=temp_f, humidity=humidity, wind_speed=wind_speed,
            wind_direction_cf=wind_direction_cf, is_night=is_night, target=target,
            baseline_type=baseline_type, dome_closed=dome_closed
        )

        # Get component breakdown
        weather = {
            'temp_f': temp_f, 'rhum': humidity, 'wspd_mph': wind_speed,
            'wind_cf': wind_direction_cf, 'is_night': is_night
        }
        components = self.explain_components(park, weather, target)

        # Get elevation and marine layer details
        elevation = self.get_elevation_contribution(park, weather, target)
        marine_layer = self.get_marine_layer_contribution(park, weather, target)

        # Update result with v6 fields
        result['meta']['version'] = '6.0.0'
        result['components'] = components['components']
        result['elevation'] = {
            'elevation_ft': elevation['elevation_ft'],
            'contribution': round(elevation['elevation_contribution'], 4),
            'interpretation': elevation['interpretation'],
        }
        result['marine_layer'] = {
            'is_marine_layer_park': marine_layer['is_marine_layer_park'],
            'contribution': round(marine_layer['total_marine_layer_contribution'], 4),
            'interpretation': marine_layer['interpretation'],
        }

        return result


def main():
    """Demo the v6 explainer functionality."""
    print("=" * 60)
    print("V6 PREDICTION EXPLAINER DEMO")
    print("=" * 60)

    try:
        explainer = PredictionExplainerV6()
    except FileNotFoundError:
        print("Error: Could not find dashboard_params_v6.json")
        print("Please run train_model_v6.py first to generate model parameters.")
        return

    # Test 1: Oracle Park night game (marine layer)
    print("\n[1] Oracle Park Night Game (Marine Layer)")
    result = explainer.explain_components(
        park='SF',
        weather={'temp_f': 55, 'rhum': 85, 'wspd_mph': 12, 'wind_cf': -5, 'is_night': True},
        target='runs'
    )
    print(json.dumps(result, indent=2))

    # Test 2: Coors Field (high elevation)
    print("\n\n[2] Coors Field Hot Day (High Elevation)")
    result = explainer.explain_components(
        park='COL',
        weather={'temp_f': 92, 'rhum': 30, 'wspd_mph': 8, 'wind_cf': 8, 'is_night': False},
        target='runs'
    )
    print(json.dumps(result, indent=2))

    # Test 3: Elevation contribution detail
    print("\n\n[3] Elevation Contribution Detail - Coors vs Oracle")
    for park in ['COL', 'SF']:
        elev = explainer.get_elevation_contribution(
            park=park,
            weather={'temp_f': 75, 'rhum': 50, 'wspd_mph': 8, 'wind_cf': 0, 'is_night': False},
            target='runs'
        )
        print(f"\n{park}: {elev['interpretation']}")

    # Test 4: Marine layer comparison
    print("\n\n[4] Marine Layer Comparison - Day vs Night at Oracle")
    for is_night in [False, True]:
        ml = explainer.get_marine_layer_contribution(
            park='SF',
            weather={'temp_f': 58, 'rhum': 80, 'wspd_mph': 10, 'wind_cf': -3, 'is_night': is_night},
            target='runs'
        )
        print(f"\n{'Night' if is_night else 'Day'}: {ml['interpretation']}")

    print("\n" + "=" * 60)
    print("V6 DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
