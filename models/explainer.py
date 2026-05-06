"""
Model Prediction Explainer Module.

Provides interpretable explanations for weather-adjusted baseball predictions
with both absolute contributions and deviation-from-baseline views.

Usage:
    from models.explainer import PredictionExplainer

    explainer = PredictionExplainer()

    # Define game scenario
    weather = {
        'temp_f': 85,
        'rhum': 45,
        'wspd_mph': 12,
        'wind_cf': 0.8,
        'is_night': False
    }

    # Get full explanation
    result = explainer.explain_combined(
        park='SF',
        weather=weather,
        target='strikeouts',
        baseline='league'
    )

    # Print narrative
    print(result.narrative)

    # Generate visualizations
    fig_waterfall = explainer.plot_waterfall(result)
    fig_bars = explainer.plot_contributions(result)

    # For web dashboard (JSON-serializable output)
    dashboard_result = explainer.explain_for_dashboard(
        park='SF',
        temp_f=85,
        humidity=45,
        wind_speed=12,
        wind_direction_cf=0.8,
        is_night=False,
        target='strikeouts',
        baseline_type='league'
    )
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings


# =============================================================================
# CONSTANTS
# =============================================================================

# Dome parks with their roof types
DOME_PARKS = {
    'ARI': {'name': 'Chase Field', 'type': 'retractable'},
    'HOU': {'name': 'Minute Maid Park', 'type': 'retractable'},
    'MIA': {'name': 'loanDepot Park', 'type': 'retractable'},
    'MIL': {'name': 'American Family Field', 'type': 'retractable'},
    'SEA': {'name': 'T-Mobile Park', 'type': 'retractable'},
    'TB': {'name': 'Tropicana Field', 'type': 'fixed'},
    'TEX': {'name': 'Globe Life Field', 'type': 'retractable'},
    'TOR': {'name': 'Rogers Centre', 'type': 'retractable'},
}

# Park names for all MLB stadiums
PARK_NAMES = {
    'ARI': 'Chase Field',
    'ATL': 'Truist Park',
    'BAL': 'Camden Yards',
    'BOS': 'Fenway Park',
    'CHC': 'Wrigley Field',
    'CIN': 'Great American Ball Park',
    'CLE': 'Progressive Field',
    'COL': 'Coors Field',
    'CWS': 'Guaranteed Rate Field',
    'DET': 'Comerica Park',
    'HOU': 'Minute Maid Park',
    'KC': 'Kauffman Stadium',
    'LAA': 'Angel Stadium',
    'LAD': 'Dodger Stadium',
    'MIA': 'loanDepot Park',
    'MIL': 'American Family Field',
    'MIN': 'Target Field',
    'NYM': 'Citi Field',
    'NYY': 'Yankee Stadium',
    'OAK': 'Oakland Coliseum',
    'PHI': 'Citizens Bank Park',
    'PIT': 'PNC Park',
    'SD': 'Petco Park',
    'SEA': 'T-Mobile Park',
    'SF': 'Oracle Park',
    'STL': 'Busch Stadium',
    'TB': 'Tropicana Field',
    'TEX': 'Globe Life Field',
    'TOR': 'Rogers Centre',
    'WSH': 'Nationals Park',
}


@dataclass
class FeatureContribution:
    """Detailed contribution information for a single feature."""
    feature: str
    raw_value: float
    baseline_value: float
    deviation: float
    coefficient: float
    absolute_contribution: float
    deviation_contribution: float
    display_name: str = ""
    unit: str = ""

    def __post_init__(self):
        # Set display names and units if not provided
        feature_info = {
            'temp_f': ('Temperature', '°F'),
            'rhum': ('Humidity', '%'),
            'wspd_mph': ('Wind Speed', 'mph'),
            'wind_cf': ('Wind to CF', ''),
            'air_density': ('Air Density', 'kg/m³'),
            'is_night': ('Night Game', ''),
            'Intercept': ('Baseline', ''),
            'park_effect': ('Park Effect', ''),
        }
        if not self.display_name:
            self.display_name = feature_info.get(self.feature, (self.feature, ''))[0]
        if not self.unit:
            self.unit = feature_info.get(self.feature, ('', ''))[1]


@dataclass
class WaterfallBar:
    """Single bar in a waterfall chart."""
    id: str
    label: str
    start: float
    end: float
    bar_type: str  # 'baseline', 'positive', 'negative', 'total'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'id': self.id,
            'label': self.label,
            'start': round(self.start, 3),
            'end': round(self.end, 3),
            'type': self.bar_type,
        }


@dataclass
class DashboardFactor:
    """Factor contribution for dashboard display."""
    id: str
    label: str
    description: str
    contribution: float
    direction: str  # 'positive', 'negative', 'neutral'
    raw_value: Optional[float] = None
    deviation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            'id': self.id,
            'label': self.label,
            'description': self.description,
            'contribution': round(self.contribution, 3),
            'direction': self.direction,
        }
        if self.raw_value is not None:
            result['raw_value'] = round(self.raw_value, 1)
        if self.deviation is not None:
            result['deviation'] = round(self.deviation, 1)
        return result


@dataclass
class ExplanationResult:
    """Complete explanation result with both view options."""
    target: str
    park: str
    prediction: float
    baseline_prediction: float
    baseline_type: str
    contributions: List[FeatureContribution]
    narrative: str = ""
    raw_inputs: Dict = field(default_factory=dict)
    standardized_inputs: Dict = field(default_factory=dict)
    model_r2: float = 0.0

    @property
    def total_absolute(self) -> float:
        """Sum of all absolute contributions."""
        return sum(c.absolute_contribution for c in self.contributions)

    @property
    def total_deviation(self) -> float:
        """Sum of all deviation contributions."""
        return sum(c.deviation_contribution for c in self.contributions
                   if c.feature not in ['Intercept', 'park_effect'])

    def get_top_contributors(self, n: int = 3, by: str = 'deviation') -> List[FeatureContribution]:
        """Get top N contributors sorted by absolute contribution magnitude."""
        if by == 'deviation':
            # Exclude intercept and park effect for deviation view
            filtered = [c for c in self.contributions
                       if c.feature not in ['Intercept', 'park_effect']]
            return sorted(filtered, key=lambda x: abs(x.deviation_contribution), reverse=True)[:n]
        else:
            # For absolute view, include all but intercept
            filtered = [c for c in self.contributions if c.feature != 'Intercept']
            return sorted(filtered, key=lambda x: abs(x.absolute_contribution), reverse=True)[:n]


class PredictionExplainer:
    """
    Explain model predictions with multiple view options.

    Provides both absolute contribution views (coefficient × value) and
    deviation-from-baseline views (coefficient × (value - baseline)).
    """

    def __init__(
        self,
        params_path: str = None,
        baseline_path: str = None
    ):
        """
        Initialize the explainer with model parameters and baselines.

        Parameters
        ----------
        params_path : str, optional
            Path to dashboard_params.json
        baseline_path : str, optional
            Path to baseline_stats.json
        """
        self.params = self._load_params(params_path)
        self.baselines = self._load_baselines(baseline_path)

        # Cache scaling parameters for easier access
        self.scaling = self.params.get('scaling', {})

        # Features used by the model
        self.features = self.params.get('_metadata', {}).get(
            'features_used',
            ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'is_night']
        )

    def _load_params(self, params_path: str = None) -> Dict:
        """Load model parameters from JSON."""
        if params_path is None:
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
            return json.load(f)

    def _load_baselines(self, baseline_path: str = None) -> Dict:
        """Load baseline statistics from JSON."""
        if baseline_path is None:
            possible_paths = [
                Path(__file__).parent / 'baseline_stats.json',
                Path('models/baseline_stats.json'),
                Path('baseline_stats.json'),
            ]
            for p in possible_paths:
                if p.exists():
                    baseline_path = p
                    break
            if baseline_path is None:
                raise FileNotFoundError("Could not find baseline_stats.json")

        with open(baseline_path, 'r') as f:
            return json.load(f)

    def _compute_air_density(
        self,
        temp_f: float,
        rhum: float,
        pres: float = 1013.25
    ) -> float:
        """Compute air density from weather conditions."""
        temp_k = (temp_f - 32) * 5/9 + 273.15
        temp_c = (temp_f - 32) * 5/9
        e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
        e = (rhum / 100) * e_sat
        p_d = (pres - e) * 100
        R_d = 287.05
        R_v = 461.5
        rho_d = p_d / (R_d * temp_k)
        rho_v = (e * 100) / (R_v * temp_k)
        return rho_d + rho_v

    def _standardize_value(self, feature: str, value: float) -> float:
        """Standardize a single feature value using model scaling parameters."""
        if feature not in self.scaling:
            return value  # No scaling for this feature (e.g., is_night)
        center = self.scaling[feature]['center']
        scale = self.scaling[feature]['scale']
        return (value - center) / scale

    def _get_baseline_value(
        self,
        feature: str,
        baseline_type: str = 'league',
        park: str = None
    ) -> float:
        """Get baseline value for a feature."""
        if baseline_type == 'park' and park:
            park_baselines = self.baselines.get('park_specific', {}).get(park, {})
            if feature in park_baselines:
                return park_baselines[feature].get('mean', 0)

        # Fall back to league-wide
        league_baselines = self.baselines.get('league_wide', {})
        if feature in league_baselines:
            return league_baselines[feature].get('mean', 0)

        return 0

    def predict(
        self,
        park: str,
        weather: Dict,
        target: str = 'strikeouts'
    ) -> Tuple[float, Dict, Dict]:
        """
        Generate prediction for given conditions.

        Parameters
        ----------
        park : str
            Park/team code (e.g., 'SF', 'COL')
        weather : dict
            Weather conditions with keys: temp_f, rhum, wspd_mph, wind_cf, is_night
        target : str
            'strikeouts' or 'runs'

        Returns
        -------
        Tuple containing:
            - prediction: float
            - raw_inputs: Dict of raw input values
            - standardized_inputs: Dict of standardized values
        """
        if target not in self.params:
            raise ValueError(f"Unknown target: {target}. Use 'strikeouts' or 'runs'.")

        valid_parks = list(self.params[target]['park_effects'].keys())
        if park not in valid_parks:
            raise ValueError(f"Unknown park: {park}. Valid parks: {valid_parks}")

        # Extract weather values
        temp_f = weather.get('temp_f', 72)
        rhum = weather.get('rhum', 60)
        wspd_mph = weather.get('wspd_mph', 8)
        wind_cf = weather.get('wind_cf', 0)
        is_night = 1 if weather.get('is_night', False) else 0

        # Compute air density if not provided
        air_density = weather.get('air_density')
        if air_density is None:
            air_density = self._compute_air_density(temp_f, rhum)

        # Build raw inputs dict
        raw_inputs = {
            'temp_f': temp_f,
            'rhum': rhum,
            'wspd_mph': wspd_mph,
            'wind_cf': wind_cf,
            'air_density': air_density,
            'is_night': is_night,
            'park': park,
        }

        # Standardize features
        standardized = {
            'temp_f': self._standardize_value('temp_f', temp_f),
            'rhum': self._standardize_value('rhum', rhum),
            'wspd_mph': self._standardize_value('wspd_mph', wspd_mph),
            'wind_cf': self._standardize_value('wind_cf', wind_cf),
            'air_density': self._standardize_value('air_density', air_density),
        }

        # Compute prediction
        fe = self.params[target]['fixed_effects']
        park_effect = self.params[target]['park_effects'].get(park, 0.0)

        prediction = fe['Intercept']
        prediction += fe['temp_f'] * standardized['temp_f']
        prediction += fe['rhum'] * standardized['rhum']
        prediction += fe['wspd_mph'] * standardized['wspd_mph']
        prediction += fe['wind_cf'] * standardized['wind_cf']
        prediction += fe['air_density'] * standardized['air_density']
        prediction += fe['is_night'] * is_night
        prediction += park_effect

        return prediction, raw_inputs, standardized

    def explain_absolute(
        self,
        park: str,
        weather: Dict,
        target: str = 'strikeouts'
    ) -> Dict[str, float]:
        """
        Return raw coefficient × standardized_value contributions.

        This shows how much each feature contributes to the final prediction
        in an additive sense.

        Parameters
        ----------
        park : str
            Park/team code
        weather : dict
            Weather conditions
        target : str
            'strikeouts' or 'runs'

        Returns
        -------
        Dict[str, float]
            Feature contributions that sum to the prediction
        """
        prediction, raw_inputs, standardized = self.predict(park, weather, target)
        fe = self.params[target]['fixed_effects']
        park_effect = self.params[target]['park_effects'].get(park, 0.0)

        contributions = {
            'Intercept': fe['Intercept'],
            'temp_f': fe['temp_f'] * standardized['temp_f'],
            'rhum': fe['rhum'] * standardized['rhum'],
            'wspd_mph': fe['wspd_mph'] * standardized['wspd_mph'],
            'wind_cf': fe['wind_cf'] * standardized['wind_cf'],
            'air_density': fe['air_density'] * standardized['air_density'],
            'is_night': fe['is_night'] * raw_inputs['is_night'],
            'park_effect': park_effect,
        }

        return contributions

    def explain_deviation(
        self,
        park: str,
        weather: Dict,
        target: str = 'strikeouts',
        baseline: str = 'league'
    ) -> Dict[str, Dict]:
        """
        Return deviation-from-baseline contributions.

        Shows how each feature differs from baseline and its impact on prediction.

        Parameters
        ----------
        park : str
            Park/team code
        weather : dict
            Weather conditions
        target : str
            'strikeouts' or 'runs'
        baseline : str
            'league' for league-wide baseline, 'park' for park-specific

        Returns
        -------
        Dict[str, Dict]
            For each feature: deviation from baseline and contribution
        """
        prediction, raw_inputs, standardized = self.predict(park, weather, target)
        fe = self.params[target]['fixed_effects']

        # Compute baseline standardized values
        baseline_std = {}
        for feat in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']:
            baseline_raw = self._get_baseline_value(feat, baseline, park)
            baseline_std[feat] = self._standardize_value(feat, baseline_raw)

        # Baseline is_night
        baseline_is_night = self._get_baseline_value('is_night', baseline, park)

        # Compute deviation contributions
        deviations = {}
        for feat in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']:
            raw_baseline = self._get_baseline_value(feat, baseline, park)
            deviation_raw = raw_inputs[feat] - raw_baseline
            deviation_std = standardized[feat] - baseline_std[feat]
            contribution = fe[feat] * deviation_std

            deviations[feat] = {
                'raw_value': raw_inputs[feat],
                'baseline_value': raw_baseline,
                'deviation': deviation_raw,
                'standardized_deviation': deviation_std,
                'coefficient': fe[feat],
                'contribution': contribution,
            }

        # is_night deviation
        is_night_val = raw_inputs['is_night']
        is_night_deviation = is_night_val - baseline_is_night
        deviations['is_night'] = {
            'raw_value': is_night_val,
            'baseline_value': baseline_is_night,
            'deviation': is_night_deviation,
            'standardized_deviation': is_night_deviation,  # Not scaled
            'coefficient': fe['is_night'],
            'contribution': fe['is_night'] * is_night_deviation,
        }

        return deviations

    def explain_combined(
        self,
        park: str,
        weather: Dict,
        target: str = 'strikeouts',
        baseline: str = 'league'
    ) -> ExplanationResult:
        """
        Full explanation with both absolute and deviation views.

        Parameters
        ----------
        park : str
            Park/team code
        weather : dict
            Weather conditions
        target : str
            'strikeouts' or 'runs'
        baseline : str
            'league' or 'park'

        Returns
        -------
        ExplanationResult
            Complete explanation with contributions, narrative, and metadata
        """
        # Get predictions and contributions
        prediction, raw_inputs, standardized = self.predict(park, weather, target)
        absolute = self.explain_absolute(park, weather, target)
        deviation = self.explain_deviation(park, weather, target, baseline)

        # Compute baseline prediction
        baseline_weather = {}
        for feat in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf']:
            baseline_weather[feat] = self._get_baseline_value(feat, baseline, park)
        baseline_weather['is_night'] = self._get_baseline_value('is_night', baseline, park) >= 0.5
        baseline_pred, _, _ = self.predict(park, baseline_weather, target)

        # Build FeatureContribution objects
        contributions = []

        # Intercept
        contributions.append(FeatureContribution(
            feature='Intercept',
            raw_value=0,
            baseline_value=0,
            deviation=0,
            coefficient=self.params[target]['fixed_effects']['Intercept'],
            absolute_contribution=absolute['Intercept'],
            deviation_contribution=0,
        ))

        # Weather features
        for feat in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'is_night']:
            dev_info = deviation.get(feat, {})
            contributions.append(FeatureContribution(
                feature=feat,
                raw_value=dev_info.get('raw_value', 0),
                baseline_value=dev_info.get('baseline_value', 0),
                deviation=dev_info.get('deviation', 0),
                coefficient=dev_info.get('coefficient', 0),
                absolute_contribution=absolute.get(feat, 0),
                deviation_contribution=dev_info.get('contribution', 0),
            ))

        # Park effect
        park_effect = self.params[target]['park_effects'].get(park, 0.0)
        contributions.append(FeatureContribution(
            feature='park_effect',
            raw_value=park_effect,
            baseline_value=0,
            deviation=0,
            coefficient=1.0,
            absolute_contribution=park_effect,
            deviation_contribution=0,
            display_name=f'{park} Park Effect',
        ))

        # Get model R²
        model_r2 = self.params[target].get('model_r2', {}).get('marginal', 0)

        # Generate narrative
        narrative = self.generate_narrative(
            prediction=prediction,
            baseline_prediction=baseline_pred,
            contributions=contributions,
            target=target,
            park=park,
            baseline_type=baseline,
        )

        return ExplanationResult(
            target=target,
            park=park,
            prediction=round(prediction, 2),
            baseline_prediction=round(baseline_pred, 2),
            baseline_type=baseline,
            contributions=contributions,
            narrative=narrative,
            raw_inputs=raw_inputs,
            standardized_inputs=standardized,
            model_r2=model_r2,
        )

    def generate_narrative(
        self,
        prediction: float,
        baseline_prediction: float,
        contributions: List[FeatureContribution],
        target: str,
        park: str,
        baseline_type: str,
    ) -> str:
        """
        Generate plain-English explanation of the prediction.

        Parameters
        ----------
        prediction : float
            Final prediction value
        baseline_prediction : float
            Prediction at baseline conditions
        contributions : List[FeatureContribution]
            List of feature contributions
        target : str
            'strikeouts' or 'runs'
        park : str
            Park code
        baseline_type : str
            'league' or 'park'

        Returns
        -------
        str
            Formatted narrative explanation
        """
        target_short = 'K' if target == 'strikeouts' else 'R'
        diff = prediction - baseline_prediction
        diff_sign = '+' if diff >= 0 else ''

        lines = [
            f"PREDICTION: {prediction:.1f} {target} (vs {baseline_prediction:.1f} {baseline_type} baseline)",
            "",
        ]

        # Key factors section (deviation view)
        lines.append(f"KEY FACTORS (vs {baseline_type} average):")

        # Filter to weather features with meaningful deviations
        weather_contributions = [
            c for c in contributions
            if c.feature not in ['Intercept', 'park_effect']
            and abs(c.deviation_contribution) > 0.01
        ]

        # Sort by absolute deviation contribution
        weather_contributions.sort(key=lambda x: abs(x.deviation_contribution), reverse=True)

        for c in weather_contributions[:5]:
            sign = '+' if c.deviation_contribution >= 0 else ''
            dev_sign = '+' if c.deviation >= 0 else ''

            if c.feature == 'is_night':
                if c.raw_value == 1:
                    desc = "Night game"
                else:
                    desc = "Day game"
                lines.append(f"  - {desc} -> {sign}{c.deviation_contribution:.2f} {target_short}")
            elif c.unit:
                lines.append(
                    f"  - {c.display_name} {c.raw_value:.0f}{c.unit} "
                    f"({dev_sign}{c.deviation:.1f} vs avg) -> {sign}{c.deviation_contribution:.2f} {target_short}"
                )
            else:
                lines.append(
                    f"  - {c.display_name} {c.raw_value:.2f} "
                    f"({dev_sign}{c.deviation:.2f} vs avg) -> {sign}{c.deviation_contribution:.2f} {target_short}"
                )

        # Park effect
        park_contrib = next((c for c in contributions if c.feature == 'park_effect'), None)
        if park_contrib and abs(park_contrib.absolute_contribution) > 0.01:
            sign = '+' if park_contrib.absolute_contribution >= 0 else ''
            lines.append(f"  - {park} park effect -> {sign}{park_contrib.absolute_contribution:.2f} {target_short}")

        # Absolute contributions section
        lines.extend(["", "ABSOLUTE CONTRIBUTIONS:"])

        # Sort by absolute contribution
        abs_contributions = sorted(
            contributions,
            key=lambda x: abs(x.absolute_contribution),
            reverse=True
        )

        for c in abs_contributions:
            sign = '+' if c.absolute_contribution >= 0 else ''
            if c.feature == 'Intercept':
                lines.append(f"  - Baseline (intercept): {c.absolute_contribution:.2f} {target_short}")
            elif c.feature == 'park_effect':
                lines.append(f"  - {park} park effect: {sign}{c.absolute_contribution:.2f} {target_short}")
            else:
                lines.append(f"  - {c.display_name}: {sign}{c.absolute_contribution:.2f} {target_short}")

        # Total check
        total = sum(c.absolute_contribution for c in contributions)
        lines.extend([
            "",
            f"TOTAL: {total:.2f} {target_short} (should match prediction: {prediction:.1f})",
        ])

        return "\n".join(lines)

    def plot_waterfall(
        self,
        result: ExplanationResult,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Create waterfall chart showing deviations from baseline.

        Parameters
        ----------
        result : ExplanationResult
            Explanation result from explain_combined()
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            Waterfall chart figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot create visualization.")
            return None

        # Filter to features with meaningful contributions
        features = [c for c in result.contributions
                   if c.feature not in ['Intercept']
                   and (abs(c.deviation_contribution) > 0.01 or c.feature == 'park_effect')]

        # Sort by order of impact
        features.sort(key=lambda x: abs(x.deviation_contribution), reverse=True)

        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)

        # Start from baseline
        current = result.baseline_prediction
        positions = []
        widths = []
        labels = []
        colors = []

        # Add baseline bar
        positions.append(0)
        widths.append(result.baseline_prediction)
        labels.append(f'Baseline\n({result.baseline_type})')
        colors.append('#808080')  # Gray

        # Add feature contributions
        for i, feat in enumerate(features):
            contrib = feat.deviation_contribution if feat.feature != 'park_effect' else feat.absolute_contribution
            positions.append(current)
            widths.append(contrib)
            labels.append(feat.display_name)

            if contrib >= 0:
                colors.append('#2ecc71')  # Green for positive
            else:
                colors.append('#e74c3c')  # Red for negative

            current += contrib

        # Add final prediction bar
        positions.append(0)
        widths.append(result.prediction)
        labels.append('Prediction')
        colors.append('#3498db')  # Blue

        # Create horizontal bar chart
        y_positions = list(range(len(labels)))

        # Plot bars
        for i, (pos, width, color) in enumerate(zip(positions, widths, colors)):
            if i == 0:  # Baseline
                ax.barh(y_positions[i], width, left=0, color=color, alpha=0.7, edgecolor='black')
            elif i == len(labels) - 1:  # Prediction
                ax.barh(y_positions[i], width, left=0, color=color, alpha=0.7, edgecolor='black')
            else:  # Contributions
                ax.barh(y_positions[i], width, left=pos, color=color, alpha=0.7, edgecolor='black')

        # Add connecting lines
        for i in range(1, len(labels) - 1):
            prev_end = positions[i]
            ax.plot([prev_end, prev_end], [y_positions[i-1], y_positions[i]],
                   'k--', linewidth=0.5, alpha=0.5)

        # Labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        # Value annotations
        for i, (pos, width) in enumerate(zip(positions, widths)):
            if i == 0 or i == len(labels) - 1:
                ax.text(width + 0.1, y_positions[i], f'{width:.1f}',
                       va='center', fontsize=10, fontweight='bold')
            else:
                end = pos + width
                sign = '+' if width >= 0 else ''
                ax.text(end + 0.1, y_positions[i], f'{sign}{width:.2f}',
                       va='center', fontsize=9)

        # Title and labels
        target_name = 'Strikeouts' if result.target == 'strikeouts' else 'Runs'
        ax.set_title(f'{target_name} Prediction Breakdown: {result.park}', fontsize=14, fontweight='bold')
        ax.set_xlabel(target_name, fontsize=12)

        # Legend
        positive_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Increases prediction')
        negative_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Decreases prediction')
        ax.legend(handles=[positive_patch, negative_patch], loc='lower right')

        plt.tight_layout()
        return fig

    def plot_contributions(
        self,
        result: ExplanationResult,
        figsize: Tuple[int, int] = (10, 6),
        show_intercept: bool = False
    ):
        """
        Create horizontal bar chart showing absolute contributions.

        Parameters
        ----------
        result : ExplanationResult
            Explanation result from explain_combined()
        figsize : tuple
            Figure size (width, height)
        show_intercept : bool
            Whether to include intercept in chart

        Returns
        -------
        matplotlib.figure.Figure
            Bar chart figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot create visualization.")
            return None

        # Filter contributions
        contributions = [c for c in result.contributions
                        if (show_intercept or c.feature != 'Intercept')]

        # Sort by absolute contribution magnitude
        contributions.sort(key=lambda x: x.absolute_contribution)

        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)

        # Data
        labels = [c.display_name for c in contributions]
        values = [c.absolute_contribution for c in contributions]
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

        # Create bars
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            sign = '+' if val >= 0 else ''
            offset = 0.05 if val >= 0 else -0.05
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + offset, i, f'{sign}{val:.2f}', va='center', ha=ha, fontsize=9)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)

        # Add vertical line at 0
        ax.axvline(x=0, color='black', linewidth=0.5)

        # Title
        target_name = 'Strikeouts' if result.target == 'strikeouts' else 'Runs'
        ax.set_title(f'Absolute Contributions to {target_name}: {result.park}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Contribution to Predicted {target_name}', fontsize=12)

        # Add prediction annotation
        total = sum(values)
        ax.text(0.98, 0.02, f'Total: {total:.1f} {target_name}',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        result: ExplanationResult,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Create side-by-side comparison of absolute and deviation views.

        Parameters
        ----------
        result : ExplanationResult
            Explanation result
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            Combined figure with both charts
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot create visualization.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: Absolute contributions
        contributions = [c for c in result.contributions if c.feature != 'Intercept']
        contributions.sort(key=lambda x: x.absolute_contribution)

        labels = [c.display_name for c in contributions]
        values = [c.absolute_contribution for c in contributions]
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

        ax1.barh(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.set_title('Absolute Contributions', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Contribution')

        # Right: Deviation contributions
        dev_contributions = [c for c in result.contributions
                            if c.feature not in ['Intercept', 'park_effect']]
        dev_contributions.sort(key=lambda x: x.deviation_contribution)

        labels2 = [c.display_name for c in dev_contributions]
        values2 = [c.deviation_contribution for c in dev_contributions]
        colors2 = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values2]

        ax2.barh(range(len(labels2)), values2, color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(labels2)))
        ax2.set_yticklabels(labels2)
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.set_title(f'Deviation from {result.baseline_type.title()} Baseline', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Contribution')

        # Overall title
        target_name = 'Strikeouts' if result.target == 'strikeouts' else 'Runs'
        fig.suptitle(f'{target_name} Prediction: {result.prediction:.1f} at {result.park}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    # =========================================================================
    # DASHBOARD API METHODS
    # =========================================================================

    def is_dome_park(self, park: str) -> bool:
        """Check if park has a dome/retractable roof."""
        return park in DOME_PARKS

    def get_park_name(self, park: str) -> str:
        """Get full stadium name for a park code."""
        return PARK_NAMES.get(park, park)

    def _calculate_combined_wind(
        self,
        wspd_mph: float,
        wind_cf: float,
        target: str,
        park: str,
        baseline_type: str = 'league'
    ) -> DashboardFactor:
        """
        Combine wind speed and direction into a single wind factor.

        Parameters
        ----------
        wspd_mph : float
            Wind speed in mph
        wind_cf : float
            Wind component toward center field (positive = blowing out)
        target : str
            'strikeouts' or 'runs'
        park : str
            Park code
        baseline_type : str
            'league' or 'park'

        Returns
        -------
        DashboardFactor
            Combined wind factor with description
        """
        fe = self.params[target]['fixed_effects']

        # Get baseline values
        wspd_baseline = self._get_baseline_value('wspd_mph', baseline_type, park)
        wind_cf_baseline = self._get_baseline_value('wind_cf', baseline_type, park)

        # Standardize values
        wspd_std = self._standardize_value('wspd_mph', wspd_mph)
        wspd_baseline_std = self._standardize_value('wspd_mph', wspd_baseline)

        wind_cf_std = self._standardize_value('wind_cf', wind_cf)
        wind_cf_baseline_std = self._standardize_value('wind_cf', wind_cf_baseline)

        # Calculate contributions using fixed effects only (matches predict())
        wspd_contrib = fe['wspd_mph'] * (wspd_std - wspd_baseline_std)
        wind_cf_contrib = fe['wind_cf'] * (wind_cf_std - wind_cf_baseline_std)

        total_contrib = wspd_contrib + wind_cf_contrib

        # Generate description
        if wind_cf > 0:
            direction_desc = "blowing out"
        elif wind_cf < 0:
            direction_desc = "blowing in"
        else:
            direction_desc = "crosswind"

        target_unit = 'K' if target == 'strikeouts' else 'R'
        direction = 'positive' if total_contrib > 0.01 else ('negative' if total_contrib < -0.01 else 'neutral')

        return DashboardFactor(
            id='wind',
            label='Wind',
            description=f"Wind ({wspd_mph:.0f} mph, {direction_desc}) contributes {total_contrib:+.2f} {target_unit}",
            contribution=total_contrib,
            direction=direction,
            raw_value=wspd_mph,
        )

    def _calculate_temperature_factor(
        self,
        temp_f: float,
        target: str,
        park: str,
        baseline_type: str = 'league'
    ) -> DashboardFactor:
        """Calculate temperature factor."""
        fe = self.params[target]['fixed_effects']

        # Get baseline
        temp_baseline = self._get_baseline_value('temp_f', baseline_type, park)

        # Standardize
        temp_std = self._standardize_value('temp_f', temp_f)
        temp_baseline_std = self._standardize_value('temp_f', temp_baseline)

        # Calculate contribution using fixed effects only (matches predict())
        contribution = fe['temp_f'] * (temp_std - temp_baseline_std)

        deviation = temp_f - temp_baseline
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')

        sign = '+' if deviation >= 0 else ''
        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(
            id='temperature',
            label='Temperature',
            description=f"Temperature ({temp_f:.0f}°F, {sign}{deviation:.0f}° vs avg) contributes {contribution:+.2f} {target_unit}",
            contribution=contribution,
            direction=direction,
            raw_value=temp_f,
            deviation=deviation,
        )

    def _calculate_humidity_factor(
        self,
        humidity: float,
        target: str,
        park: str,
        baseline_type: str = 'league'
    ) -> DashboardFactor:
        """Calculate humidity factor."""
        fe = self.params[target]['fixed_effects']

        # Get baseline
        rhum_baseline = self._get_baseline_value('rhum', baseline_type, park)

        # Standardize
        rhum_std = self._standardize_value('rhum', humidity)
        rhum_baseline_std = self._standardize_value('rhum', rhum_baseline)

        # Calculate contribution (humidity has no park interaction in current model)
        contribution = fe['rhum'] * (rhum_std - rhum_baseline_std)

        deviation = humidity - rhum_baseline
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')

        sign = '+' if deviation >= 0 else ''
        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(
            id='humidity',
            label='Humidity',
            description=f"Humidity ({humidity:.0f}%, {sign}{deviation:.0f}% vs avg) contributes {contribution:+.2f} {target_unit}",
            contribution=contribution,
            direction=direction,
            raw_value=humidity,
            deviation=deviation,
        )

    def _calculate_day_night_factor(
        self,
        is_night: bool,
        target: str,
        park: str,
        baseline_type: str = 'league'
    ) -> DashboardFactor:
        """Calculate day/night factor."""
        fe = self.params[target]['fixed_effects']

        # Get baseline and convert to binary (matches predict() behavior)
        # Baseline is 1 (night) if proportion >= 0.5, else 0 (day)
        night_baseline_raw = self._get_baseline_value('is_night', baseline_type, park)
        night_baseline = 1 if night_baseline_raw >= 0.5 else 0

        is_night_val = 1 if is_night else 0
        contribution = fe['is_night'] * (is_night_val - night_baseline)

        game_type = "Night game" if is_night else "Day game"
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')

        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(
            id='day_night',
            label='Day/Night',
            description=f"{game_type} contributes {contribution:+.2f} {target_unit}",
            contribution=contribution,
            direction=direction,
        )

    def _calculate_park_factor(
        self,
        park: str,
        target: str
    ) -> DashboardFactor:
        """Calculate combined park effect (fixed effect only, not weather interactions)."""
        park_effect = self.params[target]['park_effects'].get(park, 0.0)

        direction = 'positive' if park_effect > 0.01 else ('negative' if park_effect < -0.01 else 'neutral')
        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(
            id='park',
            label=f'{park} Park',
            description=f"{self.get_park_name(park)} contributes {park_effect:+.2f} {target_unit}",
            contribution=park_effect,
            direction=direction,
        )

    def _calculate_air_density_factor(
        self,
        temp_f: float,
        humidity: float,
        target: str,
        park: str,
        baseline_type: str = 'league'
    ) -> DashboardFactor:
        """Calculate air density factor."""
        fe = self.params[target]['fixed_effects']

        # Compute actual air density
        air_density = self._compute_air_density(temp_f, humidity)

        # Compute baseline air density from baseline temp and humidity
        # (matches how predict() calculates the baseline)
        baseline_temp = self._get_baseline_value('temp_f', baseline_type, park)
        baseline_humidity = self._get_baseline_value('rhum', baseline_type, park)
        density_baseline = self._compute_air_density(baseline_temp, baseline_humidity)

        # Standardize
        density_std = self._standardize_value('air_density', air_density)
        density_baseline_std = self._standardize_value('air_density', density_baseline)

        # Calculate contribution
        contribution = fe['air_density'] * (density_std - density_baseline_std)

        deviation = air_density - density_baseline
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')

        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(
            id='air_density',
            label='Air Density',
            description=f"Air density ({air_density:.3f} kg/m³) contributes {contribution:+.2f} {target_unit}",
            contribution=contribution,
            direction=direction,
            raw_value=air_density,
            deviation=deviation,
        )

    def _build_waterfall_data(
        self,
        baseline: float,
        factors: List[DashboardFactor],
        prediction: float,
        baseline_label: str
    ) -> List[Dict[str, Any]]:
        """
        Build waterfall chart data structure.

        Parameters
        ----------
        baseline : float
            Starting baseline value
        factors : List[DashboardFactor]
            List of factors with contributions
        prediction : float
            Final prediction value
        baseline_label : str
            Label for baseline bar

        Returns
        -------
        List[Dict]
            List of bar data dictionaries
        """
        bars = []

        # Baseline bar
        bars.append(WaterfallBar(
            id='baseline',
            label=baseline_label,
            start=0,
            end=baseline,
            bar_type='baseline'
        ))

        # Factor bars
        current = baseline
        for factor in factors:
            if abs(factor.contribution) > 0.001:  # Skip negligible contributions
                bar_type = 'positive' if factor.contribution > 0 else 'negative'
                bars.append(WaterfallBar(
                    id=factor.id,
                    label=factor.label,
                    start=current,
                    end=current + factor.contribution,
                    bar_type=bar_type
                ))
                current += factor.contribution

        # Total bar
        bars.append(WaterfallBar(
            id='total',
            label='Prediction',
            start=0,
            end=prediction,
            bar_type='total'
        ))

        return [bar.to_dict() for bar in bars]

    def _generate_disclaimer(self, target: str) -> Dict[str, Any]:
        """
        Generate disclaimer with model R² and caveats.

        Parameters
        ----------
        target : str
            'strikeouts' or 'runs'

        Returns
        -------
        Dict
            Disclaimer data
        """
        r2 = self.params[target].get('model_r2', {}).get('marginal', 0)
        target_name = 'strikeout' if target == 'strikeouts' else 'run'

        return {
            'text': f"Weather explains approximately {r2*100:.0f}% of {target_name} variance. Team quality and pitcher matchups have larger effects.",
            'r_squared': round(r2, 3),
            'note': "Predictions are statistical estimates, not guarantees."
        }

    def _generate_summary(
        self,
        prediction: float,
        baseline: float,
        factors: List[DashboardFactor],
        target: str,
        is_dome: bool,
        dome_closed: bool = True
    ) -> Dict[str, Any]:
        """
        Generate summary headline and key drivers.

        Parameters
        ----------
        prediction : float
            Final prediction
        baseline : float
            Baseline value
        factors : List[DashboardFactor]
            Contributing factors
        target : str
            'strikeouts' or 'runs'
        is_dome : bool
            Whether this is a dome park
        dome_closed : bool
            Whether dome is closed (weather zeroed)

        Returns
        -------
        Dict
            Summary with headline and key_drivers
        """
        delta = prediction - baseline
        target_unit = 'strikeouts' if target == 'strikeouts' else 'runs'
        target_short = 'K' if target == 'strikeouts' else 'R'

        # Generate headline
        if abs(delta) < 0.1:
            headline = f"Expect roughly average {target_unit} ({prediction:.1f} {target_short})"
        elif delta > 0:
            headline = f"Expect +{delta:.1f} more {target_unit} than baseline ({prediction:.1f} {target_short})"
        else:
            headline = f"Expect {delta:.1f} fewer {target_unit} than baseline ({prediction:.1f} {target_short})"

        # Generate key drivers
        key_drivers = []

        if is_dome and dome_closed:
            key_drivers.append("Dome closed - weather effects neutralized")
        else:
            # Sort factors by absolute contribution
            sorted_factors = sorted(
                [f for f in factors if f.id not in ['park']],
                key=lambda x: abs(x.contribution),
                reverse=True
            )

            for factor in sorted_factors[:3]:
                if abs(factor.contribution) > 0.05:
                    effect = "increases" if factor.contribution > 0 else "decreases"
                    key_drivers.append(f"{factor.label} {effect} {target_unit} by {abs(factor.contribution):.2f}")

        # Add park effect if significant
        park_factor = next((f for f in factors if f.id == 'park'), None)
        if park_factor and abs(park_factor.contribution) > 0.1:
            effect = "favors more" if park_factor.contribution > 0 else "suppresses"
            key_drivers.append(f"Park historically {effect} {target_unit}")

        return {
            'headline': headline,
            'key_drivers': key_drivers if key_drivers else ["Weather conditions near league average"]
        }

    def explain_for_dashboard(
        self,
        park: str,
        temp_f: float,
        humidity: float,
        wind_speed: float,
        wind_direction_cf: float,
        is_night: bool,
        target: str = 'strikeouts',
        baseline_type: str = 'league',
        dome_closed: bool = True
    ) -> Dict[str, Any]:
        """
        Generate JSON-serializable explanation for web dashboard.

        Parameters
        ----------
        park : str
            Park/team code (e.g., 'SF', 'COL')
        temp_f : float
            Temperature in Fahrenheit
        humidity : float
            Relative humidity percentage
        wind_speed : float
            Wind speed in mph
        wind_direction_cf : float
            Wind component toward center field (positive = blowing out)
        is_night : bool
            Whether it's a night game
        target : str
            'strikeouts' or 'runs'
        baseline_type : str
            'league' for league-wide baseline, 'park' for park-specific
        dome_closed : bool
            For dome parks, whether the roof is closed (True zeros weather effects)

        Returns
        -------
        Dict
            JSON-serializable dashboard response
        """
        if target not in self.params:
            raise ValueError(f"Unknown target: {target}. Use 'strikeouts' or 'runs'.")

        valid_parks = list(self.params[target]['park_effects'].keys())
        if park not in valid_parks:
            raise ValueError(f"Unknown park: {park}. Valid parks: {valid_parks}")

        is_dome = self.is_dome_park(park)

        # Determine if we should zero out weather effects
        zero_weather = is_dome and dome_closed

        # Get baseline prediction
        baseline_weather = {
            'temp_f': self._get_baseline_value('temp_f', baseline_type, park),
            'rhum': self._get_baseline_value('rhum', baseline_type, park),
            'wspd_mph': self._get_baseline_value('wspd_mph', baseline_type, park),
            'wind_cf': self._get_baseline_value('wind_cf', baseline_type, park),
            'is_night': self._get_baseline_value('is_night', baseline_type, park) >= 0.5,
        }
        baseline_pred, _, _ = self.predict(park, baseline_weather, target)

        # Get actual prediction
        weather = {
            'temp_f': temp_f,
            'rhum': humidity,
            'wspd_mph': wind_speed,
            'wind_cf': wind_direction_cf,
            'is_night': is_night,
        }
        prediction, _, _ = self.predict(park, weather, target)

        # Calculate individual factors
        factors = []

        # Temperature
        temp_factor = self._calculate_temperature_factor(temp_f, target, park, baseline_type)
        if zero_weather:
            temp_factor.contribution = 0.0
            temp_factor.direction = 'neutral'
            temp_factor.description = f"Temperature ({temp_f:.0f}°F) - dome closed, no effect"
        factors.append(temp_factor)

        # Humidity
        humidity_factor = self._calculate_humidity_factor(humidity, target, park, baseline_type)
        if zero_weather:
            humidity_factor.contribution = 0.0
            humidity_factor.direction = 'neutral'
            humidity_factor.description = f"Humidity ({humidity:.0f}%) - dome closed, no effect"
        factors.append(humidity_factor)

        # Wind (combined)
        wind_factor = self._calculate_combined_wind(wind_speed, wind_direction_cf, target, park, baseline_type)
        if zero_weather:
            wind_factor.contribution = 0.0
            wind_factor.direction = 'neutral'
            wind_factor.description = f"Wind ({wind_speed:.0f} mph) - dome closed, no effect"
        factors.append(wind_factor)

        # Air density
        air_density_factor = self._calculate_air_density_factor(temp_f, humidity, target, park, baseline_type)
        if zero_weather:
            air_density_factor.contribution = 0.0
            air_density_factor.direction = 'neutral'
            air_density_factor.description = "Air density - dome closed, no effect"
        factors.append(air_density_factor)

        # Day/Night
        day_night_factor = self._calculate_day_night_factor(is_night, target, park, baseline_type)
        factors.append(day_night_factor)

        # Park effect
        park_factor = self._calculate_park_factor(park, target)
        factors.append(park_factor)

        # If dome is closed, adjust prediction to remove weather effects
        if zero_weather:
            # Recalculate with baseline weather but keeping day/night and park
            adjusted_weather = baseline_weather.copy()
            adjusted_weather['is_night'] = is_night
            prediction, _, _ = self.predict(park, adjusted_weather, target)

        # Build response
        target_unit = 'K' if target == 'strikeouts' else 'R'

        response = {
            'meta': {
                'version': '3.0.0',
                'target': target,
                'park': park,
                'park_name': self.get_park_name(park),
                'is_dome': is_dome,
                'dome_closed': dome_closed if is_dome else None,
                'baseline_type': baseline_type,
            },
            'prediction': {
                'value': round(prediction, 2),
                'baseline': round(baseline_pred, 2),
                'delta': round(prediction - baseline_pred, 2),
                'unit': target_unit,
            },
            'factors': [f.to_dict() for f in factors],
            'waterfall': {
                # Exclude park factor from waterfall - it's already in the baseline prediction
                'bars': self._build_waterfall_data(
                    baseline=baseline_pred,
                    factors=[f for f in factors if f.id != 'park'],
                    prediction=prediction,
                    baseline_label=f"{baseline_type.title()} Baseline"
                )
            },
            'summary': self._generate_summary(
                prediction=prediction,
                baseline=baseline_pred,
                factors=factors,
                target=target,
                is_dome=is_dome,
                dome_closed=dome_closed
            ),
            'disclaimer': self._generate_disclaimer(target),
        }

        return response


def main():
    """Demo the explainer functionality."""
    print("=" * 60)
    print("PREDICTION EXPLAINER DEMO")
    print("=" * 60)

    explainer = PredictionExplainer()

    # Example 1: Hot day game at Coors Field
    print("\n[1] HOT DAY GAME AT COORS FIELD")
    print("-" * 40)

    weather1 = {
        'temp_f': 88,
        'rhum': 35,
        'wspd_mph': 8,
        'wind_cf': 5.0,
        'is_night': False,
    }

    result1 = explainer.explain_combined(
        park='COL',
        weather=weather1,
        target='strikeouts',
        baseline='league'
    )
    print(result1.narrative)

    # Example 2: Cold night game at Oracle Park
    print("\n\n[2] COLD NIGHT GAME AT ORACLE PARK")
    print("-" * 40)

    weather2 = {
        'temp_f': 52,
        'rhum': 85,
        'wspd_mph': 15,
        'wind_cf': -8.0,  # Wind blowing in
        'is_night': True,
    }

    result2 = explainer.explain_combined(
        park='SF',
        weather=weather2,
        target='runs',
        baseline='league'
    )
    print(result2.narrative)

    # Example 3: Using park-specific baseline
    print("\n\n[3] COMPARING TO PARK-SPECIFIC BASELINE")
    print("-" * 40)

    result3 = explainer.explain_combined(
        park='SF',
        weather=weather2,
        target='runs',
        baseline='park'
    )
    print(f"Using park-specific baseline: {result3.baseline_prediction:.1f} runs")
    print(f"Prediction: {result3.prediction:.1f} runs")
    print(f"Net effect: {result3.prediction - result3.baseline_prediction:+.2f} runs")

    # Example 4: Dashboard API demo
    print("\n\n[4] DASHBOARD API (JSON OUTPUT)")
    print("-" * 40)

    dashboard_result = explainer.explain_for_dashboard(
        park='SF',
        temp_f=85,
        humidity=45,
        wind_speed=12,
        wind_direction_cf=0.8,
        is_night=False,
        target='strikeouts',
        baseline_type='league'
    )

    print(json.dumps(dashboard_result, indent=2))

    # Example 5: Dome park with roof closed
    print("\n\n[5] DOME PARK - TROPICANA FIELD (ROOF CLOSED)")
    print("-" * 40)

    dome_result = explainer.explain_for_dashboard(
        park='TB',
        temp_f=95,
        humidity=80,
        wind_speed=15,
        wind_direction_cf=10,
        is_night=True,
        target='strikeouts',
        baseline_type='league',
        dome_closed=True
    )

    print(f"Park: {dome_result['meta']['park_name']}")
    print(f"Is Dome: {dome_result['meta']['is_dome']}")
    print(f"Dome Closed: {dome_result['meta']['dome_closed']}")
    print(f"Prediction: {dome_result['prediction']['value']} K")
    print(f"Summary: {dome_result['summary']['headline']}")
    print("Key Drivers:")
    for driver in dome_result['summary']['key_drivers']:
        print(f"  - {driver}")

    # Example 6: Dome park with roof open
    print("\n\n[6] DOME PARK - MINUTE MAID PARK (ROOF OPEN)")
    print("-" * 40)

    dome_open_result = explainer.explain_for_dashboard(
        park='HOU',
        temp_f=88,
        humidity=70,
        wind_speed=8,
        wind_direction_cf=5,
        is_night=True,
        target='runs',
        baseline_type='league',
        dome_closed=False
    )

    print(f"Park: {dome_open_result['meta']['park_name']}")
    print(f"Is Dome: {dome_open_result['meta']['is_dome']}")
    print(f"Dome Closed: {dome_open_result['meta']['dome_closed']}")
    print(f"Prediction: {dome_open_result['prediction']['value']} R")
    print("\nFactors:")
    for factor in dome_open_result['factors']:
        print(f"  {factor['label']}: {factor['contribution']:+.3f} ({factor['direction']})")

    # Test visualization if matplotlib is available
    print("\n\n[7] GENERATING VISUALIZATIONS")
    print("-" * 40)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig1 = explainer.plot_waterfall(result1)
        if fig1:
            fig1.savefig('explainer_waterfall_demo.png', dpi=150, bbox_inches='tight')
            print("Saved: explainer_waterfall_demo.png")

        fig2 = explainer.plot_contributions(result1)
        if fig2:
            fig2.savefig('explainer_contributions_demo.png', dpi=150, bbox_inches='tight')
            print("Saved: explainer_contributions_demo.png")

        fig3 = explainer.plot_comparison(result1)
        if fig3:
            fig3.savefig('explainer_comparison_demo.png', dpi=150, bbox_inches='tight')
            print("Saved: explainer_comparison_demo.png")

    except ImportError:
        print("matplotlib not installed - skipping visualization demo")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
