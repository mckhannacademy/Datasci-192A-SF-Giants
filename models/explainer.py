"""
Model Prediction Explainer Module.

Provides interpretable explanations for weather-adjusted baseball predictions
with both absolute contributions and deviation-from-baseline views.

Usage:
    explainer = PredictionExplainer()
    result = explainer.explain_combined(park='SF', weather={'temp_f': 85, ...}, target='strikeouts')
    dashboard_data = explainer.explain_for_dashboard(park='SF', temp_f=85, ...)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

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

PARK_NAMES = {
    'ARI': 'Chase Field', 'ATL': 'Truist Park', 'BAL': 'Camden Yards',
    'BOS': 'Fenway Park', 'CHC': 'Wrigley Field', 'CIN': 'Great American Ball Park',
    'CLE': 'Progressive Field', 'COL': 'Coors Field', 'CWS': 'Guaranteed Rate Field',
    'DET': 'Comerica Park', 'HOU': 'Minute Maid Park', 'KC': 'Kauffman Stadium',
    'LAA': 'Angel Stadium', 'LAD': 'Dodger Stadium', 'MIA': 'loanDepot Park',
    'MIL': 'American Family Field', 'MIN': 'Target Field', 'NYM': 'Citi Field',
    'NYY': 'Yankee Stadium', 'OAK': 'Oakland Coliseum', 'PHI': 'Citizens Bank Park',
    'PIT': 'PNC Park', 'SD': 'Petco Park', 'SEA': 'T-Mobile Park',
    'SF': 'Oracle Park', 'STL': 'Busch Stadium', 'TB': 'Tropicana Field',
    'TEX': 'Globe Life Field', 'TOR': 'Rogers Centre', 'WSH': 'Nationals Park',
}

FEATURE_INFO = {
    'temp_f': ('Temperature', '°F'),
    'rhum': ('Humidity', '%'),
    'wspd_mph': ('Wind Speed', 'mph'),
    'wind_cf': ('Wind to CF', ''),
    'air_density': ('Air Density', 'kg/m³'),
    'is_night': ('Night Game', ''),
    'Intercept': ('Baseline', ''),
    'park_effect': ('Park Effect', ''),
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
        if not self.display_name:
            self.display_name = FEATURE_INFO.get(self.feature, (self.feature, ''))[0]
        if not self.unit:
            self.unit = FEATURE_INFO.get(self.feature, ('', ''))[1]


@dataclass
class WaterfallBar:
    """Single bar in a waterfall chart."""
    id: str
    label: str
    start: float
    end: float
    bar_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {'id': self.id, 'label': self.label, 'start': round(self.start, 3),
                'end': round(self.end, 3), 'type': self.bar_type}


@dataclass
class DashboardFactor:
    """Factor contribution for dashboard display."""
    id: str
    label: str
    description: str
    contribution: float
    direction: str
    raw_value: Optional[float] = None
    deviation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {'id': self.id, 'label': self.label, 'description': self.description,
                  'contribution': round(self.contribution, 3), 'direction': self.direction}
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
        return sum(c.absolute_contribution for c in self.contributions)

    @property
    def total_deviation(self) -> float:
        return sum(c.deviation_contribution for c in self.contributions
                   if c.feature not in ['Intercept', 'park_effect'])

    def get_top_contributors(self, n: int = 3, by: str = 'deviation') -> List[FeatureContribution]:
        if by == 'deviation':
            filtered = [c for c in self.contributions if c.feature not in ['Intercept', 'park_effect']]
            return sorted(filtered, key=lambda x: abs(x.deviation_contribution), reverse=True)[:n]
        filtered = [c for c in self.contributions if c.feature != 'Intercept']
        return sorted(filtered, key=lambda x: abs(x.absolute_contribution), reverse=True)[:n]


class PredictionExplainer:
    """
    Explain model predictions with both absolute and deviation views.

    Absolute: coefficient × value
    Deviation: coefficient × (value - baseline)
    """

    def __init__(self, params_path: str = None, baseline_path: str = None):
        self.params = self._load_params(params_path)
        self.baselines = self._load_baselines(baseline_path)
        self.scaling = self.params.get('scaling', {})
        self.features = self.params.get('_metadata', {}).get(
            'features_used', ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'is_night'])

    def _load_params(self, params_path: str = None) -> Dict:
        if params_path is None:
            # Prefer v5 (full features), then v4, then generic
            search_paths = [
                Path(__file__).parent / 'dashboard_params_v5.json',
                Path(__file__).parent / 'dashboard_params_v4.json',
                Path(__file__).parent / 'dashboard_params.json',
                Path('models/dashboard_params_v5.json'),
                Path('models/dashboard_params_v4.json'),
                Path('models/dashboard_params.json'),
            ]
            for p in search_paths:
                if p.exists():
                    params_path = p
                    break
            if params_path is None:
                raise FileNotFoundError("Could not find dashboard_params.json (v5, v4, or generic)")
        with open(params_path, 'r') as f:
            return json.load(f)

    def _load_baselines(self, baseline_path: str = None) -> Dict:
        if baseline_path is None:
            for p in [Path(__file__).parent / 'baseline_stats.json',
                      Path('models/baseline_stats.json'), Path('baseline_stats.json')]:
                if p.exists():
                    baseline_path = p
                    break
            if baseline_path is None:
                raise FileNotFoundError("Could not find baseline_stats.json")
        with open(baseline_path, 'r') as f:
            return json.load(f)

    def _compute_air_density(self, temp_f: float, rhum: float, pres: float = 1013.25) -> float:
        temp_k = (temp_f - 32) * 5/9 + 273.15
        temp_c = (temp_f - 32) * 5/9
        e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
        e = (rhum / 100) * e_sat
        p_d = (pres - e) * 100
        return p_d / (287.05 * temp_k) + (e * 100) / (461.5 * temp_k)

    def _standardize_value(self, feature: str, value: float) -> float:
        if feature not in self.scaling:
            return value
        return (value - self.scaling[feature]['center']) / self.scaling[feature]['scale']

    def _get_baseline_value(self, feature: str, baseline_type: str = 'league', park: str = None) -> float:
        if baseline_type == 'park' and park:
            park_baselines = self.baselines.get('park_specific', {}).get(park, {})
            if feature in park_baselines:
                return park_baselines[feature].get('mean', 0)
        league_baselines = self.baselines.get('league_wide', {})
        return league_baselines.get(feature, {}).get('mean', 0)

    def predict(self, park: str, weather: Dict, target: str = 'strikeouts') -> Tuple[float, Dict, Dict]:
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

        raw_inputs = {'temp_f': temp_f, 'rhum': rhum, 'wspd_mph': wspd_mph,
                      'wind_cf': wind_cf, 'air_density': air_density, 'is_night': is_night, 'park': park}

        standardized = {f: self._standardize_value(f, raw_inputs[f])
                        for f in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']}

        fe = self.params[target]['fixed_effects']
        prediction = (fe['Intercept'] + fe['temp_f'] * standardized['temp_f'] +
                      fe['rhum'] * standardized['rhum'] + fe['wspd_mph'] * standardized['wspd_mph'] +
                      fe['wind_cf'] * standardized['wind_cf'] + fe['air_density'] * standardized['air_density'] +
                      fe['is_night'] * is_night + self.params[target]['park_effects'].get(park, 0.0))

        return prediction, raw_inputs, standardized

    def explain_absolute(self, park: str, weather: Dict, target: str = 'strikeouts') -> Dict[str, float]:
        _, raw_inputs, standardized = self.predict(park, weather, target)
        fe = self.params[target]['fixed_effects']
        return {
            'Intercept': fe['Intercept'],
            'temp_f': fe['temp_f'] * standardized['temp_f'],
            'rhum': fe['rhum'] * standardized['rhum'],
            'wspd_mph': fe['wspd_mph'] * standardized['wspd_mph'],
            'wind_cf': fe['wind_cf'] * standardized['wind_cf'],
            'air_density': fe['air_density'] * standardized['air_density'],
            'is_night': fe['is_night'] * raw_inputs['is_night'],
            'park_effect': self.params[target]['park_effects'].get(park, 0.0),
        }

    def explain_deviation(self, park: str, weather: Dict, target: str = 'strikeouts',
                          baseline: str = 'league') -> Dict[str, Dict]:
        _, raw_inputs, standardized = self.predict(park, weather, target)
        fe = self.params[target]['fixed_effects']

        baseline_std = {f: self._standardize_value(f, self._get_baseline_value(f, baseline, park))
                        for f in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']}
        baseline_is_night = self._get_baseline_value('is_night', baseline, park)

        deviations = {}
        for feat in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density']:
            raw_baseline = self._get_baseline_value(feat, baseline, park)
            deviation_std = standardized[feat] - baseline_std[feat]
            deviations[feat] = {
                'raw_value': raw_inputs[feat], 'baseline_value': raw_baseline,
                'deviation': raw_inputs[feat] - raw_baseline, 'standardized_deviation': deviation_std,
                'coefficient': fe[feat], 'contribution': fe[feat] * deviation_std,
            }

        is_night_deviation = raw_inputs['is_night'] - baseline_is_night
        deviations['is_night'] = {
            'raw_value': raw_inputs['is_night'], 'baseline_value': baseline_is_night,
            'deviation': is_night_deviation, 'standardized_deviation': is_night_deviation,
            'coefficient': fe['is_night'], 'contribution': fe['is_night'] * is_night_deviation,
        }
        return deviations

    def explain_combined(self, park: str, weather: Dict, target: str = 'strikeouts',
                         baseline: str = 'league') -> ExplanationResult:
        prediction, raw_inputs, standardized = self.predict(park, weather, target)
        absolute = self.explain_absolute(park, weather, target)
        deviation = self.explain_deviation(park, weather, target, baseline)

        baseline_weather = {f: self._get_baseline_value(f, baseline, park)
                            for f in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf']}
        baseline_weather['is_night'] = self._get_baseline_value('is_night', baseline, park) >= 0.5
        baseline_pred, _, _ = self.predict(park, baseline_weather, target)

        contributions = [FeatureContribution(
            feature='Intercept', raw_value=0, baseline_value=0, deviation=0,
            coefficient=self.params[target]['fixed_effects']['Intercept'],
            absolute_contribution=absolute['Intercept'], deviation_contribution=0)]

        for feat in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'is_night']:
            dev_info = deviation.get(feat, {})
            contributions.append(FeatureContribution(
                feature=feat, raw_value=dev_info.get('raw_value', 0),
                baseline_value=dev_info.get('baseline_value', 0), deviation=dev_info.get('deviation', 0),
                coefficient=dev_info.get('coefficient', 0), absolute_contribution=absolute.get(feat, 0),
                deviation_contribution=dev_info.get('contribution', 0)))

        park_effect = self.params[target]['park_effects'].get(park, 0.0)
        contributions.append(FeatureContribution(
            feature='park_effect', raw_value=park_effect, baseline_value=0, deviation=0,
            coefficient=1.0, absolute_contribution=park_effect, deviation_contribution=0,
            display_name=f'{park} Park Effect'))

        narrative = self.generate_narrative(prediction, baseline_pred, contributions, target, park, baseline)
        return ExplanationResult(
            target=target, park=park, prediction=round(prediction, 2),
            baseline_prediction=round(baseline_pred, 2), baseline_type=baseline,
            contributions=contributions, narrative=narrative, raw_inputs=raw_inputs,
            standardized_inputs=standardized,
            model_r2=self.params[target].get('model_r2', {}).get('marginal', 0))

    def generate_narrative(self, prediction: float, baseline_prediction: float,
                           contributions: List[FeatureContribution], target: str,
                           park: str, baseline_type: str) -> str:
        target_short = 'K' if target == 'strikeouts' else 'R'
        lines = [f"PREDICTION: {prediction:.1f} {target} (vs {baseline_prediction:.1f} {baseline_type} baseline)", ""]
        lines.append(f"KEY FACTORS (vs {baseline_type} average):")

        weather_contributions = sorted(
            [c for c in contributions if c.feature not in ['Intercept', 'park_effect'] and abs(c.deviation_contribution) > 0.01],
            key=lambda x: abs(x.deviation_contribution), reverse=True)

        for c in weather_contributions[:5]:
            sign = '+' if c.deviation_contribution >= 0 else ''
            dev_sign = '+' if c.deviation >= 0 else ''
            if c.feature == 'is_night':
                desc = "Night game" if c.raw_value == 1 else "Day game"
                lines.append(f"  - {desc} -> {sign}{c.deviation_contribution:.2f} {target_short}")
            elif c.unit:
                lines.append(f"  - {c.display_name} {c.raw_value:.0f}{c.unit} ({dev_sign}{c.deviation:.1f} vs avg) -> {sign}{c.deviation_contribution:.2f} {target_short}")
            else:
                lines.append(f"  - {c.display_name} {c.raw_value:.2f} ({dev_sign}{c.deviation:.2f} vs avg) -> {sign}{c.deviation_contribution:.2f} {target_short}")

        park_contrib = next((c for c in contributions if c.feature == 'park_effect'), None)
        if park_contrib and abs(park_contrib.absolute_contribution) > 0.01:
            sign = '+' if park_contrib.absolute_contribution >= 0 else ''
            lines.append(f"  - {park} park effect -> {sign}{park_contrib.absolute_contribution:.2f} {target_short}")

        lines.extend(["", "ABSOLUTE CONTRIBUTIONS:"])
        for c in sorted(contributions, key=lambda x: abs(x.absolute_contribution), reverse=True):
            sign = '+' if c.absolute_contribution >= 0 else ''
            if c.feature == 'Intercept':
                lines.append(f"  - Baseline (intercept): {c.absolute_contribution:.2f} {target_short}")
            elif c.feature == 'park_effect':
                lines.append(f"  - {park} park effect: {sign}{c.absolute_contribution:.2f} {target_short}")
            else:
                lines.append(f"  - {c.display_name}: {sign}{c.absolute_contribution:.2f} {target_short}")

        total = sum(c.absolute_contribution for c in contributions)
        lines.extend(["", f"TOTAL: {total:.2f} {target_short} (should match prediction: {prediction:.1f})"])
        return "\n".join(lines)

    def plot_waterfall(self, result: ExplanationResult, figsize: Tuple[int, int] = (10, 6)):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            warnings.warn("matplotlib not installed.")
            return None

        features = sorted([c for c in result.contributions if c.feature != 'Intercept' and
                          (abs(c.deviation_contribution) > 0.01 or c.feature == 'park_effect')],
                         key=lambda x: abs(x.deviation_contribution), reverse=True)

        fig, ax = plt.subplots(figsize=figsize)
        current = result.baseline_prediction
        positions, widths, labels, colors = [0], [result.baseline_prediction], [f'Baseline\n({result.baseline_type})'], ['#808080']

        for feat in features:
            contrib = feat.deviation_contribution if feat.feature != 'park_effect' else feat.absolute_contribution
            positions.append(current)
            widths.append(contrib)
            labels.append(feat.display_name)
            colors.append('#2ecc71' if contrib >= 0 else '#e74c3c')
            current += contrib

        positions.append(0)
        widths.append(result.prediction)
        labels.append('Prediction')
        colors.append('#3498db')

        y_positions = list(range(len(labels)))
        for i, (pos, width, color) in enumerate(zip(positions, widths, colors)):
            left = 0 if i == 0 or i == len(labels) - 1 else pos
            ax.barh(y_positions[i], width, left=left, color=color, alpha=0.7, edgecolor='black')

        for i in range(1, len(labels) - 1):
            ax.plot([positions[i], positions[i]], [y_positions[i-1], y_positions[i]], 'k--', linewidth=0.5, alpha=0.5)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        for i, (pos, width) in enumerate(zip(positions, widths)):
            if i == 0 or i == len(labels) - 1:
                ax.text(width + 0.1, y_positions[i], f'{width:.1f}', va='center', fontsize=10, fontweight='bold')
            else:
                ax.text(pos + width + 0.1, y_positions[i], f'{("+" if width >= 0 else "")}{width:.2f}', va='center', fontsize=9)

        target_name = 'Strikeouts' if result.target == 'strikeouts' else 'Runs'
        ax.set_title(f'{target_name} Prediction Breakdown: {result.park}', fontsize=14, fontweight='bold')
        ax.set_xlabel(target_name, fontsize=12)
        ax.legend(handles=[mpatches.Patch(color='#2ecc71', alpha=0.7, label='Increases prediction'),
                          mpatches.Patch(color='#e74c3c', alpha=0.7, label='Decreases prediction')], loc='lower right')
        plt.tight_layout()
        return fig

    def plot_contributions(self, result: ExplanationResult, figsize: Tuple[int, int] = (10, 6),
                           show_intercept: bool = False):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed.")
            return None

        contributions = sorted([c for c in result.contributions if show_intercept or c.feature != 'Intercept'],
                               key=lambda x: x.absolute_contribution)
        fig, ax = plt.subplots(figsize=figsize)
        labels = [c.display_name for c in contributions]
        values = [c.absolute_contribution for c in contributions]
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
        for i, (bar, val) in enumerate(zip(bars, values)):
            sign = '+' if val >= 0 else ''
            ax.text(val + (0.05 if val >= 0 else -0.05), i, f'{sign}{val:.2f}',
                   va='center', ha='left' if val >= 0 else 'right', fontsize=9)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(x=0, color='black', linewidth=0.5)

        target_name = 'Strikeouts' if result.target == 'strikeouts' else 'Runs'
        ax.set_title(f'Absolute Contributions to {target_name}: {result.park}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Contribution to Predicted {target_name}', fontsize=12)
        ax.text(0.98, 0.02, f'Total: {sum(values):.1f} {target_name}', transform=ax.transAxes,
               ha='right', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        return fig

    def plot_comparison(self, result: ExplanationResult, figsize: Tuple[int, int] = (14, 6)):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for ax, data, title in [
            (ax1, [(c.display_name, c.absolute_contribution) for c in result.contributions if c.feature != 'Intercept'], 'Absolute Contributions'),
            (ax2, [(c.display_name, c.deviation_contribution) for c in result.contributions if c.feature not in ['Intercept', 'park_effect']], f'Deviation from {result.baseline_type.title()} Baseline')
        ]:
            data = sorted(data, key=lambda x: x[1])
            labels, values = zip(*data) if data else ([], [])
            colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
            ax.barh(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Contribution')

        target_name = 'Strikeouts' if result.target == 'strikeouts' else 'Runs'
        fig.suptitle(f'{target_name} Prediction: {result.prediction:.1f} at {result.park}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def is_dome_park(self, park: str) -> bool:
        return park in DOME_PARKS

    def get_park_name(self, park: str) -> str:
        return PARK_NAMES.get(park, park)

    def _calculate_combined_wind(self, wspd_mph: float, wind_cf: float, target: str,
                                 park: str, baseline_type: str = 'league') -> DashboardFactor:
        fe = self.params[target]['fixed_effects']
        wspd_baseline = self._get_baseline_value('wspd_mph', baseline_type, park)
        wind_cf_baseline = self._get_baseline_value('wind_cf', baseline_type, park)

        wspd_contrib = fe['wspd_mph'] * (self._standardize_value('wspd_mph', wspd_mph) -
                                          self._standardize_value('wspd_mph', wspd_baseline))
        wind_cf_contrib = fe['wind_cf'] * (self._standardize_value('wind_cf', wind_cf) -
                                            self._standardize_value('wind_cf', wind_cf_baseline))
        total_contrib = wspd_contrib + wind_cf_contrib

        direction_desc = "blowing out" if wind_cf > 0 else ("blowing in" if wind_cf < 0 else "crosswind")
        target_unit = 'K' if target == 'strikeouts' else 'R'
        direction = 'positive' if total_contrib > 0.01 else ('negative' if total_contrib < -0.01 else 'neutral')

        return DashboardFactor(id='wind', label='Wind',
            description=f"Wind ({wspd_mph:.0f} mph, {direction_desc}) contributes {total_contrib:+.2f} {target_unit}",
            contribution=total_contrib, direction=direction, raw_value=wspd_mph)

    def _calculate_temperature_factor(self, temp_f: float, target: str, park: str,
                                       baseline_type: str = 'league') -> DashboardFactor:
        fe = self.params[target]['fixed_effects']
        temp_baseline = self._get_baseline_value('temp_f', baseline_type, park)
        contribution = fe['temp_f'] * (self._standardize_value('temp_f', temp_f) -
                                        self._standardize_value('temp_f', temp_baseline))
        deviation = temp_f - temp_baseline
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')
        target_unit = 'K' if target == 'strikeouts' else 'R'
        sign = '+' if deviation >= 0 else ''

        return DashboardFactor(id='temperature', label='Temperature',
            description=f"Temperature ({temp_f:.0f}°F, {sign}{deviation:.0f}° vs avg) contributes {contribution:+.2f} {target_unit}",
            contribution=contribution, direction=direction, raw_value=temp_f, deviation=deviation)

    def _calculate_humidity_factor(self, humidity: float, target: str, park: str,
                                    baseline_type: str = 'league') -> DashboardFactor:
        fe = self.params[target]['fixed_effects']
        rhum_baseline = self._get_baseline_value('rhum', baseline_type, park)
        contribution = fe['rhum'] * (self._standardize_value('rhum', humidity) -
                                      self._standardize_value('rhum', rhum_baseline))
        deviation = humidity - rhum_baseline
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')
        target_unit = 'K' if target == 'strikeouts' else 'R'
        sign = '+' if deviation >= 0 else ''

        return DashboardFactor(id='humidity', label='Humidity',
            description=f"Humidity ({humidity:.0f}%, {sign}{deviation:.0f}% vs avg) contributes {contribution:+.2f} {target_unit}",
            contribution=contribution, direction=direction, raw_value=humidity, deviation=deviation)

    def _calculate_day_night_factor(self, is_night: bool, target: str, park: str,
                                     baseline_type: str = 'league') -> DashboardFactor:
        fe = self.params[target]['fixed_effects']
        night_baseline = 1 if self._get_baseline_value('is_night', baseline_type, park) >= 0.5 else 0
        contribution = fe['is_night'] * ((1 if is_night else 0) - night_baseline)
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')
        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(id='day_night', label='Day/Night',
            description=f"{'Night game' if is_night else 'Day game'} contributes {contribution:+.2f} {target_unit}",
            contribution=contribution, direction=direction)

    def _calculate_park_factor(self, park: str, target: str) -> DashboardFactor:
        park_effect = self.params[target]['park_effects'].get(park, 0.0)
        direction = 'positive' if park_effect > 0.01 else ('negative' if park_effect < -0.01 else 'neutral')
        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(id='park', label=f'{park} Park',
            description=f"{self.get_park_name(park)} contributes {park_effect:+.2f} {target_unit}",
            contribution=park_effect, direction=direction)

    def _calculate_air_density_factor(self, temp_f: float, humidity: float, target: str,
                                       park: str, baseline_type: str = 'league') -> DashboardFactor:
        fe = self.params[target]['fixed_effects']
        air_density = self._compute_air_density(temp_f, humidity)
        baseline_temp = self._get_baseline_value('temp_f', baseline_type, park)
        baseline_humidity = self._get_baseline_value('rhum', baseline_type, park)
        density_baseline = self._compute_air_density(baseline_temp, baseline_humidity)

        contribution = fe['air_density'] * (self._standardize_value('air_density', air_density) -
                                             self._standardize_value('air_density', density_baseline))
        direction = 'positive' if contribution > 0.01 else ('negative' if contribution < -0.01 else 'neutral')
        target_unit = 'K' if target == 'strikeouts' else 'R'

        return DashboardFactor(id='air_density', label='Air Density',
            description=f"Air density ({air_density:.3f} kg/m³) contributes {contribution:+.2f} {target_unit}",
            contribution=contribution, direction=direction, raw_value=air_density, deviation=air_density - density_baseline)

    def _build_waterfall_data(self, baseline: float, factors: List[DashboardFactor],
                               prediction: float, baseline_label: str) -> List[Dict[str, Any]]:
        bars = [WaterfallBar(id='baseline', label=baseline_label, start=0, end=baseline, bar_type='baseline')]
        current = baseline
        for factor in factors:
            if abs(factor.contribution) > 0.001:
                bars.append(WaterfallBar(id=factor.id, label=factor.label, start=current,
                    end=current + factor.contribution, bar_type='positive' if factor.contribution > 0 else 'negative'))
                current += factor.contribution
        bars.append(WaterfallBar(id='total', label='Prediction', start=0, end=prediction, bar_type='total'))
        return [bar.to_dict() for bar in bars]

    def _generate_disclaimer(self, target: str) -> Dict[str, Any]:
        r2 = self.params[target].get('model_r2', {}).get('marginal', 0)
        target_name = 'strikeout' if target == 'strikeouts' else 'run'
        return {'text': f"Weather explains approximately {r2*100:.0f}% of {target_name} variance. Team quality and pitcher matchups have larger effects.",
                'r_squared': round(r2, 3), 'note': "Predictions are statistical estimates, not guarantees."}

    def _generate_summary(self, prediction: float, baseline: float, factors: List[DashboardFactor],
                          target: str, is_dome: bool, dome_closed: bool = True) -> Dict[str, Any]:
        delta = prediction - baseline
        target_unit = 'strikeouts' if target == 'strikeouts' else 'runs'
        target_short = 'K' if target == 'strikeouts' else 'R'

        if abs(delta) < 0.1:
            headline = f"Expect roughly average {target_unit} ({prediction:.1f} {target_short})"
        elif delta > 0:
            headline = f"Expect +{delta:.1f} more {target_unit} than baseline ({prediction:.1f} {target_short})"
        else:
            headline = f"Expect {delta:.1f} fewer {target_unit} than baseline ({prediction:.1f} {target_short})"

        key_drivers = []
        if is_dome and dome_closed:
            key_drivers.append("Dome closed - weather effects neutralized")
        else:
            for factor in sorted([f for f in factors if f.id != 'park'], key=lambda x: abs(x.contribution), reverse=True)[:3]:
                if abs(factor.contribution) > 0.05:
                    effect = "increases" if factor.contribution > 0 else "decreases"
                    key_drivers.append(f"{factor.label} {effect} {target_unit} by {abs(factor.contribution):.2f}")

        park_factor = next((f for f in factors if f.id == 'park'), None)
        if park_factor and abs(park_factor.contribution) > 0.1:
            key_drivers.append(f"Park historically {'favors more' if park_factor.contribution > 0 else 'suppresses'} {target_unit}")

        return {'headline': headline, 'key_drivers': key_drivers or ["Weather conditions near league average"]}

    def explain_for_dashboard(self, park: str, temp_f: float, humidity: float, wind_speed: float,
                               wind_direction_cf: float, is_night: bool, target: str = 'strikeouts',
                               baseline_type: str = 'league', dome_closed: bool = True) -> Dict[str, Any]:
        if target not in self.params:
            raise ValueError(f"Unknown target: {target}. Use 'strikeouts' or 'runs'.")
        if park not in self.params[target]['park_effects']:
            raise ValueError(f"Unknown park: {park}")

        is_dome = self.is_dome_park(park)
        zero_weather = is_dome and dome_closed

        baseline_weather = {f: self._get_baseline_value(f, baseline_type, park)
                            for f in ['temp_f', 'rhum', 'wspd_mph', 'wind_cf']}
        baseline_weather['is_night'] = self._get_baseline_value('is_night', baseline_type, park) >= 0.5
        baseline_pred, _, _ = self.predict(park, baseline_weather, target)

        weather = {'temp_f': temp_f, 'rhum': humidity, 'wspd_mph': wind_speed,
                   'wind_cf': wind_direction_cf, 'is_night': is_night}
        prediction, _, _ = self.predict(park, weather, target)

        factors = [
            self._calculate_temperature_factor(temp_f, target, park, baseline_type),
            self._calculate_humidity_factor(humidity, target, park, baseline_type),
            self._calculate_combined_wind(wind_speed, wind_direction_cf, target, park, baseline_type),
            self._calculate_air_density_factor(temp_f, humidity, target, park, baseline_type),
            self._calculate_day_night_factor(is_night, target, park, baseline_type),
            self._calculate_park_factor(park, target),
        ]

        if zero_weather:
            for f in factors[:4]:  # Temperature, humidity, wind, air density
                f.contribution = 0.0
                f.direction = 'neutral'
                f.description = f.description.split(')')[0] + ') - dome closed, no effect'
            adjusted_weather = baseline_weather.copy()
            adjusted_weather['is_night'] = is_night
            prediction, _, _ = self.predict(park, adjusted_weather, target)

        target_unit = 'K' if target == 'strikeouts' else 'R'
        return {
            'meta': {'version': '3.0.0', 'target': target, 'park': park, 'park_name': self.get_park_name(park),
                     'is_dome': is_dome, 'dome_closed': dome_closed if is_dome else None, 'baseline_type': baseline_type},
            'prediction': {'value': round(prediction, 2), 'baseline': round(baseline_pred, 2),
                           'delta': round(prediction - baseline_pred, 2), 'unit': target_unit},
            'factors': [f.to_dict() for f in factors],
            'waterfall': {'bars': self._build_waterfall_data(baseline_pred, [f for f in factors if f.id != 'park'],
                                                              prediction, f"{baseline_type.title()} Baseline")},
            'summary': self._generate_summary(prediction, baseline_pred, factors, target, is_dome, dome_closed),
            'disclaimer': self._generate_disclaimer(target),
        }


def main():
    """Demo the explainer functionality."""
    print("=" * 60)
    print("PREDICTION EXPLAINER DEMO")
    print("=" * 60)

    explainer = PredictionExplainer()

    # Hot day at Coors
    result1 = explainer.explain_combined(park='COL',
        weather={'temp_f': 88, 'rhum': 35, 'wspd_mph': 8, 'wind_cf': 5.0, 'is_night': False},
        target='strikeouts', baseline='league')
    print("\n[1] HOT DAY AT COORS FIELD")
    print(result1.narrative)

    # Cold night at Oracle
    result2 = explainer.explain_combined(park='SF',
        weather={'temp_f': 52, 'rhum': 85, 'wspd_mph': 15, 'wind_cf': -8.0, 'is_night': True},
        target='runs', baseline='league')
    print("\n\n[2] COLD NIGHT AT ORACLE PARK")
    print(result2.narrative)

    # Dashboard API
    dashboard_result = explainer.explain_for_dashboard(
        park='SF', temp_f=85, humidity=45, wind_speed=12, wind_direction_cf=0.8,
        is_night=False, target='strikeouts', baseline_type='league')
    print("\n\n[3] DASHBOARD API OUTPUT")
    print(json.dumps(dashboard_result, indent=2))

    # Dome park
    dome_result = explainer.explain_for_dashboard(
        park='TB', temp_f=95, humidity=80, wind_speed=15, wind_direction_cf=10,
        is_night=True, target='strikeouts', baseline_type='league', dome_closed=True)
    print(f"\n\n[4] DOME PARK (TROPICANA)")
    print(f"Prediction: {dome_result['prediction']['value']} K")
    print(f"Summary: {dome_result['summary']['headline']}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
