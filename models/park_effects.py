"""
Park-specific effect extraction and visualization for mixed-effects models.

This module provides tools to:
1. Extract Best Linear Unbiased Predictors (BLUPs) for each park
2. Compute park-specific weather coefficients
3. Generate visualizations of park effects
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .mixed_effects_model import MixedEffectsModelFitter


# Park metadata for enhanced visualizations
PARK_INFO = {
    'SF': {'name': 'Oracle Park', 'city': 'San Francisco', 'region': 'West Coast'},
    'COL': {'name': 'Coors Field', 'city': 'Denver', 'region': 'Mountain'},
    'BOS': {'name': 'Fenway Park', 'city': 'Boston', 'region': 'Northeast'},
    'NYY': {'name': 'Yankee Stadium', 'city': 'New York', 'region': 'Northeast'},
    'LAD': {'name': 'Dodger Stadium', 'city': 'Los Angeles', 'region': 'West Coast'},
    'CHC': {'name': 'Wrigley Field', 'city': 'Chicago', 'region': 'Midwest'},
    'SEA': {'name': 'T-Mobile Park', 'city': 'Seattle', 'region': 'West Coast'},
    'MIA': {'name': 'LoanDepot Park', 'city': 'Miami', 'region': 'Southeast'},
    'HOU': {'name': 'Minute Maid Park', 'city': 'Houston', 'region': 'South'},
    'TEX': {'name': 'Globe Life Field', 'city': 'Arlington', 'region': 'South'},
    'ARI': {'name': 'Chase Field', 'city': 'Phoenix', 'region': 'Southwest'},
    'SD': {'name': 'Petco Park', 'city': 'San Diego', 'region': 'West Coast'},
    'OAK': {'name': 'Oakland Coliseum', 'city': 'Oakland', 'region': 'West Coast'},
    'LAA': {'name': 'Angel Stadium', 'city': 'Anaheim', 'region': 'West Coast'},
    'MIN': {'name': 'Target Field', 'city': 'Minneapolis', 'region': 'Midwest'},
    'DET': {'name': 'Comerica Park', 'city': 'Detroit', 'region': 'Midwest'},
    'CLE': {'name': 'Progressive Field', 'city': 'Cleveland', 'region': 'Midwest'},
    'CWS': {'name': 'Guaranteed Rate Field', 'city': 'Chicago', 'region': 'Midwest'},
    'KC': {'name': 'Kauffman Stadium', 'city': 'Kansas City', 'region': 'Midwest'},
    'STL': {'name': 'Busch Stadium', 'city': 'St. Louis', 'region': 'Midwest'},
    'MIL': {'name': 'American Family Field', 'city': 'Milwaukee', 'region': 'Midwest'},
    'CIN': {'name': 'Great American Ball Park', 'city': 'Cincinnati', 'region': 'Midwest'},
    'PIT': {'name': 'PNC Park', 'city': 'Pittsburgh', 'region': 'Northeast'},
    'ATL': {'name': 'Truist Park', 'city': 'Atlanta', 'region': 'Southeast'},
    'PHI': {'name': 'Citizens Bank Park', 'city': 'Philadelphia', 'region': 'Northeast'},
    'NYM': {'name': 'Citi Field', 'city': 'New York', 'region': 'Northeast'},
    'WSH': {'name': 'Nationals Park', 'city': 'Washington', 'region': 'Northeast'},
    'BAL': {'name': 'Camden Yards', 'city': 'Baltimore', 'region': 'Northeast'},
    'TB': {'name': 'Tropicana Field', 'city': 'St. Petersburg', 'region': 'Southeast'},
    'TOR': {'name': 'Rogers Centre', 'city': 'Toronto', 'region': 'Northeast'},
}


def extract_park_specific_coefficients(
    fitter: MixedEffectsModelFitter,
    model_name: str = 'park_slopes'
) -> pd.DataFrame:
    """
    Compute park-specific weather coefficients.

    For random slope models, each park's coefficient is:
    β_park = β_fixed + b_park

    where β_fixed is the population average and b_park is the park's deviation.

    Parameters
    ----------
    fitter : MixedEffectsModelFitter
        Fitted model fitter object
    model_name : str
        Which model to extract from

    Returns
    -------
    pd.DataFrame
        Park-specific coefficients for each weather feature
    """
    # Get fixed effects
    fe_df = fitter.get_fixed_effects(model_name)
    fixed_effects = dict(zip(fe_df['Parameter'], fe_df['Coefficient']))

    # Get random effects
    re_df = fitter.get_random_effects(model_name)

    # Build park-specific coefficients
    rows = []
    for _, row in re_df.iterrows():
        park = row['Park']
        park_row = {'Park': park}

        # For each effect in the random effects
        for col in re_df.columns:
            if col == 'Park':
                continue

            # Get fixed effect for this parameter
            param_name = col if col != 'Group' else 'Intercept'
            fixed_val = fixed_effects.get(param_name, 0)

            # Park-specific = fixed + random
            park_row[f'{param_name}_fixed'] = fixed_val
            park_row[f'{param_name}_random'] = row[col]
            park_row[f'{param_name}_total'] = fixed_val + row[col]

        rows.append(park_row)

    return pd.DataFrame(rows).sort_values('Park')


def compute_weather_effect_summary(
    fitter: MixedEffectsModelFitter,
    model_name: str = 'park_slopes'
) -> pd.DataFrame:
    """
    Create summary table of weather effects across parks.

    Parameters
    ----------
    fitter : MixedEffectsModelFitter
        Fitted model fitter
    model_name : str
        Model to use

    Returns
    -------
    pd.DataFrame
        Summary with park, effect estimates, and confidence indicators
    """
    # Get random effects for parks
    re_df = fitter.get_random_effects(model_name)

    # Get fixed effects for context
    fe_df = fitter.get_fixed_effects(model_name)
    fixed_effects = dict(zip(fe_df['Parameter'], fe_df['Coefficient']))

    # Identify weather-related columns in random effects
    weather_cols = [c for c in re_df.columns if c != 'Park' and c != 'Group']

    rows = []
    for _, row in re_df.iterrows():
        park = row['Park']
        park_data = {
            'Park': park,
            'Park_Name': PARK_INFO.get(park, {}).get('name', park),
            'Region': PARK_INFO.get(park, {}).get('region', 'Unknown'),
        }

        for col in weather_cols:
            # Clean column name for display
            clean_name = col.replace('Group', 'Intercept')

            # Random effect value
            re_val = row[col]

            # Fixed effect value
            fe_val = fixed_effects.get(clean_name, 0)

            # Total effect
            total = fe_val + re_val

            park_data[f'{clean_name}_effect'] = total
            park_data[f'{clean_name}_deviation'] = re_val

        rows.append(park_data)

    result = pd.DataFrame(rows)

    return result.sort_values('Park')


def bootstrap_park_effects(
    fitter: MixedEffectsModelFitter,
    df_train: pd.DataFrame,
    n_bootstrap: int = 100,
    model_name: str = 'park_intercept'
) -> Dict[str, pd.DataFrame]:
    """
    Bootstrap confidence intervals for park random effects.

    Note: This is computationally intensive. For production use,
    consider profile likelihood or parametric bootstrap.

    Parameters
    ----------
    fitter : MixedEffectsModelFitter
        Original fitted model
    df_train : pd.DataFrame
        Training data
    n_bootstrap : int
        Number of bootstrap samples
    model_name : str
        Model to bootstrap (simpler models recommended)

    Returns
    -------
    Dict with 'estimates' and 'ci_lower', 'ci_upper' DataFrames
    """
    from .mixed_effects_model import MixedEffectsModelFitter

    # Collect bootstrap estimates
    bootstrap_effects = []

    print(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{n_bootstrap}")

        # Resample with replacement
        boot_df = df_train.sample(n=len(df_train), replace=True)

        # Fit model
        boot_fitter = MixedEffectsModelFitter(
            target=fitter.target,
            weather_features=fitter.weather_features
        )

        try:
            # Fit just the requested model
            if model_name == 'park_intercept':
                boot_fitter.models[model_name] = boot_fitter._fit_park_intercept_model(
                    ' + '.join(fitter.weather_features) + ' + is_night',
                    'lbfgs', 100
                )
            else:
                boot_fitter.fit_model_hierarchy(boot_df)

            re_df = boot_fitter.get_random_effects(model_name)
            bootstrap_effects.append(re_df)
        except Exception:
            continue  # Skip failed fits

    if len(bootstrap_effects) < 10:
        raise ValueError("Too few successful bootstrap fits")

    # Combine and compute CIs
    # Stack all bootstrap results
    all_parks = bootstrap_effects[0]['Park'].values
    effect_cols = [c for c in bootstrap_effects[0].columns if c != 'Park']

    ci_results = {'Park': all_parks}

    for col in effect_cols:
        values = np.array([[df.loc[df['Park'] == p, col].values[0]
                          for p in all_parks]
                          for df in bootstrap_effects])

        ci_results[f'{col}_mean'] = np.mean(values, axis=0)
        ci_results[f'{col}_lower'] = np.percentile(values, 2.5, axis=0)
        ci_results[f'{col}_upper'] = np.percentile(values, 97.5, axis=0)

    return pd.DataFrame(ci_results)


def plot_park_random_effects(
    fitter: MixedEffectsModelFitter,
    model_name: str = 'park_intercept',
    effect_col: str = 'Intercept',
    title: str = None,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Create caterpillar plot of park random effects.

    Parameters
    ----------
    fitter : MixedEffectsModelFitter
        Fitted model
    model_name : str
        Model to visualize
    effect_col : str
        Which effect column to plot (default: 'Intercept')
    title : str
        Plot title
    ax : plt.Axes
        Existing axes to plot on

    Returns
    -------
    plt.Axes
    """
    re_df = fitter.get_random_effects(model_name)

    # Handle column name - try Intercept first, then Group
    if effect_col not in re_df.columns:
        if 'Intercept' in re_df.columns:
            effect_col = 'Intercept'
        elif 'Group' in re_df.columns:
            effect_col = 'Group'
        else:
            # Use first numeric column
            numeric_cols = re_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                effect_col = numeric_cols[0]
            else:
                raise ValueError(f"No suitable effect column found in {re_df.columns.tolist()}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))

    # Sort by effect value
    re_df = re_df.sort_values(effect_col).reset_index(drop=True)

    # Plot
    y_pos = np.arange(len(re_df))
    colors = ['orange' if p == 'SF' else 'green' if p == 'COL' else 'steelblue'
              for p in re_df['Park']]
    ax.barh(y_pos, re_df[effect_col], color=colors, alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)

    # Labels with park names
    labels = [f"{p} ({PARK_INFO.get(p, {}).get('name', p)[:20]})"
              for p in re_df['Park']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Random Effect (Deviation from Average)')
    ax.set_title(title or f'Park Random Effects: {effect_col}')

    return ax


def plot_weather_effects_heatmap(
    fitter: MixedEffectsModelFitter,
    model_name: str = 'park_slopes',
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create heatmap of park-specific weather effects.

    Parameters
    ----------
    fitter : MixedEffectsModelFitter
        Fitted model
    model_name : str
        Model to visualize
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    # Get park-specific coefficients
    coef_df = extract_park_specific_coefficients(fitter, model_name)

    # Extract total effect columns
    total_cols = [c for c in coef_df.columns if c.endswith('_total')]

    if not total_cols:
        # Fallback to random effects only
        re_df = fitter.get_random_effects(model_name)
        effect_cols = [c for c in re_df.columns if c != 'Park']
        plot_df = re_df.set_index('Park')[effect_cols]
    else:
        # Use total effects
        plot_df = coef_df.set_index('Park')[total_cols]
        plot_df.columns = [c.replace('_total', '') for c in plot_df.columns]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        plot_df,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        ax=ax,
        cbar_kws={'label': 'Effect Size'}
    )

    ax.set_title(f'Park-Specific Weather Effects ({fitter.target})')
    ax.set_xlabel('Weather Feature')
    ax.set_ylabel('Park')

    plt.tight_layout()
    return fig


def plot_variance_decomposition(
    fitter: MixedEffectsModelFitter,
    model_name: str = 'park_slopes',
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Create pie/bar chart of variance decomposition.

    Parameters
    ----------
    fitter : MixedEffectsModelFitter
        Fitted model
    model_name : str
        Model to visualize
    ax : plt.Axes
        Existing axes

    Returns
    -------
    plt.Axes
    """
    vc_df = fitter.get_variance_components(model_name)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(vc_df)))
    bars = ax.barh(vc_df['Source'], vc_df['Pct_of_Total'], color=colors)

    # Add percentage labels
    for bar, pct in zip(bars, vc_df['Pct_of_Total']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center')

    ax.set_xlabel('Percentage of Total Variance')
    ax.set_title(f'Variance Decomposition: {fitter.target}')
    ax.set_xlim(0, 100)

    return ax


def plot_park_weather_heatmap(
    model: 'ParkWeatherInteractionModel' = None,
    coef_df: pd.DataFrame = None,
    weather_features: List[str] = None,
    target: str = 'strikeouts',
    show_interactions_only: bool = False,
    figsize: Tuple[int, int] = (12, 14),
    cmap: str = 'RdBu_r',
    annot_fmt: str = '.3f'
) -> plt.Figure:
    """
    Create heatmap of park-specific weather sensitivities.

    Shows how each park responds to different weather features, making it
    easy to identify parks with unusual weather sensitivity patterns.

    Parameters
    ----------
    model : ParkWeatherInteractionModel, optional
        Fitted interaction model. If provided, extracts coefficients automatically.
    coef_df : pd.DataFrame, optional
        Pre-computed coefficient DataFrame (alternative to model).
        Must have parks as index and weather features as columns.
    weather_features : List[str], optional
        Weather features to include. If None, uses ['temp_f', 'wspd_mph', 'wind_cf'].
    target : str
        Target variable name (for title)
    show_interactions_only : bool
        If True, show only interaction effects (park deviation from average).
        If False, show total effects (fixed + interaction).
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    annot_fmt : str
        Format string for annotations

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if weather_features is None:
        weather_features = ['temp_f', 'wspd_mph', 'wind_cf']

    # Get coefficient data
    if model is not None:
        if show_interactions_only:
            coef_df = model.get_interaction_coefficients()
        else:
            coef_df = model.get_park_weather_coefficients()
        coef_df = coef_df[weather_features]
    elif coef_df is None:
        raise ValueError("Must provide either model or coef_df")
    else:
        # Filter to requested weather features
        coef_df = coef_df[[c for c in weather_features if c in coef_df.columns]]

    # Sort parks by total sensitivity for better visualization
    coef_df['_total'] = coef_df.abs().sum(axis=1)
    coef_df = coef_df.sort_values('_total', ascending=False).drop('_total', axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        coef_df,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=annot_fmt,
        ax=ax,
        cbar_kws={'label': 'Coefficient'},
        linewidths=0.5
    )

    # Labels
    effect_type = "Interaction Effects" if show_interactions_only else "Total Effects"
    ax.set_title(f'Park Weather Sensitivities: {target.capitalize()} ({effect_type})')
    ax.set_xlabel('Weather Feature')
    ax.set_ylabel('Park')

    # Add park names to y-axis labels
    labels = [f"{p} ({PARK_INFO.get(p, {}).get('name', p)[:15]})"
              for p in coef_df.index]
    ax.set_yticklabels(labels, fontsize=9)

    plt.tight_layout()
    return fig


def summarize_park_sensitivities(
    model: 'ParkWeatherInteractionModel' = None,
    coef_df: pd.DataFrame = None,
    weather_features: List[str] = None,
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Rank parks by their weather sensitivity.

    Returns rankings for:
    - Overall sensitivity (sum of |coefficients|)
    - Each individual weather feature
    - High/low sensitivity parks for each feature

    Parameters
    ----------
    model : ParkWeatherInteractionModel, optional
        Fitted interaction model
    coef_df : pd.DataFrame, optional
        Pre-computed total effect DataFrame
    weather_features : List[str], optional
        Weather features to analyze
    top_n : int
        Number of top/bottom parks to highlight

    Returns
    -------
    Dict with:
        - 'overall': Overall sensitivity ranking
        - 'by_feature': Dict with rankings per feature
        - 'summary': Text summary of key findings
    """
    if weather_features is None:
        weather_features = ['temp_f', 'wspd_mph', 'wind_cf']

    if model is not None:
        coef_df = model.get_park_weather_coefficients()
        coef_df = coef_df[weather_features]
    elif coef_df is None:
        raise ValueError("Must provide either model or coef_df")

    results = {}

    # Overall sensitivity
    overall = coef_df.abs().sum(axis=1).sort_values(ascending=False)
    results['overall'] = pd.DataFrame({
        'Park': overall.index,
        'Total_Sensitivity': overall.values,
        'Rank': range(1, len(overall) + 1)
    })

    # By feature
    results['by_feature'] = {}
    for feat in weather_features:
        if feat in coef_df.columns:
            feat_vals = coef_df[feat].sort_values(ascending=False)
            results['by_feature'][feat] = pd.DataFrame({
                'Park': feat_vals.index,
                f'{feat}_effect': feat_vals.values,
                'Rank': range(1, len(feat_vals) + 1)
            })

    # Summary text
    summary_lines = [
        "PARK WEATHER SENSITIVITY SUMMARY",
        "=" * 40,
        "",
        f"Top {top_n} Most Weather-Sensitive Parks (Overall):",
    ]
    for i, (park, sensitivity) in enumerate(zip(overall.index[:top_n], overall.values[:top_n])):
        name = PARK_INFO.get(park, {}).get('name', park)
        summary_lines.append(f"  {i+1}. {park} ({name}): {sensitivity:.3f}")

    summary_lines.extend([
        "",
        f"Bottom {top_n} Least Weather-Sensitive Parks:",
    ])
    bottom = overall.sort_values(ascending=True)[:top_n]
    for i, (park, sensitivity) in enumerate(zip(bottom.index, bottom.values)):
        name = PARK_INFO.get(park, {}).get('name', park)
        summary_lines.append(f"  {i+1}. {park} ({name}): {sensitivity:.3f}")

    summary_lines.extend([
        "",
        "Feature-Specific Insights:",
    ])

    for feat in weather_features:
        if feat in results['by_feature']:
            feat_df = results['by_feature'][feat]
            high_park = feat_df.iloc[0]['Park']
            high_val = feat_df.iloc[0][f'{feat}_effect']
            low_park = feat_df.iloc[-1]['Park']
            low_val = feat_df.iloc[-1][f'{feat}_effect']
            summary_lines.append(f"  {feat}:")
            summary_lines.append(f"    Highest: {high_park} ({high_val:+.3f})")
            summary_lines.append(f"    Lowest:  {low_park} ({low_val:+.3f})")

    results['summary'] = "\n".join(summary_lines)

    return results


def compare_park_weather_sensitivity(
    strikeouts_fitter: MixedEffectsModelFitter,
    runs_fitter: MixedEffectsModelFitter,
    model_name: str = 'park_slopes'
) -> pd.DataFrame:
    """
    Compare how parks respond to weather for different outcomes.

    Parameters
    ----------
    strikeouts_fitter : MixedEffectsModelFitter
        Strikeouts model
    runs_fitter : MixedEffectsModelFitter
        Runs model
    model_name : str
        Model to compare

    Returns
    -------
    pd.DataFrame
        Comparison of park sensitivities
    """
    k_re = strikeouts_fitter.get_random_effects(model_name)
    runs_re = runs_fitter.get_random_effects(model_name)

    # Merge on Park
    comparison = k_re.merge(
        runs_re,
        on='Park',
        suffixes=('_strikeouts', '_runs')
    )

    # Add park metadata
    comparison['Park_Name'] = comparison['Park'].map(
        lambda x: PARK_INFO.get(x, {}).get('name', x)
    )
    comparison['Region'] = comparison['Park'].map(
        lambda x: PARK_INFO.get(x, {}).get('region', 'Unknown')
    )

    return comparison


def create_park_effects_report(
    strikeouts_fitter: MixedEffectsModelFitter,
    runs_fitter: MixedEffectsModelFitter,
    output_dir: str = None,
    model_name: str = 'park_intercept'
) -> Dict[str, any]:
    """
    Generate comprehensive park effects report with visualizations.

    Parameters
    ----------
    strikeouts_fitter : MixedEffectsModelFitter
        Strikeouts model
    runs_fitter : MixedEffectsModelFitter
        Runs model
    output_dir : str, optional
        Directory to save figures
    model_name : str
        Model to report on

    Returns
    -------
    Dict with DataFrames and figure objects
    """
    from pathlib import Path

    results = {}

    # 1. Variance decomposition
    print("Computing variance decomposition...")
    results['k_variance'] = strikeouts_fitter.get_variance_components(model_name)
    results['runs_variance'] = runs_fitter.get_variance_components(model_name)

    # 2. Random effects
    print("Extracting random effects...")
    results['k_random_effects'] = strikeouts_fitter.get_random_effects(model_name)
    results['runs_random_effects'] = runs_fitter.get_random_effects(model_name)

    # 3. Fixed effects
    print("Extracting fixed effects...")
    results['k_fixed_effects'] = strikeouts_fitter.get_fixed_effects(model_name)
    results['runs_fixed_effects'] = runs_fitter.get_fixed_effects(model_name)

    # 4. R² metrics
    print("Computing R² metrics...")
    results['k_r2'] = strikeouts_fitter.compute_r_squared(model_name)
    results['runs_r2'] = runs_fitter.compute_r_squared(model_name)

    # 5. Create visualizations
    print("Creating visualizations...")

    # Variance decomposition figure
    fig_var, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_variance_decomposition(strikeouts_fitter, model_name, axes[0])
    axes[0].set_title('Variance Decomposition: Strikeouts')
    plot_variance_decomposition(runs_fitter, model_name, axes[1])
    axes[1].set_title('Variance Decomposition: Runs')
    plt.tight_layout()
    results['fig_variance'] = fig_var

    # Random effects caterpillar plots
    fig_re, axes = plt.subplots(1, 2, figsize=(14, 12))
    plot_park_random_effects(strikeouts_fitter, model_name, 'Intercept', 'Park Effects: Strikeouts', axes[0])
    plot_park_random_effects(runs_fitter, model_name, 'Intercept', 'Park Effects: Runs', axes[1])
    plt.tight_layout()
    results['fig_random_effects'] = fig_re

    # Save figures if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig_var.savefig(output_path / 'variance_decomposition.png', dpi=150, bbox_inches='tight')
        fig_re.savefig(output_path / 'park_random_effects.png', dpi=150, bbox_inches='tight')
        print(f"Figures saved to {output_path}")

    return results


def create_park_weather_interaction_report(
    k_model: 'ParkWeatherInteractionModel',
    runs_model: 'ParkWeatherInteractionModel',
    output_dir: str = None,
    show_plots: bool = True
) -> Dict[str, any]:
    """
    Generate comprehensive park-weather interaction report.

    This function creates visualizations and summaries for the
    park × weather interaction models, highlighting:
    - Which parks are most/least weather sensitive
    - Park-specific responses to each weather feature
    - Comparison between strikeouts and runs sensitivities

    Parameters
    ----------
    k_model : ParkWeatherInteractionModel
        Fitted strikeouts model
    runs_model : ParkWeatherInteractionModel
        Fitted runs model
    output_dir : str, optional
        Directory to save figures and CSVs
    show_plots : bool
        Whether to display plots interactively

    Returns
    -------
    Dict with:
        - 'k_sensitivities': Strikeouts sensitivity rankings
        - 'runs_sensitivities': Runs sensitivity rankings
        - 'k_interactions': Strikeouts interaction coefficients
        - 'runs_interactions': Runs interaction coefficients
        - 'figures': Dict of matplotlib figures
    """
    from pathlib import Path

    results = {}

    # Get sensitivity summaries
    print("Analyzing strikeouts model...")
    k_sens = summarize_park_sensitivities(k_model)
    results['k_sensitivities'] = k_sens
    print(k_sens['summary'])

    print("\n" + "=" * 60 + "\n")

    print("Analyzing runs model...")
    runs_sens = summarize_park_sensitivities(runs_model)
    results['runs_sensitivities'] = runs_sens
    print(runs_sens['summary'])

    # Get interaction coefficients
    results['k_interactions'] = k_model.get_interaction_coefficients()
    results['runs_interactions'] = runs_model.get_interaction_coefficients()

    # Create figures
    figures = {}

    # Heatmap for strikeouts
    print("\nCreating visualizations...")
    fig_k = plot_park_weather_heatmap(
        model=k_model,
        target='strikeouts',
        show_interactions_only=False
    )
    figures['k_heatmap'] = fig_k

    # Heatmap for runs
    fig_runs = plot_park_weather_heatmap(
        model=runs_model,
        target='runs',
        show_interactions_only=False
    )
    figures['runs_heatmap'] = fig_runs

    # Side-by-side comparison of key parks
    key_parks = ['SF', 'COL', 'CHC', 'TB', 'MIA', 'ARI']
    fig_compare, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (model, target) in zip(axes, [(k_model, 'Strikeouts'), (runs_model, 'Runs')]):
        coef_df = model.get_park_weather_coefficients()
        key_df = coef_df.loc[[p for p in key_parks if p in coef_df.index]]

        x = np.arange(len(key_df.index))
        width = 0.25

        for i, feat in enumerate(['temp_f', 'wspd_mph', 'wind_cf']):
            ax.bar(x + i*width, key_df[feat], width, label=feat)

        ax.set_xticks(x + width)
        ax.set_xticklabels(key_df.index)
        ax.set_ylabel('Total Weather Effect')
        ax.set_title(f'{target}: Key Park Comparison')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    figures['key_parks_comparison'] = fig_compare

    results['figures'] = figures

    # Save outputs if directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save figures
        fig_k.savefig(output_path / 'strikeouts_park_weather_heatmap.png', dpi=150, bbox_inches='tight')
        fig_runs.savefig(output_path / 'runs_park_weather_heatmap.png', dpi=150, bbox_inches='tight')
        fig_compare.savefig(output_path / 'key_parks_comparison.png', dpi=150, bbox_inches='tight')

        # Save CSVs
        results['k_interactions'].to_csv(output_path / 'strikeouts_park_interactions.csv')
        results['runs_interactions'].to_csv(output_path / 'runs_park_interactions.csv')
        k_sens['overall'].to_csv(output_path / 'strikeouts_sensitivity_ranking.csv', index=False)
        runs_sens['overall'].to_csv(output_path / 'runs_sensitivity_ranking.csv', index=False)

        # Save summary text
        with open(output_path / 'sensitivity_summary.txt', 'w') as f:
            f.write("STRIKEOUTS MODEL\n")
            f.write("=" * 60 + "\n")
            f.write(k_sens['summary'])
            f.write("\n\n")
            f.write("RUNS MODEL\n")
            f.write("=" * 60 + "\n")
            f.write(runs_sens['summary'])

        print(f"\nOutputs saved to {output_path}")

    if show_plots:
        plt.show()

    return results


if __name__ == '__main__':
    from pathlib import Path
    from data_prep import load_all_team_data, prepare_mixed_effects_data
    from mixed_effects_model import fit_mixed_effects_models

    # Load and prepare data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\nPreparing mixed-effects data...")
    me_data = prepare_mixed_effects_data(df)

    print("\nFitting mixed-effects models...")
    k_fitter, runs_fitter = fit_mixed_effects_models(me_data)

    print("\n" + "="*60)
    print("PARK EFFECTS ANALYSIS")
    print("="*60)

    # Use park_intercept model (more stable)
    model_name = 'park_intercept'

    # Variance decomposition
    print("\nVariance Decomposition (Strikeouts):")
    print(k_fitter.get_variance_components(model_name))

    print("\nVariance Decomposition (Runs):")
    print(runs_fitter.get_variance_components(model_name))

    # Random effects
    print("\nPark Random Effects (Strikeouts):")
    k_re = k_fitter.get_random_effects(model_name)
    k_re_sorted = k_re.sort_values('Group', ascending=False)
    print(k_re_sorted.head(10))

    # Create report
    print("\nGenerating park effects report...")
    output_dir = script_dir.parent / 'analysis' / 'model_outputs' / 'mixed_effects'
    report = create_park_effects_report(k_fitter, runs_fitter, str(output_dir), model_name)

    print("\nReport complete!")
    print(f"\nStrikeouts R² - Marginal: {report['k_r2']['marginal_r2']:.4f}, Conditional: {report['k_r2']['conditional_r2']:.4f}")
    print(f"Runs R² - Marginal: {report['runs_r2']['marginal_r2']:.4f}, Conditional: {report['runs_r2']['conditional_r2']:.4f}")
