"""
Regression models for away team performance prediction.

Enhanced with physics-based weather features:
- air_density: Affects ball flight
- heat_index: Affects player fatigue
- elevation_ft: Stadium elevation
- has_roof: Retractable roof indicator
"""

from .data_prep import (
    load_all_team_data,
    prepare_features,
    train_test_split_by_season,
    prepare_nn_data,
    prepare_mixed_effects_data,
    get_park_game_counts,
    # Feature constants
    WEATHER_FEATURES,
    WEATHER_FEATURES_BASIC,
    WEATHER_FEATURES_ENHANCED,
    WEATHER_FEATURES_FULL,
    TARGET_STRIKEOUTS,
    TARGET_RUNS,
    # Physics-based feature functions
    compute_air_density,
    compute_heat_index,
    compute_altitude_adjusted_density,
    add_enhanced_weather_features,
    load_stadium_parameters,
)

from .ridge_lasso_model import (
    ParkWeatherRegressor,
    train_all_models,
)

from .nn_embedding_model import (
    ParkEmbeddingNet,
    EmbeddingModelTrainer,
    train_nn_models,
)

from .evaluate import (
    compute_metrics,
    compare_models,
    bootstrap_confidence_interval,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_coefficient_importance,
    plot_park_embeddings_2d,
    compute_park_similarity,
    find_similar_parks,
    create_full_comparison_report,
)

from .mixed_effects_model import (
    MixedEffectsModelFitter,
    fit_mixed_effects_models,
    compare_basic_vs_enhanced,
    BASIC_WEATHER_FEATURES,
    ENHANCED_WEATHER_FEATURES,
    BASIC_RANDOM_SLOPES,
    ENHANCED_RANDOM_SLOPES,
)

from .park_effects import (
    extract_park_specific_coefficients,
    compute_weather_effect_summary,
    plot_park_random_effects,
    plot_weather_effects_heatmap,
    plot_variance_decomposition,
    compare_park_weather_sensitivity,
    create_park_effects_report,
    PARK_INFO,
)

__all__ = [
    # Data preparation
    'load_all_team_data',
    'prepare_features',
    'train_test_split_by_season',
    'prepare_nn_data',
    'prepare_mixed_effects_data',
    'get_park_game_counts',
    # Feature constants
    'WEATHER_FEATURES',
    'WEATHER_FEATURES_BASIC',
    'WEATHER_FEATURES_ENHANCED',
    'WEATHER_FEATURES_FULL',
    'TARGET_STRIKEOUTS',
    'TARGET_RUNS',
    # Physics-based feature functions
    'compute_air_density',
    'compute_heat_index',
    'compute_altitude_adjusted_density',
    'add_enhanced_weather_features',
    'load_stadium_parameters',
    # Ridge/Lasso models
    'ParkWeatherRegressor',
    'train_all_models',
    # Neural network models
    'ParkEmbeddingNet',
    'EmbeddingModelTrainer',
    'train_nn_models',
    # Evaluation
    'compute_metrics',
    'compare_models',
    'bootstrap_confidence_interval',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_coefficient_importance',
    'plot_park_embeddings_2d',
    'compute_park_similarity',
    'find_similar_parks',
    'create_full_comparison_report',
    # Mixed-effects models
    'MixedEffectsModelFitter',
    'fit_mixed_effects_models',
    'compare_basic_vs_enhanced',
    'BASIC_WEATHER_FEATURES',
    'ENHANCED_WEATHER_FEATURES',
    'BASIC_RANDOM_SLOPES',
    'ENHANCED_RANDOM_SLOPES',
    # Park effects analysis
    'extract_park_specific_coefficients',
    'compute_weather_effect_summary',
    'plot_park_random_effects',
    'plot_weather_effects_heatmap',
    'plot_variance_decomposition',
    'compare_park_weather_sensitivity',
    'create_park_effects_report',
    'PARK_INFO',
]
