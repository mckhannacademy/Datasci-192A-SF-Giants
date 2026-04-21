"""
Ridge and Lasso regression models for away team strikeouts and runs prediction.
Uses one-hot encoding for parks with weather × park interaction terms.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings


class ParkWeatherRegressor:
    """
    Ridge/Lasso regression model with one-hot park encoding and weather interactions.
    """

    def __init__(
        self,
        model_type: str = 'ridge',
        alphas: Optional[np.ndarray] = None,
        cv: int = 5
    ):
        """
        Initialize the regressor.

        Parameters
        ----------
        model_type : str
            'ridge' for L2 regularization, 'lasso' for L1
        alphas : np.ndarray, optional
            Array of alpha values to try in cross-validation
        cv : int
            Number of cross-validation folds
        """
        self.model_type = model_type.lower()
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 50)
        self.cv = cv
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> 'ParkWeatherRegressor':
        """
        Fit the model with cross-validation to select best alpha.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target variable
        feature_names : list, optional
            Names of features for interpretation

        Returns
        -------
        self
        """
        self.feature_names = feature_names

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model with cross-validation
        if self.model_type == 'ridge':
            self.model = RidgeCV(alphas=self.alphas, cv=self.cv)
        elif self.model_type == 'lasso':
            self.model = LassoCV(alphas=self.alphas, cv=self.cv, max_iter=10000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        print(f"{self.model_type.capitalize()} fitted with alpha = {self.model.alpha_:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True target values

        Returns
        -------
        Dict with RMSE, MAE, and R² metrics
        """
        y_pred = self.predict(X)

        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature names.

        Returns
        -------
        pd.DataFrame
            Coefficients sorted by absolute value
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        coefs = self.model.coef_
        names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(coefs))]

        df = pd.DataFrame({
            'feature': names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs)
        })

        return df.sort_values('abs_coefficient', ascending=False)

    def get_weather_main_effects(self) -> pd.DataFrame:
        """
        Extract the main weather effect coefficients (not interactions).

        Returns
        -------
        pd.DataFrame
            Main weather effect coefficients
        """
        coefs = self.get_coefficients()
        # Filter to non-interaction weather features
        weather_features = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'wind_lcf', 'wind_rcf', 'is_night']
        main_effects = coefs[coefs['feature'].isin(weather_features)]
        return main_effects

    def get_park_effects(self) -> pd.DataFrame:
        """
        Extract the park-specific coefficients.

        Returns
        -------
        pd.DataFrame
            Park main effect coefficients
        """
        coefs = self.get_coefficients()
        park_effects = coefs[coefs['feature'].str.startswith('park_') & ~coefs['feature'].str.contains('_x_')]
        return park_effects

    def get_interaction_effects(self) -> pd.DataFrame:
        """
        Extract weather × park interaction coefficients.

        Returns
        -------
        pd.DataFrame
            Interaction coefficients
        """
        coefs = self.get_coefficients()
        interactions = coefs[coefs['feature'].str.contains('_x_')]
        return interactions

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ParkWeatherRegressor':
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        return instance


def train_all_models(
    splits: Dict[str, np.ndarray],
    model_type: str = 'ridge'
) -> Tuple[ParkWeatherRegressor, ParkWeatherRegressor]:
    """
    Train both strikeouts and runs models.

    Parameters
    ----------
    splits : Dict
        Output from data_prep.train_test_split_by_season
    model_type : str
        'ridge' or 'lasso'

    Returns
    -------
    Tuple of (strikeouts_model, runs_model)
    """
    feature_names = splits['feature_names']

    # Train strikeouts model
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} model for STRIKEOUTS...")
    strikeouts_model = ParkWeatherRegressor(model_type=model_type)
    strikeouts_model.fit(
        splits['X_train'],
        splits['y_train_strikeouts'],
        feature_names=feature_names
    )

    # Evaluate on train and test
    train_metrics = strikeouts_model.evaluate(splits['X_train'], splits['y_train_strikeouts'])
    test_metrics = strikeouts_model.evaluate(splits['X_test'], splits['y_test_strikeouts'])
    print(f"  Train - RMSE: {train_metrics['rmse']:.3f}, MAE: {train_metrics['mae']:.3f}, R²: {train_metrics['r2']:.3f}")
    print(f"  Test  - RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}, R²: {test_metrics['r2']:.3f}")

    # Train runs model
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} model for RUNS...")
    runs_model = ParkWeatherRegressor(model_type=model_type)
    runs_model.fit(
        splits['X_train'],
        splits['y_train_runs'],
        feature_names=feature_names
    )

    train_metrics = runs_model.evaluate(splits['X_train'], splits['y_train_runs'])
    test_metrics = runs_model.evaluate(splits['X_test'], splits['y_test_runs'])
    print(f"  Train - RMSE: {train_metrics['rmse']:.3f}, MAE: {train_metrics['mae']:.3f}, R²: {train_metrics['r2']:.3f}")
    print(f"  Test  - RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}, R²: {test_metrics['r2']:.3f}")

    return strikeouts_model, runs_model


if __name__ == '__main__':
    from data_prep import load_all_team_data, prepare_features, train_test_split_by_season

    # Load and prepare data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\nPreparing features...")
    X, y_k, y_runs = prepare_features(df, include_interactions=True)

    print("\nSplitting data...")
    splits = train_test_split_by_season(df, X, y_k, y_runs)

    # Train Ridge models
    ridge_k_model, ridge_runs_model = train_all_models(splits, model_type='ridge')

    # Train Lasso models
    lasso_k_model, lasso_runs_model = train_all_models(splits, model_type='lasso')

    # Show top coefficients for Ridge strikeouts model
    print("\n" + "="*50)
    print("Top 20 coefficients for Ridge STRIKEOUTS model:")
    print(ridge_k_model.get_coefficients().head(20).to_string(index=False))

    print("\nMain weather effects:")
    print(ridge_k_model.get_weather_main_effects().to_string(index=False))

    # Save models
    models_dir = script_dir
    ridge_k_model.save(models_dir / 'ridge_strikeouts.joblib')
    ridge_runs_model.save(models_dir / 'ridge_runs.joblib')
    lasso_k_model.save(models_dir / 'lasso_strikeouts.joblib')
    lasso_runs_model.save(models_dir / 'lasso_runs.joblib')
    print("\nModels saved to models/ directory")
