"""
Neural network models with learned park embeddings for away team prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


class ParkEmbeddingNet(nn.Module):
    """
    Neural network with park embedding layer for weather-based prediction.
    """

    def __init__(
        self,
        n_parks: int,
        n_weather_features: int,
        embedding_dim: int = 8,
        hidden_dims: list = [64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize the network.

        Parameters
        ----------
        n_parks : int
            Number of unique parks (for embedding layer)
        n_weather_features : int
            Number of weather features
        embedding_dim : int
            Dimension of park embedding vectors
        hidden_dims : list
            Sizes of hidden layers
        dropout : float
            Dropout rate for regularization
        """
        super().__init__()

        self.n_parks = n_parks
        self.n_weather_features = n_weather_features
        self.embedding_dim = embedding_dim

        # Park embedding layer
        self.park_embedding = nn.Embedding(n_parks, embedding_dim)

        # Build hidden layers
        input_dim = n_weather_features + embedding_dim
        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, weather_features: torch.Tensor, park_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        weather_features : torch.Tensor
            Weather features (batch_size, n_weather_features)
        park_ids : torch.Tensor
            Park indices (batch_size,)

        Returns
        -------
        torch.Tensor
            Predictions (batch_size, 1)
        """
        # Get park embeddings
        park_emb = self.park_embedding(park_ids)  # (batch_size, embedding_dim)

        # Concatenate weather features and park embeddings
        x = torch.cat([weather_features, park_emb], dim=1)

        # Pass through network
        return self.network(x)

    def get_park_embeddings(self) -> np.ndarray:
        """Get the learned park embedding vectors."""
        return self.park_embedding.weight.detach().cpu().numpy()


class EmbeddingModelTrainer:
    """
    Trainer for park embedding neural network models.
    """

    def __init__(
        self,
        n_parks: int,
        n_weather_features: int,
        embedding_dim: int = 8,
        hidden_dims: list = [64, 32],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = 'auto'
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        n_parks : int
            Number of unique parks
        n_weather_features : int
            Number of weather features
        embedding_dim : int
            Park embedding dimension
        hidden_dims : list
            Hidden layer sizes
        dropout : float
            Dropout rate
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            L2 regularization weight
        device : str
            'auto', 'cuda', or 'cpu'
        """
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create model
        self.model = ParkEmbeddingNet(
            n_parks=n_parks,
            n_weather_features=n_weather_features,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Training history
        self.train_losses = []
        self.val_losses = []

    def fit(
        self,
        X_weather: np.ndarray,
        park_ids: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.15,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        verbose: bool = True
    ) -> 'EmbeddingModelTrainer':
        """
        Train the model with early stopping.

        Parameters
        ----------
        X_weather : np.ndarray
            Weather features (n_samples, n_features)
        park_ids : np.ndarray
            Park indices (n_samples,)
        y : np.ndarray
            Target variable
        val_split : float
            Fraction of training data to use for validation
        epochs : int
            Maximum number of epochs
        batch_size : int
            Batch size for training
        patience : int
            Early stopping patience
        verbose : bool
            Whether to print progress

        Returns
        -------
        self
        """
        # Split into train and validation
        n = len(y)
        n_val = int(n * val_split)
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_weather[train_idx]),
            torch.LongTensor(park_ids[train_idx]),
            torch.FloatTensor(y[train_idx]).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_weather[val_idx]),
            torch.LongTensor(park_ids[val_idx]),
            torch.FloatTensor(y[val_idx]).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for weather, parks, targets in train_loader:
                weather = weather.to(self.device)
                parks = parks.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(weather, parks)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(targets)

            train_loss /= len(train_idx)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for weather, parks, targets in val_loader:
                    weather = weather.to(self.device)
                    parks = parks.to(self.device)
                    targets = targets.to(self.device)

                    predictions = self.model(weather, parks)
                    loss = self.criterion(predictions, targets)
                    val_loss += loss.item() * len(targets)

            val_loss /= len(val_idx)
            self.val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"  Best validation loss: {best_val_loss:.4f}")

        return self

    def predict(self, X_weather: np.ndarray, park_ids: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X_weather : np.ndarray
            Weather features
        park_ids : np.ndarray
            Park indices

        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            weather = torch.FloatTensor(X_weather).to(self.device)
            parks = torch.LongTensor(park_ids).to(self.device)
            predictions = self.model(weather, parks)
            return predictions.cpu().numpy().flatten()

    def evaluate(self, X_weather: np.ndarray, park_ids: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        X_weather : np.ndarray
            Weather features
        park_ids : np.ndarray
            Park indices
        y : np.ndarray
            True target values

        Returns
        -------
        Dict with RMSE, MAE, and R² metrics
        """
        y_pred = self.predict(X_weather, park_ids)

        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

    def get_park_embeddings(self, id_to_park: Dict[int, str]) -> pd.DataFrame:
        """
        Get park embeddings as a DataFrame.

        Parameters
        ----------
        id_to_park : Dict
            Mapping from park ID to park name

        Returns
        -------
        pd.DataFrame
            Park embeddings with park names as index
        """
        embeddings = self.model.get_park_embeddings()
        parks = [id_to_park[i] for i in range(len(id_to_park))]

        df = pd.DataFrame(
            embeddings,
            index=parks,
            columns=[f'emb_{i}' for i in range(embeddings.shape[1])]
        )
        return df

    def save(self, filepath: str):
        """Save model to disk."""
        torch.save({
            'model_state': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        data = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(data['model_state'])
        self.train_losses = data['train_losses']
        self.val_losses = data['val_losses']


def train_nn_models(
    nn_data: Dict[str, np.ndarray],
    embedding_dim: int = 8,
    hidden_dims: list = [64, 32],
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15
) -> Tuple[EmbeddingModelTrainer, EmbeddingModelTrainer]:
    """
    Train both strikeouts and runs neural network models.

    Parameters
    ----------
    nn_data : Dict
        Output from data_prep.prepare_nn_data
    embedding_dim : int
        Park embedding dimension
    hidden_dims : list
        Hidden layer sizes
    epochs : int
        Maximum training epochs
    batch_size : int
        Batch size
    patience : int
        Early stopping patience

    Returns
    -------
    Tuple of (strikeouts_trainer, runs_trainer)
    """
    n_parks = nn_data['n_parks']
    n_weather_features = nn_data['n_weather_features']

    # Train strikeouts model
    print(f"\n{'='*50}")
    print(f"Training Neural Network for STRIKEOUTS...")
    print(f"  Architecture: {n_weather_features} weather + {embedding_dim}D park embedding -> {hidden_dims} -> 1")

    k_trainer = EmbeddingModelTrainer(
        n_parks=n_parks,
        n_weather_features=n_weather_features,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims
    )
    k_trainer.fit(
        nn_data['X_weather_train'],
        nn_data['park_ids_train'],
        nn_data['y_train_strikeouts'],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience
    )

    train_metrics = k_trainer.evaluate(
        nn_data['X_weather_train'],
        nn_data['park_ids_train'],
        nn_data['y_train_strikeouts']
    )
    test_metrics = k_trainer.evaluate(
        nn_data['X_weather_test'],
        nn_data['park_ids_test'],
        nn_data['y_test_strikeouts']
    )
    print(f"  Train - RMSE: {train_metrics['rmse']:.3f}, MAE: {train_metrics['mae']:.3f}, R²: {train_metrics['r2']:.3f}")
    print(f"  Test  - RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}, R²: {test_metrics['r2']:.3f}")

    # Train runs model
    print(f"\n{'='*50}")
    print(f"Training Neural Network for RUNS...")
    print(f"  Architecture: {n_weather_features} weather + {embedding_dim}D park embedding -> {hidden_dims} -> 1")

    runs_trainer = EmbeddingModelTrainer(
        n_parks=n_parks,
        n_weather_features=n_weather_features,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims
    )
    runs_trainer.fit(
        nn_data['X_weather_train'],
        nn_data['park_ids_train'],
        nn_data['y_train_runs'],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience
    )

    train_metrics = runs_trainer.evaluate(
        nn_data['X_weather_train'],
        nn_data['park_ids_train'],
        nn_data['y_train_runs']
    )
    test_metrics = runs_trainer.evaluate(
        nn_data['X_weather_test'],
        nn_data['park_ids_test'],
        nn_data['y_test_runs']
    )
    print(f"  Train - RMSE: {train_metrics['rmse']:.3f}, MAE: {train_metrics['mae']:.3f}, R²: {train_metrics['r2']:.3f}")
    print(f"  Test  - RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}, R²: {test_metrics['r2']:.3f}")

    return k_trainer, runs_trainer


if __name__ == '__main__':
    from data_prep import load_all_team_data, prepare_nn_data

    # Load and prepare data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    print("Loading data...")
    df = load_all_team_data(data_dir)

    print("\nPreparing NN data...")
    nn_data = prepare_nn_data(df)

    # Train models
    k_trainer, runs_trainer = train_nn_models(
        nn_data,
        embedding_dim=8,
        hidden_dims=[64, 32],
        epochs=100,
        patience=15
    )

    # Show park embeddings
    print("\n" + "="*50)
    print("Park Embeddings (first 5 dimensions):")
    embeddings = k_trainer.get_park_embeddings(nn_data['id_to_park'])
    print(embeddings.iloc[:, :5].to_string())

    # Save models
    k_trainer.save(script_dir / 'nn_strikeouts.pt')
    runs_trainer.save(script_dir / 'nn_runs.pt')
    print("\nModels saved to models/ directory")
