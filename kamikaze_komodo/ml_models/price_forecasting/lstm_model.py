# FILE: kamikaze_komodo/ml_models/price_forecasting/lstm_model.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any, List, Union, Tuple
from sklearn.preprocessing import MinMaxScaler

from kamikaze_komodo.ml_models.price_forecasting.base_forecaster import BasePriceForecaster
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

# Define the PyTorch LSTM Model
class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Get the output from the last time step
        return out

class LSTMForecaster(BasePriceForecaster):
    """
    LSTM-based price forecaster using PyTorch.
    Predicts future price movement based on a sequence of historical data.
    """
    def __init__(self, model_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_path, params)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.sequence_length = int(self.params.get('sequencelength', 60))
        self.feature_columns_str = self.params.get('featurecolumns', 'close,log_return_lag_1,close_change_lag_1,volatility_5,RSI_14')
        self.feature_columns = [col.strip() for col in self.feature_columns_str.split(',')]
        self.num_features = len(self.feature_columns)


        # Model hyperparameters
        self.hidden_size = int(self.params.get('hiddensize', 50))
        self.num_layers = int(self.params.get('numlayers', 2))
        self.dropout = float(self.params.get('dropout', 0.2))
        self.num_epochs = int(self.params.get('numepochs', 20))
        self.batch_size = int(self.params.get('batchsize', 32))
        self.learning_rate = float(self.params.get('learningrate', 0.001))

        # Initialize model architecture but don't load weights from super() as it's not implemented there for torch
        self.model = LSTMNetwork(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout
        )
        if self.model_path:
            self.load_model(self.model_path)

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()
        df = data.copy()

        import pandas_ta as ta
        for lag in [1, 3, 5]:
            df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
            df[f'close_change_lag_{lag}'] = df['close'].pct_change(lag)
        df['volatility_5'] = df['log_return_lag_1'].rolling(window=5).std()
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))

        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _create_sequences(self, data: np.ndarray, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        # Exclude target from feature set
        feature_indices = [i for i in range(data.shape[1]) if i != target_idx]
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), feature_indices])
            y.append(data[i + self.sequence_length, target_idx])
        return np.array(X), np.array(y)


    def train(self, historical_data: pd.DataFrame, target_column: str = 'close_change_lag_1_future', feature_columns: Optional[list] = None):
        logger.info("Starting LSTM training...")
        df_features = self.create_features(historical_data)

        if feature_columns is None:
            feature_columns = self.feature_columns

        df_features['target'] = df_features['close'].pct_change(1).shift(-1)

        # --- FIX: Select final columns BEFORE dropping NaN values ---
        final_columns_to_use = feature_columns + ['target']
        
        # Check if all columns are present
        missing_cols = set(final_columns_to_use) - set(df_features.columns)
        if missing_cols:
            logger.error(f"Missing required columns for LSTM training: {missing_cols}. Cannot proceed.")
            return

        features_with_target = df_features[final_columns_to_use]
        features_with_target = features_with_target.dropna()

        if features_with_target.empty:
            logger.error("DataFrame is empty after selecting features and dropping NaNs. Not enough data to create complete feature/target rows. Cannot train LSTM model.")
            return
        # --- END FIX ---

        # Scale data
        scaled_data = self.scaler.fit_transform(features_with_target)
        
        # Find index of target column for sequence creation
        target_idx = features_with_target.columns.get_loc('target')
        
        X, y = self._create_sequences(scaled_data, target_idx)
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float().view(-1, 1)

        # Training Loop
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            outputs = self.model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.6f}')

        self.trained_feature_columns_ = feature_columns
        logger.info("LSTM model training completed.")

    def predict(self, new_data: pd.DataFrame, feature_columns: Optional[list] = None) -> Union[pd.Series, float, None]:
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not available.")
            return None

        self.model.eval()
        df_features = self.create_features(new_data)

        if feature_columns is None:
            feature_columns = self.trained_feature_columns_ or self.feature_columns

        if len(df_features) < self.sequence_length:
            return None

        last_sequence_unscaled = df_features[feature_columns].iloc[-self.sequence_length:]
        if last_sequence_unscaled.isnull().values.any():
            return None # Cannot predict with NaNs

        # Use a temporary scaler fitted on the feature columns only to avoid data leakage
        temp_scaler = MinMaxScaler(feature_range=(-1,1))
        # This scaler should ideally be fitted on the same training distribution
        # For simplicity here, we transform based on the scaler fitted during training, which included the target
        # A more robust solution involves saving separate scalers for features and target.
        
        # We need to create a dummy dataframe with all original columns to use the fitted scaler
        all_cols = self.trained_feature_columns_ + ['target']
        dummy_df_for_scaling = pd.DataFrame(columns=all_cols)
        for col in feature_columns:
            dummy_df_for_scaling[col] = last_sequence_unscaled[col]
        dummy_df_for_scaling = dummy_df_for_scaling.fillna(0) # fill target with dummy values
        
        scaled_data = self.scaler.transform(dummy_df_for_scaling)
        
        # Extract only the feature columns from the scaled data
        target_idx = all_cols.index('target')
        feature_indices = [i for i, col in enumerate(all_cols) if i != target_idx]
        scaled_sequence = scaled_data[:, feature_indices]


        with torch.no_grad():
            input_tensor = torch.from_numpy(scaled_sequence).float().unsqueeze(0) # Add batch dimension
            prediction_scaled = self.model(input_tensor)

        # We need to inverse transform the prediction. This requires a dummy array.
        dummy_array = np.zeros((1, len(all_cols)))
        dummy_array[0, target_idx] = prediction_scaled.item()
        prediction_unscaled = self.scaler.inverse_transform(dummy_array)[0, target_idx]

        return float(prediction_unscaled)

    def save_model(self, path: str):
        try:
            state = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'params': self.params,
                'trained_feature_columns': getattr(self, 'trained_feature_columns_', None)
            }
            torch.save(state, path)
            logger.info(f"LSTM model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}", exc_info=True)

    def load_model(self, path: str):
        try:
            # FIX: Set weights_only=False to allow loading sklearn scaler object
            # This is safe as we are loading a file we just saved in a trusted environment.
            state = torch.load(path, weights_only=False)
            self.model.load_state_dict(state['model_state_dict'])
            self.scaler = state['scaler']
            self.params = state['params']
            self.trained_feature_columns_ = state.get('trained_feature_columns')
            self.model.eval()
            self.model_path = path
            logger.info(f"LSTM model loaded from {path}")
        except FileNotFoundError:
            logger.error(f"LSTM model file not found at {path}.")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}", exc_info=True)
            self.model = None