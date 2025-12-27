LSTM Stock Price Predictor – PyTorch + Streamlit
This project is an educational demo that downloads historical stock prices from Yahoo Finance, trains an LSTM model in PyTorch to forecast prices, and exposes everything through an interactive Streamlit web app. ​

1. Overview
The app does three main things:

Fetches historical stock data (close prices) from Yahoo Finance using yfinance.​

Preprocesses the data into supervised sequences and trains an LSTM regression model with PyTorch.​

Visualizes historical prices and model predictions interactively using Streamlit charts.​

You can choose the ticker, history window, and basic LSTM hyperparameters directly from the sidebar.

It can only give you advice on what trades to make, it cannot be considered true financial advice.

2. Project Structure
A typical layout for the code shown earlier:

text
.
├── app.py        # Streamlit UI + orchestration
├── data.py       # Data download & preprocessing (yfinance + scaling)
└── model.py      # PyTorch LSTM model definition and training utilities
app.py is the entry point you run with Streamlit.

data.py handles pulling prices and converting them into sequences suitable for time‑series forecasting.​

model.py defines the LSTM network and training / prediction functions using PyTorch.​

3. Setup and Installation
Dependencies
Python 3.8+

PyTorch (CPU or GPU)

Streamlit

yfinance

pandas, numpy, scikit‑learn

You can install them via:

bash
pip install torch torchvision torchaudio  # choose correct index URL if needed
pip install streamlit yfinance pandas numpy scikit-learn
yfinance is a convenient wrapper around Yahoo Finance’s market data, which is widely used in stock price demos.​
Streamlit is used for quick dashboarding and time‑series visualization.​

Running the app
From the project root:

bash
streamlit run app.py
Streamlit will open a browser window (or give you a local URL, usually http://localhost:8501).

4. Data Pipeline (data.py)
4.1. Downloading stock data
python
df = yf.download(ticker, period=period, interval=interval)
df = df[['Close']].dropna()
ticker – stock symbol (e.g., AAPL, MSFT, TSLA).

period – history length (e.g., 1y, 2y, 5y, max).

interval – sampling frequency (e.g., 1d, 1h, 15m).

The pipeline keeps only the Close price to forecast a single target series, which is a common simplification in LSTM stock tutorials.​

4.2. Scaling and sequence creation
The close price is scaled to 
[
0
,
1
]
[0,1] via MinMaxScaler from scikit‑learn to help the LSTM train more stably.​

A sliding window of length seq_len is used to create supervised examples:

Input 
X
X: last seq_len scaled close prices.

Target 
y
y: the next scaled close price.

The dataset is split into train and test sets with train_ratio (e.g., 80% train, 20% test).

The result:

X_train, y_train – numpy arrays of shape (N_train, seq_len, 1) and (N_train, 1).

X_test, y_test – same shapes for evaluation.

scaler – needed later to inverse‑transform predictions back to price levels.

5. Model and Training (model.py)
5.1. LSTM architecture
python
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        ...
input_size=1 because each time step has one feature (close price).

hidden_size – number of hidden units in the LSTM; tunable from the UI.

num_layers – stacked LSTM layers; deeper networks can capture more complex patterns at the cost of more overfitting risk.​

dropout – applied between LSTM layers (if more than one) as regularization.​

forward:

Passes a batch of sequences through the LSTM: shape (batch, seq_len, input_size).

Takes the last time step’s hidden state and runs it through a fully connected layer to get a single predicted value per sequence.​

5.2. Training loop
train_model:

Uses MSELoss as the regression loss function.

Uses the Adam optimizer with a default learning rate (e.g., 1e-3).

Trains for num_epochs over the DataLoader created in app.py.

At each epoch:

Zero gradients, forward pass, compute loss, backpropagate, and update weights.

Accumulates average training loss and prints progress.

predict:

Puts the model in eval() mode.

Converts numpy arrays to tensors, feeds them through the model, and returns predictions as numpy arrays.

6. Streamlit App (app.py)
6.1. UI controls
The sidebar parameters include:

Ticker – stock symbol to download.

History period – how far back to fetch data.

Interval – time resolution.

Sequence length – number of past points per input sequence.

Epochs, Hidden size, LSTM layers, Batch size – training hyperparameters.

When you click Run:

The app downloads data via load_stock_data.​

It plots the historical close prices using st.line_chart.​

It prepares train/test sequences and builds PyTorch TensorDataset + DataLoader.

It instantiates StockLSTM with your chosen hyperparameters and trains it.

6.2. Visualization
After training:

The model predicts on the test set: preds_scaled = predict(model, X_test, ...).

Predictions and targets are inverse‑scaled back to price space using the same scaler.

A DataFrame with columns Actual and Predicted is created, indexed by the corresponding test dates.

st.line_chart(pred_df) shows both curves over time, making it easy to visually compare model performance.​

6.3. One‑step‑ahead forecast
The app also:

Takes the last seq_len close prices from df.

Scales them, reshapes to (1, seq_len, 1), and passes through the model.

Inverse‑transforms the output to get the next predicted close price, which is printed as a numeric value.

This demonstrates how to use the trained model for a simple next‑day (or next‑interval) forecast.

7. Limitations and Notes
The model is a simple LSTM over close prices only; it ignores features like volume, technical indicators, or macro data.​

It uses a fixed train/test split without walk‑forward or cross‑validation.​

Predictions are for educational purposes only and should not be used for real trading decisions.

You can extend the project by:

Adding more input features (Open, High, Low, Volume, indicators).​

Using more robust evaluation (rolling windows, different loss metrics).​

Replacing st.line_chart with Plotly charts for richer inter
