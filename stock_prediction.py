import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def fetch_stock_data(stock_symbol):
    # Simulated function to fetch stock data (Replace with actual API)
    data = {
        "ds": pd.date_range(start="2023-01-01", periods=365, freq="D"),
        "y": [i + (i * 0.01) for i in range(365)]  # Simulated price trend
    }
    df = pd.DataFrame(data)

    # Remove timezone from 'ds' column
    df['ds'] = df['ds'].dt.tz_localize(None)

    return df

def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def predict_future(model, days=60):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

def plot_predictions(df, forecast):
    plt.plot(df['ds'], df['y'], label="Actual Stock Price")
    plt.plot(forecast['ds'], forecast['yhat'], label="Predicted Price", linestyle="dashed")
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    plt.legend()
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

if __name__ == "__main__":
    stock_symbol = "AAPL"  # Change this to any stock ticker
    df = fetch_stock_data(stock_symbol)
    model = train_prophet_model(df)
    forecast = predict_future(model, days=60)
    plot_predictions(df, forecast)
