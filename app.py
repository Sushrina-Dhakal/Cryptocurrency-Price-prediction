# app.py
from flask import Flask, render_template,request
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import time
import hmac
import hashlib
from urllib.parse import urlencode
import joblib
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = Flask(__name__)

# Binance API base URL for Testnet
BASE_URL = "https://testnet.binancefuture.com/fapi/v1"

API_KEY = "5e7d1385e687bcc1271ffe07ffc9dc295c50d6d400ae85a1e270b08a9c91118d" #d
API_SECRET = "473a3584d4af60fd093aaf5922dee984ece5e35a7f436ed71fff517c9cfac8d5" #5

def generate_recommendation(last_day_price, predicted_prices):
    if predicted_prices[-1] > last_day_price:
        return "Recommendation: Buy"
    else:
        return "Recommendation: Sell"

def generate_signature(data):
    return hmac.new(API_SECRET.encode(), urlencode(data).encode(), hashlib.sha256).hexdigest()

def download_data(symbol, startdate, enddate):
    return yf.download(symbol, start=startdate, end=enddate)


CRYPTO_MODELS = {
    'BTC': 'BTC_model_gru.pkl', #20
    'ETH': 'Ethereum_model_lstm.pkl', #60
    'BNB': 'BNB_model_lstm.pkl', #25
    'BCH': 'BCH_model_gru.pkl', #25
    'LTC': 'LTC_model_gru.pkl' #20
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trade', methods=['GET', 'POST'])
def trade():
    if request.method == "POST":
        symbol = request.form['symbol']
        amount = request.form['amount']
        side = request.form['side'] # buy or sell

        if side == 'buy':
            # Place Buy Order Logic
            endpoint = f"{BASE_URL}/order"
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': amount,
                'timestamp': int(time.time() * 1000)
            }
            headers = {
                'X-MBX-APIKEY': API_KEY
            }
            params['signature'] = generate_signature(params)
            response = requests.post(endpoint, params=params, headers=headers)
            if response.status_code == 200:
                print(response.json())  # You can handle the response as per your requirement
                return "Trade successful"
            
            else:
                error_message = f"Error: {response.status_code} - {response.reason}"
                return error_message
        
        elif side == 'sell':
            # Place Sell Order Logic
            endpoint = f"{BASE_URL}/order"
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': amount,
                'timestamp': int(time.time() * 1000)
            }
            headers = {
                'X-MBX-APIKEY': API_KEY
            }
            params['signature'] = generate_signature(params)
            response = requests.post(endpoint, params=params, headers=headers)
            if response.status_code == 200:
                print(response.json())  # You can handle the response as per your requirement
                return "Trade successful"
            
            else:
                error_message = f"Error: {response.status_code} - {response.reason}"
                return error_message
        
        else:
            return "Invalid trade side"
    else:
        return render_template('trade.html')
    

@app.route('/analysis',methods=['GET', 'POST'])
def analysis():
    if request.method == "POST":
        crypto_symbol = request.form.get('crypto_symbol') # BTC
        if crypto_symbol: #BTC
            model_path = CRYPTO_MODELS.get(crypto_symbol) # BTC_model_gru.pkl
            if model_path:
                # Analysis logic
                # BTC-USD
                df = download_data(crypto_symbol+"-USD", "2023-01-01", "2024-03-02")
                df = df.filter(['Close']).values

                scaler = MinMaxScaler(feature_range=(0,1))

                df_scaled = scaler.fit_transform(df)

                X, Y = [], []

                sequence_length = 0

                if crypto_symbol == "BTC" or crypto_symbol == "LTC":
                    sequence_length = 20
                elif crypto_symbol == "BCH" or crypto_symbol == "BNB":
                    sequence_length = 25
                elif crypto_symbol == "ETH":
                    sequence_length = 25

                for i in range(sequence_length, len(df_scaled)):
                    X.append(df_scaled[i-sequence_length:i, 0])
                    Y.append(df_scaled[i, 0])
                
                X = np.array(X)
                Y = np.array(Y)

                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                model = joblib.load(model_path) #BTC_model_gru.pkl

                last_sequence = X[-1]

                last_day_price = Y[-1]

                # Reshape last_sequence to match the input shape of the model
                last_sequence = last_sequence.reshape((1, sequence_length, 1))

                # Predict the next 7 days
                predicted_prices = []
                for _ in range(7):
                    lstm_predicted_price = model.predict(last_sequence)
                    predicted_prices.append(lstm_predicted_price[0, 0])
                    
                    # Update last_sequence with the newly predicted price
                    last_sequence = np.append(last_sequence[:,1:,:], lstm_predicted_price.reshape((1,1,1)), axis=1)

                # Inverse scale the predicted prices
                predicted_prices = np.array(predicted_prices).reshape(-1,1)
                predicted_prices = scaler.inverse_transform(predicted_prices)

                last_day_price = last_day_price.reshape(-1, 1)
                last_day_price = scaler.inverse_transform(last_day_price)

                predictions = predicted_prices

                recommendation = generate_recommendation(last_day_price,predicted_prices)

               
                today = datetime.now()  # Get today's date
                dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

                return render_template('analysis.html', dates=dates, crypto_symbol=crypto_symbol, predictions=predictions, recommendation=recommendation, last_day = last_day_price)
            else:
                return "Model not found for the selected cryptocurrency."
        else:
            return "No crypto symbol provided"
    else:
        return render_template('analysis.html')  # Render analysis.html for GET requests
    
   


if __name__ == '__main__':
    app.run(debug=True)
