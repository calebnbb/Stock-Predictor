import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# Step 1: Load data
ticker = 'AAPL'  # You can change this to any stock symbol
data = yf.download(ticker, start="2020-01-01", end="2024-12-31")
data = data.reset_index()

# Step 2: Prepare data
data['Date'] = data['Date'].map(datetime.datetime.toordinal)
X = np.array(data['Date']).reshape(-1, 1)
y = np.array(data['Close'])

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict future
future_date = datetime.date.today() + datetime.timedelta(days=30)
future_date_ordinal = np.array([[future_date.toordinal()]])
prediction = model.predict(future_date_ordinal)

print(f"Predicted price of {ticker} in 30 days: ${prediction[0]:.2f}")

# Step 5: Visualize
plt.plot(data['Date'], y, label='Historical Prices')
plt.plot(future_date_ordinal, prediction, 'ro', label='Predicted Price')
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
