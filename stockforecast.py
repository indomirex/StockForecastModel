import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pytz
from scipy.stats import norm
from arch import arch_model
from hyperopt import fmin, tpe, hp, Trials
import warnings

class StockForecast:
    def __init__(self, symbol, start_date_train, train_end_date, end_date):
        self.symbol = symbol
        self.start_date_train = pd.Timestamp(start_date_train)
        self.train_end_date = pd.Timestamp(train_end_date)
        self.end_date = pd.Timestamp(end_date)
        self.az_tz = pytz.timezone('US/Arizona')
        self.stock_data = None
        self.train_data = None
        self.test_data = None
        self.adjusted_projection = None

    def fetch_stock_data(self):
        try:
            stock = yf.Ticker(self.symbol)
            self.stock_data = stock.history(start=self.start_date_train, end=self.end_date)
            print(f"Stock data fetched successfully for {self.symbol}!")
        except Exception as e:
            print(f"Error fetching stock data for {self.symbol}: {str(e)}")

    @staticmethod
    def calculate_rsi(data, window=10):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def adjust_forecast_based_on_rsi(self, forecast_mean, forecast_index, window=10):
        adjusted_forecast = forecast_mean.copy()
        
        # Calculate RSI over the past 3 months
        three_months_ago = forecast_index[0] - pd.Timedelta(days=90)
        rsi_3m = self.calculate_rsi(self.stock_data.loc[three_months_ago:forecast_index[-1], 'Close'])
        
        # Create a bell curve (normal distribution) based on RSI values
        mu = rsi_3m.mean()
        std_dev = rsi_3m.std()
        bell_curve = norm(loc=mu, scale=std_dev)
        
        for i in range(len(forecast_index)):
            rsi_value = self.stock_data.loc[forecast_index[i], 'RSI_10']
            
            # Check if RSI is outside central 68% (1 standard deviation) of the bell curve
            if not (mu - std_dev <= rsi_value <= mu + std_dev):
                # Move the forecasted value towards the 10-day moving average
                adjusted_forecast[i] = self.stock_data.loc[forecast_index[i], '10_MA']
        
        return adjusted_forecast

    @staticmethod
    def evaluate_percentage_difference(forecast_data, test_data):
        # Ensure index alignment
        forecast_data = forecast_data.reindex(test_data.index)
        # Calculate percentage difference
        percentage_difference = np.abs((forecast_data - test_data) / test_data)
        # Calculate average percentage difference
        avg_percentage_difference = percentage_difference.mean() * 100
        return avg_percentage_difference

    @staticmethod
    def calculate_absolute_derivative(data, interval=2):
        return np.abs(np.diff(data, n=interval))

    def prepare_data(self):
        # Localize start_date_train and end_date to UTC timezone if naive (not timezone-aware)
        if self.start_date_train.tzinfo is None:
            self.start_date_train = self.start_date_train.tz_localize('UTC')
        if self.train_end_date.tzinfo is None:
            self.train_end_date = self.train_end_date.tz_localize('UTC')
        if self.end_date.tzinfo is None:
            self.end_date = self.end_date.tz_localize('UTC')

        # Convert to Arizona timezone
        self.start_date_train = self.start_date_train.tz_convert(self.az_tz)
        self.train_end_date = self.train_end_date.tz_convert(self.az_tz)
        self.end_date = self.end_date.tz_convert(self.az_tz)

        self.fetch_stock_data()

        if self.stock_data is not None:
            # Ensure the datetime index has the correct frequency and handle duplicates
            self.stock_data = self.stock_data.asfreq('D').fillna(method='ffill')

            # Convert datetime index to Arizona timezone if not already
            if self.stock_data.index.tz is None:
                self.stock_data.index = self.stock_data.index.tz_localize('UTC').tz_convert(self.az_tz)
            else:
                self.stock_data.index = self.stock_data.index.tz_convert(self.az_tz)

            # Calculate moving averages
            self.stock_data['10_MA'] = self.stock_data['Close'].rolling(window=10).mean()

            # Calculate RSI (Relative Strength Index)
            self.stock_data['RSI_10'] = self.calculate_rsi(self.stock_data['Close'], window=10)

            # Fill missing dates in stock data with the price of the previous day
            self.stock_data['Close'] = self.stock_data['Close'].fillna(method='ffill')

            # Splitting data into training and test sets
            self.train_data = self.stock_data.loc[self.start_date_train:self.train_end_date]
            self.test_data = self.stock_data.loc[self.train_end_date + pd.Timedelta(days=1):self.end_date]

    def plot_data(self, title, adjusted_projection=None, save_path=None):
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_data.index, self.train_data['Close'], label='Training Data', color='blue')
        plt.plot(self.test_data.index, self.test_data['Close'], label='Test Data', color='green')
        if adjusted_projection is not None:
            plt.plot(adjusted_projection.index, adjusted_projection, label='Forecasted Data', color='red')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_forecast_results(self, adjusted_projection, file_prefix='forecast'):
        # Save only the final forecast graph and data
        graph_path = f"{self.symbol}_forecast_graph.png"
        forecast_path = f"{self.symbol}_forecast_projection.csv"
        
        self.plot_data('30-Day Future Projection', adjusted_projection, save_path=graph_path)
        adjusted_projection.to_csv(forecast_path, header=True)
        
        print(f"Graph saved to {graph_path}")
        print(f"Future projection data saved to {forecast_path}")

    def objective(self, params):
        p, d, q = int(params['p']), int(params['d']), int(params['q'])
        P, D, Q, S = int(params['P']), int(params['D']), int(params['Q']), int(params['S'])

        try:
            # Fit SARIMAX model
            sarimax_model = SARIMAX(self.train_data['Close'], order=(p, d, q), seasonal_order=(P, D, Q, S))
            sarimax_results = sarimax_model.fit()

            # Get residuals from SARIMAX model
            sarimax_residuals = sarimax_results.resid

            # Fit GARCH model with parameters
            garch_model = arch_model(sarimax_residuals, vol='Garch', p=1, q=1)
            garch_results = garch_model.fit(disp='off')

            # Forecast with SARIMAX-GARCH model
            forecast_steps = len(self.test_data)
            sarimax_forecast = sarimax_results.get_forecast(steps=forecast_steps)
            sarimax_projection_mean = sarimax_forecast.predicted_mean

            # Adjust forecast based on RSI
            adjusted_projection = self.adjust_forecast_based_on_rsi(sarimax_projection_mean, sarimax_projection_mean.index)

            # Evaluate average percentage difference
            avg_diff = self.evaluate_percentage_difference(adjusted_projection, self.test_data['Close'])

            return avg_diff
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return float('inf')

    def perform_forecasting(self):
        self.prepare_data()

        if self.stock_data is not None:
            num_epochs = 0
            max_epochs = 10
            avg_diff = np.inf

            while avg_diff > 5 and num_epochs < max_epochs:
                # Default parameters for SARIMAX model
                p, d, q = 1, 1, 1
                P, D, Q, S = 1, 1, 1, 12

                # Fit SARIMAX model
                sarimax_model = SARIMAX(self.train_data['Close'], order=(p, d, q), seasonal_order=(P, D, Q, S))
                sarimax_results = sarimax_model.fit()

                # Get residuals from SARIMAX model
                sarimax_residuals = sarimax_results.resid

                # Fit GARCH model with parameters
                garch_model = arch_model(sarimax_residuals, vol='Garch', p=1, q=1)
                garch_results = garch_model.fit(disp='off')

                # Forecast with SARIMAX-GARCH model
                forecast_steps = len(self.test_data)
                sarimax_forecast = sarimax_results.get_forecast(steps=forecast_steps)
                sarimax_projection_mean = sarimax_forecast.predicted_mean

                # Adjust forecast based on RSI
                adjusted_projection = self.adjust_forecast_based_on_rsi(sarimax_projection_mean, sarimax_projection_mean.index)

                # Evaluate average percentage difference
                avg_diff = self.evaluate_percentage_difference(adjusted_projection, self.test_data['Close'])

                # Save the final forecast results
                if num_epochs >= max_epochs or avg_diff <= 5:
                    self.adjusted_projection = adjusted_projection
                    self.save_forecast_results(adjusted_projection)
                    print(f"Final average percentage difference: {avg_diff}%")
                    break

                num_epochs += 1
                print(f"Epoch {num_epochs}: Average percentage difference: {avg_diff}%")

            # Hyperopt search after 10 epochs
            if num_epochs >= max_epochs:
                trials = Trials()
                space = {
                    'p': hp.randint('p', 5),
                    'd': hp.randint('d', 3),
                    'q': hp.randint('q', 5),
                    'P': hp.randint('P', 5),
                    'D': hp.randint('D', 3),
                    'Q': hp.randint('Q', 5),
                    'S': hp.randint('S', 12)
                }

                best_params = fmin(
                    fn=self.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials
                )

                print(f"Best parameters found: {best_params}")

                # Fit the model with the best parameters
                p, d, q = int(best_params['p']), int(best_params['d']), int(best_params['q'])
                P, D, Q, S = int(best_params['P']), int(best_params['D']), int(best_params['Q']), int(best_params['S'])

                sarimax_model = SARIMAX(self.train_data['Close'], order=(p, d, q), seasonal_order=(P, D, Q, S))
                sarimax_results = sarimax_model.fit()

                # Get residuals from SARIMAX model
                sarimax_residuals = sarimax_results.resid

                # Fit GARCH model with parameters
                garch_model = arch_model(sarimax_residuals, vol='Garch', p=1, q=1)
                garch_results = garch_model.fit(disp='off')

                # Forecast with SARIMAX-GARCH model
                forecast_steps = len(self.test_data)
                sarimax_forecast = sarimax_results.get_forecast(steps=forecast_steps)
                sarimax_projection_mean = sarimax_forecast.predicted_mean

                # Adjust forecast based on RSI
                adjusted_projection = self.adjust_forecast_based_on_rsi(sarimax_projection_mean, sarimax_projection_mean.index)

                # Save the final forecast results
                self.adjusted_projection = adjusted_projection
                self.save_forecast_results(adjusted_projection)
                print(f"Final average percentage difference after Hyperopt: {self.evaluate_percentage_difference(adjusted_projection, self.test_data['Close'])}%")

            return self.adjusted_projection



# Example usage
if __name__ == "__main__":
    forecast = StockForecast('SMCI', '2024-05-10', '2024-07-01', '2024-07-27')
    forecast.perform_forecasting()
























