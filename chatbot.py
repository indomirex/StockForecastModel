import os
import pandas as pd
from datetime import datetime
from stockforecast import StockForecast
from subjectfinder import SubjectFinder
from testing import IntentClassifier

# Dictionary mapping stock names to their tickers
name_to_ticker = {
    'Carvana': 'CVNA',
    'Taiwan Semiconductor Manufacturing Company': 'TSM',
    'Palo Alto Networks': 'PANW',
    'Salesforce': 'CRM',
    'Recursion Pharmaceuticals': 'RXRX',
    'SoundHound AI': 'SOUN',
    'Arm Holdings': 'ARM',
    'Gut Health Inc.': 'GUTS',
    'Intel Corporation': 'INTC',
    'PayPal Holdings': 'PYPL',
    'Super Micro Computer': 'SMCI',
    'Exxon Mobil Corporation': 'XOM',
    'Applied Materials': 'AMAT',
    'Amazon': 'AMZN',
    'Microsoft': 'MSFT',
    'Nvidia': 'NVDA',
    'Netflix': 'NFLX',
    'Meta Platforms': 'META',
    'Alphabet Inc. (Google)': 'GOOGL',
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Broadcom Inc.': 'AVGO',
    'Lululemon Athletica': 'LULU'
}

class StockManager:
    def __init__(self):
        self.subject_finder = SubjectFinder()

    def handle_stock_query(self, stock_name, training_start_date, training_end_date):
        forecast_end_date = datetime.now().strftime('%Y-%m-%d')  # Set forecast end date to today
        ticker = name_to_ticker.get(stock_name)
        if not ticker:
            print(f"Stock name '{stock_name}' not recognized.")
            return

        print(f"Performing forecasting for {stock_name}...")
        stock_forecast = StockForecast(ticker, training_start_date, training_end_date, forecast_end_date)
        stock_forecast.perform_forecasting()

        projection_file = os.path.join(os.path.dirname(__file__), f'{ticker}_forecast_projection.csv')
        graph_file = os.path.join(os.path.dirname(__file__), f'{ticker}_forecast_graph.png')

        # Save projection data and graph
        print(f"Saving projection data to: {projection_file}")
        print(f"Saving graph to: {graph_file}")

        if os.path.exists(projection_file):
            projection_df = pd.read_csv(projection_file)
            print(f"Loaded projection data for {stock_name}:")
            print(projection_df)
            print(f"Graph available at: {graph_file}")
        else:
            print(f"Projection file for {stock_name} was not found at {projection_file}.")

    def handle_interrogative_query(self, stock_names, training_start_date, training_end_date):
        forecast_end_date = datetime.now().strftime('%Y-%m-%d')  # Set forecast end date to today
        max_change = -float('inf')
        max_change_stock = None

        for stock_name in stock_names:
            ticker = name_to_ticker.get(stock_name)
            if not ticker:
                print(f"Stock name '{stock_name}' not recognized.")
                continue

            print(f"Performing forecasting for {stock_name}...")
            stock_forecast = StockForecast(ticker, training_start_date, training_end_date, forecast_end_date)
            stock_forecast.perform_forecasting()

            projection_file = os.path.join(os.path.dirname(__file__), f'{ticker}_forecast_projection.csv')

            print(f"Checking for projection file at: {projection_file}")
            if os.path.exists(projection_file):
                projection_df = pd.read_csv(projection_file)
                initial_value = projection_df.iloc[0]['predicted_mean']
                final_value = projection_df.iloc[-1]['predicted_mean']
                percentage_change = ((final_value - initial_value) / initial_value) * 100
                print(f"Percentage change for {stock_name}: {percentage_change:.2f}%")

                if percentage_change > max_change:
                    max_change = percentage_change
                    max_change_stock = stock_name
            else:
                print(f"Projection file for {stock_name} was not found at {projection_file}.")

        if max_change_stock:
            print(f"The stock with the highest percentage change in the 30-day projection is {max_change_stock} with a change of {max_change:.2f}%")
        else:
            print("No valid projection data found for the given stocks.")

    def run(self):
        while True:
            user_question = input("Please enter your question about stock market volatility: ")
            subject_info = self.subject_finder.classify_question(user_question)

            if 'interrogative pronoun' in subject_info:
                stock_names = ['Nvidia', 'Tesla', 'Amazon', 'Super Micro Computer', 'Carvana']
                training_start_date = '2024-05-10'
                training_end_date = '2024-07-01'
                self.handle_interrogative_query(stock_names, training_start_date, training_end_date)
            else:
                stock_name = subject_info.split("'")[1] if "'" in subject_info else None
                if stock_name:
                    training_start_date = '2024-05-10'
                    training_end_date = '2024-07-01'
                    self.handle_stock_query(stock_name, training_start_date, training_end_date)
                else:
                    print("Unable to identify the subject. Please try again.")

            feedback = input("Is this response satisfactory? (yes/no/exit): ").strip().lower()
            if feedback == 'no':
                print("Thank you for your feedback! We will work on improving the system.")
            elif feedback == 'exit':
                break
            else:
                print("Thank you! Have a great day.")

chatbot = StockManager()
chatbot.run()























































