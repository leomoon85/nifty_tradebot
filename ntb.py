import os
import secrets
import pandas as pd
import argparse
from datetime import datetime

from ntb_prediction_class import StockPrediction
from ntb_get_history import StockData
from ntb_plotter import Plotter
from ntb_lstm_agent import LongShortTermMemory
from ntb_readme_gen import ReadmeGenerator


def train_LSTM_network(stock):
    data = StockData(stock)
    plotter = Plotter(True, stock.get_project_folder(), data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(stock.get_time_steps(), stock.get_project_folder())
    plotter.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())
    lstm = LongShortTermMemory(stock.get_project_folder())
    model = lstm.create_model(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train, epochs=stock.get_epochs(), batch_size=stock.get_batch_size(), validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])
    print("saving weights")
    model.save(os.path.join(stock.get_project_folder(), 'model_weights.keras'))

    plotter.plot_loss(history)
    plotter.plot_mse(history)

    print("display the content of the model")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    print("plotting prediction results")
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(stock.get_project_folder(), 'predictions.csv'))

    test_predictions_baseline.rename(columns={0: stock.get_ticker() + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)
    test_predictions_baseline.index = test_data.index
    plotter.project_plot_predictions(test_predictions_baseline, test_data)

    generator = ReadmeGenerator(stock.get_github_url(), stock.get_token(), data.get_stock_short_name())
    generator.write()

    print("prediction is finished")

# The Main function requires 3 major variables
# 1) Ticker => defines the short code of a stock
# 2) Start date => Date when we want to start using the data for training, usually the first data point of the stock
# 3) Validation date => Date when we want to start partitioning our data from training to validation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("parsing arguments"))
    parser.add_argument("-ticker", default="NIFTY")
    parser.add_argument("-start_date", default="2022-11-01")
    parser.add_argument("-validation_date", default="2023-09-01")
    parser.add_argument("-epochs", default="100")
    parser.add_argument("-batch_size", default="10")
    parser.add_argument("-time_steps", default="3")
    parser.add_argument("-github_url", default="https://github.com/leomoon85/nifty_tradebot.git")
    
    args = parser.parse_args()
    
    STOCK_TICKER = args.ticker
    STOCK_START_DATE = pd.to_datetime(args.start_date)
    STOCK_VALIDATION_DATE = pd.to_datetime(args.validation_date)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    TIME_STEPS = int(args.time_steps)
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
    GITHUB_URL = args.github_url
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Test Run Folder: ' + TOKEN)
    # create project run folder
    PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)
    if not os.path.exists(PROJECT_FOLDER):
        os.makedirs(PROJECT_FOLDER)

    stock_prediction = StockPrediction(STOCK_TICKER, 
                                       STOCK_START_DATE, 
                                       STOCK_VALIDATION_DATE, 
                                       PROJECT_FOLDER, 
                                       GITHUB_URL,
                                       EPOCHS,
                                       TIME_STEPS,
                                       TOKEN,
                                       BATCH_SIZE)
    # Execute Deep Learning model
    train_LSTM_network(stock_prediction)
