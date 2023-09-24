import os
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, blocking, project_folder, short_name, currency, stock_ticker):
        self.blocking = blocking
        self.project_folder = project_folder
        self.short_name = short_name
        self.currency = currency
        self.stock_ticker = stock_ticker

    def plot_histogram_data_split(self, training_data, test_data, validation_date):
        print("plotting Data and Histogram")
        plt.figure(figsize=(12, 5))
        plt.plot(training_data.close, color='green')
        plt.plot(test_data.close, color='red')
        plt.ylabel('Price [' + self.currency + ']')
        plt.xlabel("Date")
        plt.legend(["Training Data", "Validation Data >= " + validation_date.strftime("%Y-%m-%d")])
        plt.title(self.short_name)
        plt.savefig(os.path.join(self.project_folder, 'NIFTY_price.png'))

        fig, ax = plt.subplots()
        training_data.hist(ax=ax)
        fig.savefig(os.path.join(self.project_folder, 'NIFTY_hist.png'))

        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_loss(self, history):
        print("plotting loss")
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss/Validation Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.project_folder, 'loss.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_mse(self, history):
        print("plotting MSE")
        plt.plot(history.history['MSE'], label='MSE')
        plt.plot(history.history['val_MSE'], label='val_MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE/Validation MSE')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.project_folder, 'MSE.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)

    def project_plot_predictions(self, price_predicted, test_data):
        print("plotting predictions")
        plt.figure(figsize=(14, 5))
        plt.plot(price_predicted[self.stock_ticker + '_predicted'], color='red', label='Predicted NIFTY price')
        plt.plot(test_data.close, color='green', label='Actual NIFTY price')
        plt.xlabel('Time')
        plt.ylabel('Price [' + self.currency + ']')
        plt.legend()
        plt.title('Prediction')
        plt.savefig(os.path.join(self.project_folder, 'NIFTY_prediction.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)
