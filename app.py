from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    keyword = request.form["keyword"]
    yf.pdr_override()
    company = keyword

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)

    data = pdr.get_data_yahoo(company, start, end)
    # Create first plot
    fig, ax = plt.subplots(figsize=(12, 6))
    data['Adj Close'].plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(company)
    # save first plot to bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    # encode the first plot image in base64
    plot_image = base64.b64encode(buffer.getvalue()).decode()

    # Create second plot
    ma100 = data.Close.rolling(100).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ma100, 'r')
    ax.plot(data.Close)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(company + ' (MA100)')

    # save second plot to bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # encode the second plot image in base64
    ma100_plot_image = base64.b64encode(buffer.getvalue()).decode()

    # Create third plot
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ma100, 'r')
    ax.plot(ma200, 'g')
    ax.plot(data.Close)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(company + ' (MA100 & MA200)')

    # save third plot to bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # encode the third plot image in base64
    ma100_ma200_plot_image = base64.b64encode(buffer.getvalue()).decode()

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
    temp1 = int(len(data)*0.70)
    temp2 = int(len(data))
    data_testing = pd.DataFrame(data['Close'][temp1:temp2])
    # data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predict = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predict = y_predict * scale_factor
    y_test = y_test * scale_factor
    # Create fourth plot

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, 'r', label='Original price')
    ax.plot(y_predict, 'g', label='Predicted price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(' Accuracy of prediction')

    # save fourth plot to bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # encode the fourth plot image in base64
    final_plot = base64.b64encode(buffer.getvalue()).decode()

    start = dt.datetime(2022, 1, 1)
    end = dt.datetime.now()

    prediction_days = 100
    test_data = pdr.get_data_yahoo(company, start, end)
    # actual_price = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    ln = len(total_dataset)-len(test_data)-prediction_days
    model_inputs = total_dataset[ln:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    ln = len(model_inputs)+1-prediction_days
    real_data = [model_inputs[ln:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    reshape_val = (real_data.shape[0], real_data.shape[1], 1)
    real_data = np.reshape(real_data, reshape_val)

    prediction = model.predict(real_data)
    prd = scaler.inverse_transform(prediction)
    td = data.describe()
    kw = keyword
    fn = "result.html"
    pti = plot_image
    mpti = ma100_plot_image
    mpti2 = ma100_ma200_plot_image
    return render_template(fn, predict=prd, keyword=kw, table_data=td,
                           plot_image=pti, ma100_plot_image=mpti,
                           ma100_ma200_plot_image=mpti2, final_plot=final_plot)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
