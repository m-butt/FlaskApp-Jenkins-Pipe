import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr


def company_info(ticker):
    yf.pdr_override()
    company = ticker

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)

    data = pdr.get_data_yahoo(company, start, end)

    return data
