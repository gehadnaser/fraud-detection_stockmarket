import numpy as np, pandas as pd
import math
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})


# Import data
def Read(name):
    df = pd.read_csv(name + '.csv')
    # get the Volume colume length
    row_count=len(df)-1
    #divide the length into equal haves
    half_rowcount=row_count/2
    # round up the length in case of float
    count = math.ceil(half_rowcount)
    # Create Training and Test
    train = df.Volume[:count]
    test = df.Volume[count:]


    # 1,1,1 ARIMA Model
    model = ARIMA(df.Volume, order=(1, 1, 1))
    model_fit = model.fit(disp=0)
    #print(model_fit.summary())


    # Build Model
    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit(disp=-1)
    print(fitted.summary())
    # Forecast
    fc, se, conf = fitted.forecast(count, alpha=0.05)  # 95% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    result = adfuller(df['Volume'],autolag='AIC')
    if result[1] > 0.05:
        print("fraud ")
    else:
        print("not fraud")

Read('foodico')


