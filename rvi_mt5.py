import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt

import time

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)


mt5.initialize()

print("Inicio")
while True:
    time.sleep(2)

    # solicitamos 10 barras de GBPUSD D1 desde el día actual
    rates = mt5.copy_rates_from_pos("EURUSD",mt5.TIMEFRAME_M5,0,100)

    # creamos un DataFrame de los datos obtenidos
    rates_frame = pd.DataFrame(rates)
    # convertimos la hora en segundos al formato datetime
    rates_frame['time'] = pd.to_datetime(rates_frame['time'],unit='s')

    # mostramos los datos
    print("\nMostramos el frame de datos con la información")


    aapl = rates_frame
    aapl.tail()


    # RELATIVE VIGOR INDEX CALCULATION

    def get_rvi(open,high,low,close,lookback):
        a = close - open
        b = 2 * (close.shift(2) - open.shift(2))
        c = 2 * (close.shift(3) - open.shift(3))
        d = close.shift(4) - open.shift(4)
        numerator = a + b + c + d

        e = high - low
        f = 2 * (high.shift(2) - low.shift(2))
        g = 2 * (high.shift(3) - low.shift(3))
        h = high.shift(4) - low.shift(4)
        denominator = e+f+g+h

        numerator_sum = numerator.rolling(4).sum()
        denominator_sum = denominator.rolling(4).sum()
        rvi = (numerator_sum / denominator_sum).rolling(lookback).mean()
        rv = (numerator_sum / denominator_sum).rolling(lookback).mean()

        rvi1 = 2 * rvi.shift(1)
        rvi2 = 2 * rvi.shift(2)
        rvi3 = rvi.shift(3)
        rvi_signal = (rvi + rvi1 + rvi2 + rvi3) / 6

        a1 = rv.ewm(span=13,adjust=False,min_periods=8).mean()
        b1 = rv.ewm(span=8,adjust=False,min_periods=5).mean()
        c1 = rv.ewm(span=5,adjust=False,min_periods=3).mean()

        return a1, b1, c1


    aapl['rvi'],aapl['signal_line'],aapl['signa_line'] = get_rvi(aapl['open'],aapl['high'],aapl['low'],aapl['close'],50)

    aapl = aapl.dropna()


    list1_of_single_column =    aapl['rvi'].tolist()
    last1 = list1_of_single_column.pop()
    last1 = float('%.2f' % last1)
    list2_of_single_column = aapl['signal_line'].tolist()
    last2 = list2_of_single_column.pop()
    last2 = float('%.2f' % last2)
    list3_of_single_column = aapl['signa_line'].tolist()
    last3 = list3_of_single_column.pop()
    last3 = float('%.2f' % last3)


    print(last1,last2,last3)



    ax1 = plt.subplot2grid((11,1),(0,0),rowspan=5,colspan=1)
    ax2 = plt.subplot2grid((11,1),(6,0),rowspan=5,colspan=1)
    ax1.plot(aapl['close'],linewidth=2.5)
    ax1.set_title('EURUSD CLOSING PRICES')
    ax2.plot(aapl['rvi'],linewidth=2,color='orange',label='RVI LINE')
    ax2.plot(aapl['signal_line'],linewidth=2,color='#BA5FE3',label='SIGNAL LINE')
    ax2.plot(aapl['signa_line'],linewidth=2,color='blue',label='SIGNAL LINE')
    ax2.legend()
    ax2.set_title('EURUSD RVI 10')
    plt.show()


