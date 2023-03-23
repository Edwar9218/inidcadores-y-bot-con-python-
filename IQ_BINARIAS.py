import time
from datetime import datetime
import talib
import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import ta

# Conectar con la API de IQ Option
Iq = IQ_Option("user","password")
Iq.connect()

data =[]
def stochOsc(df, n, k):
    lows = df['min'].rolling(n).min()
    highs = df['max'].rolling(n).max()

    # Calcular %K
    K = ((df['close'] - lows) / (highs - lows)) * 100
    K = K.rolling(k).mean()

    # Calcular %D
    D = K.rolling(5).mean()

    return pd.DataFrame({'K': K, 'D': D})


while True:

    """calcular el indicador MEDIA MOVIL  5 TEMPORALIDA 200 segundos """
    goal = "GBPUSD"
    candles = Iq.get_candles(goal, 5, 1000, time.time())
    close = np.array([candle['close'] for candle in candles])
    sma = talib.SMA(close, timeperiod=200)
    df = pd.DataFrame(candles)
    dato_close_d = df.iloc[-1]['close']
    md = 'arriba' if sma[-1] > dato_close_d else 'abajo'


    """calcular el indicador Stochastic Oscillator  5 segundos """
    stochOsc_indicator = stochOsc(df, 50, 14)
    dato_k = float('%.2f' % stochOsc_indicator.iloc[-1]['K'])
    dato_D = float('%.2f' % stochOsc_indicator.iloc[-1]['D'])
    dato_KD = 'positivo' if dato_k > dato_D else 'negativo'


    """calcular el indicador Stochastic Oscillator 15 minutos  segundos """
    candles1 = Iq.get_candles(goal, 900, 1000, time.time())
    df1 = pd.DataFrame(candles1)
    stochOsc_indicator = stochOsc(df1, 50, 14)
    dato_k_15 = float('%.2f' % stochOsc_indicator.iloc[-1]['K'])
    dato_D_15 = float('%.2f' % stochOsc_indicator.iloc[-1]['D'])
    dato_15 = 'subiendo' if dato_k_15 > dato_D_15 else 'bajando'


    """calcular el indicador MACD 10 minutos  segundos """
    candles11 = Iq.get_candles(goal, 600, 1000, time.time())
    df11 = pd.DataFrame(candles11)
    macd = ta.trend.MACD(df11['close'])
    macd1 = df11['macd'] = macd.macd().iloc[-1]
    cadena1 = float("{:.5f}".format(macd1))

    if cadena1 > 0.00020:
        macd_10 ='sube'
    elif cadena1 < -0.00020:
        macd_10 ='baja'
    else:
        macd_10 = 'neutro'

    ''' PRINT '''
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -  M_D: {md} - dato_K : {dato_k}"
        f" - dato_D : {dato_D}- dato_KD : {dato_KD}- dato_k_15 : {dato_15}- macd_10 : {macd_10} "
        f"-T - + =+ : {float('%.2f' % sum(data)),data.count(-1),data.count(0.8922000000000001),len(data)}")


    if md == 'abajo' and dato_k < 20 and dato_D < 20  and macd_10== 'sube':

        while True:
            """calcular el indicador MEDIA MOVIL  5 TEMPORALIDA 200 segundos """
            goal = "GBPUSD"
            candles = Iq.get_candles(goal, 5, 1000, time.time())
            close = np.array([candle['close'] for candle in candles])
            sma = talib.SMA(close, timeperiod=200)
            df = pd.DataFrame(candles)
            dato_close_d = df.iloc[-1]['close']
            md = 'arriba' if sma[-1] > dato_close_d else 'abajo'

            """calcular el indicador Stochastic Oscillator  5 segundos """
            stochOsc_indicator = stochOsc(df, 50, 14)
            dato_k = float('%.2f' % stochOsc_indicator.iloc[-1]['K'])

            """calcular el indicador MACD 5   segundos """
            macd = ta.trend.MACD(df['close'])
            macd1 = df['macd'] = macd.macd().iloc[-1]
            cadena1 = float("{:.5f}".format(macd1))


            if dato_k > 30 and  cadena1 > 0.00002:

                ACTIVES = goal
                duration = 1  # minuto 1 or 5
                amount = 1
                action = "call"  # put
                _, id = (Iq.buy_digital_spot(ACTIVES, amount, action, duration))
                print(id)
                if id != "error":
                    while True:
                        check, win = Iq.check_win_digital_v2(id)
                        if check == True:
                            break
                    if win < 0:
                        print(" compra - Has perdido " + str(win) + "$")
                        data.append(win)
                        break
                    else:
                        print("compra - Has ganado " + str(win) + "$")
                        data.append(win)
                        break
                else:
                    print("Porfavor prueba otra vez")
                    break
            else:
                print('Esperando dato_KD para comprar', md)
                if md == 'arriba':
                    break

    if md == 'arriba' and dato_k > 80 and dato_D > 80  and macd_10 == 'baja':

        while True:
            """calcular el indicador MEDIA MOVIL  5 TEMPORALIDA 200 segundos """
            goal = "GBPUSD"
            candles = Iq.get_candles(goal, 5, 1000, time.time())
            close = np.array([candle['close'] for candle in candles])
            sma = talib.SMA(close, timeperiod=200)
            df = pd.DataFrame(candles)
            dato_close_d = df.iloc[-1]['close']
            md = 'arriba' if sma[-1] > dato_close_d else 'abajo'

            """calcular el indicador Stochastic Oscillator  5 segundos """
            stochOsc_indicator = stochOsc(df, 50, 14)
            dato_k = float('%.2f' % stochOsc_indicator.iloc[-1]['K'])

            """calcular el indicador MACD 5   segundos """
            macd = ta.trend.MACD(df['close'])
            macd1 = df['macd'] = macd.macd().iloc[-1]
            cadena1 = float("{:.5f}".format(macd1))


            if dato_k < 70 and  cadena1 < -0.00002:

                ACTIVES = goal
                duration = 1  # minuto 1 or 5
                amount = 1
                action = "put"  # put
                _, id = (Iq.buy_digital_spot(ACTIVES, amount, action, duration))
                print(id)
                if id != "error":
                    while True:
                        check, win = Iq.check_win_digital_v2(id)
                        if check == True:
                            break
                    if win < 0:
                        print("venta - Has perdido " + str(win) + "$")
                        data.append(win)
                        break
                    else:
                        print("venta - Has ganado " + str(win) + "$")
                        data.append(win)
                        break
                else:
                    print("Porfavor prueba otra vez")
                    break
            else:
                print('Esperando dato_KD para vender', md)
                if md == 'abajo':
                    break

    time.sleep(1)

