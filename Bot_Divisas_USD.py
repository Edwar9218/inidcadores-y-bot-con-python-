import MetaTrader5 as mt5
import pandas as pd
import time
from operator import itemgetter
import numpy as np
from time import localtime

import os

mt5.initialize()
t = 1
n = 3
H = 1
f = 1
orden= []
print("Inicio")
while n > 0:


    time.sleep(2)
    os.system("cls")


    class CierreApertura:
        def __init__(self, divisa, PeriodoTiempo, N_Dias, N_Periodos, alto_bajo):
            self.divisa = divisa
            self.PeriodoTiempo = PeriodoTiempo
            self.N_Dias = N_Dias
            self.N_Periodos = N_Periodos
            self.alto_bajo = alto_bajo

        def cierre_apertura(self):
            datagbp5 = pd.DataFrame()
            rates = mt5.copy_rates_from_pos(self.divisa, self.PeriodoTiempo, self.N_Dias, self.N_Periodos)
            datagbp5[self.divisa] = [y[self.alto_bajo] for y in rates]
            gbp3 = datagbp5.iloc[0, 0]

            return gbp3



    class Medias:

        def __init__(self, mm_divisa, PeriodoTiempo, N_Dias, N_Periodos, alto_bajo):
            self.mm_divisa = mm_divisa

            self.PeriodoTiempo = PeriodoTiempo
            self.N_Dias = N_Dias
            self.N_Periodos = N_Periodos
            self.alto_bajo = alto_bajo

        def media_moviles(self):

            datagb55 = pd.DataFrame()
            listas = []
            for i in range(self.N_Periodos):
                rates = mt5.copy_rates_from_pos(self.mm_divisa, self.PeriodoTiempo, self.N_Dias, self.N_Periodos)
                datagb55[self.mm_divisa] = [y['close'] for y in rates]
                listas.append(datagb55.iloc[i, 0])

            def sumar_lista(lista):
                suma = 0
                for numero in lista:
                    suma += numero
                return suma

            numeros = listas
            resultado = float(sumar_lista(numeros) / self.N_Periodos)

            return resultado


    class Precio:


        def __init__(self, precio):
            self.precio = precio

        def precios(self):
            tick = mt5.symbol_info_tick(self.precio)._asdict()
            gbp2 = float(f"{tick['ask']}")


            return gbp2

    class Buysell_Cierre:

        def __init__(self, compra, tp, sl, lt):
            self.compra = compra
            self.tp = tp
            self.sl = sl
            self.lt = lt



        def comprar(self):

            def comprargbp():

                symbol = (self.compra)
                symbol_info = mt5.symbol_info(symbol)

                mt5.symbol_select(symbol, True)

                lot = self.lt
                point = mt5.symbol_info(symbol).point
                price = mt5.symbol_info_tick(symbol).ask
                deviation = 0
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": price,
                    "sl": price - self.sl * point,
                    "tp": price + self.tp * point,
                    "deviation": deviation,
                    "magic": 234000,
                    "comment": "python script open",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,

                }

                result = mt5.order_send(request)
                # comprobamos el resultado de la ejecución
                print(
                    "1. order_send(): by {} {} lots at {} with deviation={} points compra".format(symbol, lot, price, deviation))

            return comprargbp()

        def vender(self):

            def ventagbp():
                symbol = self.compra
                symbol_info = mt5.symbol_info(symbol)

                mt5.symbol_select(symbol, True)

                lot = self.lt
                point = mt5.symbol_info(symbol).point
                price = mt5.symbol_info_tick(symbol).ask
                deviation = 0
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": price + self.sl * point,
                    "tp": price - self.tp * point,
                    "deviation": deviation,
                    "magic": 234000,
                    "comment": " python script open",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,

                }

                result = mt5.order_send(request)
                # comprobamos el resultado de la ejecución
                print(
                    "1. order_send(): by {} {} lots at {} with deviation={} points  venta".format(symbol, lot, price, deviation))

            return ventagbp()


        def cierre(self):

            def positions_get(symbol=None):
                if (symbol is None):
                    res = mt5.positions_get()
                else:
                    res = mt5.positions_get(symbol=symbol)

                if res is not None and res != ():
                    df = pd.DataFrame(list(res), columns=res[0]._asdict().keys())
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    return df

                return pd.DataFrame()

            def close_position(deal_id):
                open_positions = positions_get()
                open_positions = open_positions[open_positions['ticket'] == deal_id]
                order_type = open_positions["type"][0]
                symbol = open_positions['symbol'][0]
                volume = open_positions['volume'][0]

                if (order_type == mt5.ORDER_TYPE_BUY):
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).ask

                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(volume),
                    "type": order_type,
                    "position": deal_id,
                    "price": price,
                    "magic": 234000,
                    "comment": "Close trade",
                    "type_time":mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(close_request)

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Failed to close order :(")
                else:
                    print("Order successfully closed!")

            def close_positons_by_symbol(symbol):
                open_positions = positions_get(symbol)
                open_positions['ticket'].apply(lambda x: close_position(x))

            return close_positons_by_symbol(self.compra)

        def entrada (self):

            gbp_orders = mt5.positions_get(group="*USD*")
            if gbp_orders is None:
                print("No orders with group=\"*USD*\", error code={}".format(mt5.last_error()))
            else:

                # mostramos estas órdenes en forma de recuadro con la ayuda de pandas.DataFrame
                df = pd.DataFrame(list(gbp_orders), columns=gbp_orders[0]._asdict().keys())
                df.drop(['ticket'], axis=1, inplace=True)
                li = dict(df)
                price_pen = float(li['price_open'])
                tp_price = float(li['tp'])

                total_s = (tp_price / price_pen) * 100 - 100
                sellbuy = float('%.2f' % total_s)


                return  sellbuy

    class Indicador_macd:
        def __init__(self, divisas, alto_mediobajo):
            self.divisas = divisas
           #self.PeriodoTiempo = PeriodoTiempo
            self.alto_mediobajo = alto_mediobajo


        def indicador (self):

            # solicitamos 10 barras de GBPUSD D1 desde el día actual
            rates = mt5.copy_rates_from_pos(self.divisas,  mt5.TIMEFRAME_H12, 0, 100)

            # creamos un DataFrame de los datos obtenidos
            rates_frame = pd.DataFrame(rates)
            # convertimos la hora en segundos al formato datetime
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

            df = rates_frame

            k = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
            # Get the 12-day EMA of the closing price
            d = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()
            # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
            macd = k - d
            # Get the 9-Day EMA of the MACD for the Trigger line
            macd_s = macd.ewm(span=3, adjust=False, min_periods=3).mean()
            # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
            macd_h = macd - macd_s
            # Add all of our new values for the MACD to the dataframe
            df['macd'] = df.index.map(macd)
            df['macd_h'] = df.index.map(macd_h)
            df['macd_s'] = df.index.map(macd_s)
            # View our data
            pd.set_option("display.max_columns", None)

            list_of_single_column = df['macd'].tolist()

            last = list_of_single_column.pop()
            last = float('%.6f' % last)

            list_of_single_column = df['macd_s'].tolist()

            last1 = list_of_single_column.pop()
            last1 = float('%.6f' % last1)

            masmen1 = (last / 0.000008) * 100 - 100
            aud61 = float('%.6f' % masmen1)

            masmen9 = (last1 / 0.000008) * 100 - 100
            aud611 = float('%.6f' % masmen9)

            if aud61 > aud611:
                ad = ('positivo')
            else:
                ad = ('negativo')

            lis = [last, last1, ad]
            return (lis[self.alto_mediobajo])


    class Indicador_RSI:

        def __init__(self, mm_divisa, PeriodoTiempo, N_Dias, N_Periodos, alto_bajo,rsi_periodo):
            self.mm_divisa = mm_divisa

            self.PeriodoTiempo = PeriodoTiempo
            self.N_Dias = N_Dias
            self.N_Periodos = N_Periodos
            self.alto_bajo = alto_bajo
            self.rsi_periodo = rsi_periodo


        def indicador_R(self):

            datagb55 = pd.DataFrame()
            listas = []
            for i in range(self.N_Periodos):
                rates = mt5.copy_rates_from_pos(self.mm_divisa, self.PeriodoTiempo, self.N_Dias, self.N_Periodos)
                datagb55[self.mm_divisa] = [y[self.alto_bajo] for y in rates]
                listas.append(datagb55.iloc[i, 0])

                def RSI(t, periods):
                    length = len(t)
                    rsies = [np.nan] * length
                    # La longitud de los datos no excede el período y no se puede calcular;
                    if length <= periods:
                        return rsies
                        # Utilizado para cálculos rápidos;
                    up_avg = 0
                    down_avg = 0

                    # Primero calcule el primer RSI, use los períodos anteriores + 1 dato para formar una secuencia de períodos;
                    first_t = t[:periods + 1]
                    for i in range(1, len(first_t)):
                        # Precio aumentado;
                        if first_t[i] >= first_t[i - 1]:
                            up_avg += first_t[i] - first_t[i - 1]
                            # caída de los precios;
                        else:
                            down_avg += first_t[i - 1] - first_t[i]
                    up_avg = up_avg / periods
                    down_avg = down_avg / periods
                    rs = up_avg / down_avg
                    rsies[periods] = 100 - (100 / (1 + rs))

                    # Lo siguiente utilizará cálculo rápido;
                    for j in range(periods + 1, length):
                        up = 0
                        down = 0
                        if t[j] >= t[j - 1]:
                            up = t[j] - t[j - 1]
                            down = 0
                        else:
                            up = 0
                            down = t[j - 1] - t[j]
                            # Fórmula de cálculo similar a la media móvil;
                        up_avg = (up_avg * (periods - 1) + up) / periods
                        down_avg = (down_avg * (periods - 1) + down) / periods
                        rs = up_avg / down_avg
                        rsies[j] = 100 - 100 / (1 + rs)
                    return rsies

                rsi = RSI(listas, periods=self.rsi_periodo)
                last = rsi.pop()
                last = float('%.2f' % last)

                if last > 50:
                    ad = ('positivo')

                else:
                    ad = ('negativo')

            return ad

    class Indicador_Adx:
        def __init__(self,divisas):
            self.divisas = divisas


        def indicador(self):
            # solicitamos 10 barras de GBPUSD D1 desde el día actual
            rates = mt5.copy_rates_from_pos(self.divisas,mt5.TIMEFRAME_M30,0,100)

            # creamos un DataFrame de los datos obtenidos
            rates_frame = pd.DataFrame(rates)
            # convertimos la hora en segundos al formato datetime
            rates_frame['time'] = pd.to_datetime(rates_frame['time'],unit='s')

            aapl = rates_frame

            np.seterr(divide='ignore',invalid='ignore')
            aapl['open'] = aapl.open * aapl['close'] / aapl['close']
            aapl['high'] = aapl.high * aapl['close'] / aapl['close']
            aapl['low'] = aapl.low * aapl['close'] / aapl['close']
            aapl.dropna(inplace=True)
            from ta.trend import ADXIndicator

            adxI = ADXIndicator(aapl['high'],aapl['low'], aapl['close'],14,False)
            aapl['pos_directional_indicator'] = adxI.adx_pos()
            aapl['neg_directional_indicator'] = adxI.adx_neg()
            aapl['adx'] = adxI.adx()

            list_of_single_column = aapl['adx'].tolist()
            last = list_of_single_column.pop()
            last = float('%.2f' % last)

            list1_of_single_column = aapl['pos_directional_indicator'].tolist()
            last1 = list1_of_single_column.pop()
            last1 = float('%.2f' % last1)

            list2_of_single_column = aapl['neg_directional_indicator'].tolist()
            last2 = list2_of_single_column.pop()
            last2 = float('%.2f' % last2)

            masmen1 = (last1 / 0.08) * 100 - 100
            aud61 = float('%.6f' % masmen1)

            masmen9 = (last2 / 0.08) * 100 - 100
            aud611 = float('%.6f' % masmen9)

            if aud61 > aud611 and last1 > last:
                ad = ('positivo')

            elif aud61 < aud611 and last2 > last:
                ad = ('negativo')

            else:
                ad = ('neutro')

            return (ad)

    class PivotDia:
        def __init__(self, divisa, PeriodoTiempo,  alto_bajo):
            self.divisa = divisa
            self.PeriodoTiempo = PeriodoTiempo
            self.alto_bajo = alto_bajo

        def pivot(self):
            datagbp5 = pd.DataFrame()
            rates = mt5.copy_rates_from_pos(self.divisa, self.PeriodoTiempo, 1, 1)
            datagbp5[self.divisa] = [y['high'] for y in rates]
            gbp1 = datagbp5.iloc[0, 0]

            datagbp = pd.DataFrame()
            rates = mt5.copy_rates_from_pos(self.divisa, self.PeriodoTiempo, 1, 1)
            datagbp[self.divisa] = [y['low'] for y in rates]
            gbp2 = datagbp.iloc[0, 0]

            datagb = pd.DataFrame()
            rates = mt5.copy_rates_from_pos(self.divisa, self.PeriodoTiempo, 1, 1)
            datagb[self.divisa] = [y['close'] for y in rates]
            gbp3 = datagb.iloc[0, 0]


            r2 = (gbp3 + (gbp1 - gbp2) * .500)
            s2 = (gbp3 - (gbp1 - gbp2) * .500)
            lis = [r2, s2]
            return (lis[self.alto_bajo])


    class CierreApertura1:
        def __init__(self,PeriodoTiempo,N_Dias,N_Periodos,alto_bajo,al):

            self.PeriodoTiempo = PeriodoTiempo
            self.N_Dias = N_Dias
            self.N_Periodos = N_Periodos
            self.alto_bajo = alto_bajo
            self.al = al

        def cierre_apertura1(self):

            if self.al == 1:
                divisa1usd = 'USD'
                divisasusd = ['JPY','EUR','GBP','CAD','CHF','AUD','NZD']
            elif self.al == 2:
                divisa1usd = 'EUR'
                divisasusd = ['JPY','USD','GBP','CAD','CHF','AUD','NZD']


            elif self.al == 3:
                divisa1usd = 'GBP'
                divisasusd = ['JPY','USD','EUR','CAD','CHF','AUD','NZD']

            elif self.al == 4:
                divisa1usd = 'NZD'
                divisasusd = ['JPY','EUR','GBP','CAD','CHF','AUD','USD']

            elif self.al == 5:
                divisa1usd = 'AUD'
                divisasusd = ['JPY','USD','EUR','CAD','CHF','GBP','NZD']

            elif self.al == 6:
                divisa1usd = 'CHF'
                divisasusd = ['JPY','USD','EUR','AUD','CAD','GBP','NZD']

            elif self.al == 7:
                divisa1usd = 'CAD'
                divisasusd = ['JPY','USD','EUR','AUD','CHF','GBP','NZD']


            elif self.al == 8:
                divisa1usd = 'JPY'
                divisasusd = ['CHF','USD','EUR','AUD','CAD','GBP','NZD']

            listasusd = []

            for divisausd in divisasusd:
                resu_diviusd = divisa1usd + divisausd

                selected = mt5.symbol_select(resu_diviusd,True)
                if not selected:
                    resu_diviusd = divisausd + divisa1usd
                    a = 1
                else:

                    resu_diviusd
                    a = 0

                datagbp5 = pd.DataFrame()
                rates = mt5.copy_rates_from_pos(resu_diviusd,self.PeriodoTiempo,self.N_Dias,self.N_Periodos)
                datagbp5[resu_diviusd] = [y[self.alto_bajo] for y in rates]
                usdcad1usd = datagbp5.iloc[0,0]

                tick = mt5.symbol_info_tick(resu_diviusd)._asdict()
                usdcad3usd = float(f"{tick['ask']}")

                if a == 1:

                    usdcad4usd = (usdcad1usd / usdcad3usd) * 100 - 100
                else:

                    usdcad4usd = (usdcad3usd / usdcad1usd) * 100 - 100

                usdcad5usd = float('%.2f' % usdcad4usd)
                listasusd.append(usdcad5usd)

                meanusd = sum(listasusd) / len(listasusd)
                meanusd = float('%.2f' % meanusd)

            return meanusd


    # 1:'USD',2: 'EUR', 3:'GBP',4:'NZD' 5:'AUD' 6:'CHF' 7:'CAD'   8:'JPY'
    D1 = mt5.TIMEFRAME_D1

    usaD11 = CierreApertura1(D1,0,1,'open',1)
    usaD1 = usaD11.cierre_apertura1()
    eurD11 = CierreApertura1(D1,0,1,'open',2)
    eurD1 = eurD11.cierre_apertura1()
    gbpD11 = CierreApertura1(D1,0,1,'open',3)
    gbpD1 = gbpD11.cierre_apertura1()
    nzdD11 = CierreApertura1(D1,0,1,'open',4)
    nzdD1 = nzdD11.cierre_apertura1()
    audD11 = CierreApertura1(D1,0,1,'open',5)
    audD1 = audD11.cierre_apertura1()
    chfD11 = CierreApertura1(D1,0,1,'open',6)
    chfD1 = chfD11.cierre_apertura1()
    cadD11 = CierreApertura1(D1,0,1,'open',7)
    cadD1 = cadD11.cierre_apertura1()
    jpyD11 = CierreApertura1(D1,0,1,'open',8)
    jpyD1 = jpyD11.cierre_apertura1()

    divi = {'USD':usaD1,'EUR':eurD1,'GBP':gbpD1,'NZD':nzdD1,'CAD':cadD1,'CHF':chfD1,'AUD':audD1,
            'JPY':jpyD1}
    divi_asc = sorted(divi.items(),key=itemgetter(1))
    #print(divi_asc)

    divis = {'EUR':eurD1,'GBP':gbpD1,'NZD':nzdD1,'CAD':cadD1,'CHF':chfD1,'AUD':audD1,
            'JPY':jpyD1}
    divis_asc = sorted(divis.items(),key=itemgetter(1))


    las = divi_asc
    la = list(las)
    lst = np.array(la)
    result = np.where(lst == 'USD')
    a = int(result[0])


    if a >= 4:
        usdeur = 'USD'
        last = divis_asc.pop()
        last9 = list(last)
        last2 = usdeur
        last3 = last9.pop(-1)
        last1 = divis_asc[0]
        last1 = list(last1)
        last1 = last1.pop(-2)
        comparar = last1
        divi = last2 + last1
        selected = mt5.symbol_select(divi,True)
        if not selected:
            divi = last1 + last2
        else:
            divi

        cadena = divi
        buysell = cadena.find(last2)
        if buysell == 0:
            buy_sell = 'positivo'
        else:
            buy_sell = 'negativo'
    else:
        usdeur = 'USD'
        last = divis_asc.pop()
        last9 = list(last)
        last2 = last9.pop(-2)

        last1 = usdeur
        comparar = last2
        divi = last2 + last1
        selected = mt5.symbol_select(divi,True)
        if not selected:
            divi = last1 + last2
        else:
            divi

        cadena = divi
        buysell = cadena.find(last2)
        if buysell == 0:
            buy_sell = 'positivo'
        else:
            buy_sell = 'negativo'

    if f > 0:
        cadena = cadena


    else:
        cadena = orden

    print()

    closepeurg = CierreApertura(cadena,mt5.TIMEFRAME_M15,1,1,'close')
    closep1eurg = closepeurg.cierre_apertura()
    # print(closep1eurg)
    closeurg = CierreApertura(cadena,mt5.TIMEFRAME_M15,2,1,'close')
    closurg = closeurg.cierre_apertura()
    '''print(closurg)'''
    boopeneurg = Precio(cadena)
    precioeureurg = boopeneurg.precios()
    mediaseurg = (precioeureurg / closep1eurg) * 100 - 100
    meidacierre = float('%.2f' % mediaseurg)
    velaseurg = (closep1eurg / closurg) * 100 - 100
    velarre = float('%.2f' % velaseurg)
    both1adxgp = Indicador_RSI(cadena,mt5.TIMEFRAME_H4,1,100,'close',14)
    closeadxgp = both1adxgp.indicador_R()
    media02 = Medias(cadena,mt5.TIMEFRAME_M15,0,10,'close')
    media2 = media02.media_moviles()
    media03 = Medias(cadena,mt5.TIMEFRAME_M15,0,2,'close')
    media3 = media03.media_moviles()
    adx = Indicador_Adx(cadena)
    adx0 = adx.indicador()
    closep = PivotDia(cadena,mt5.TIMEFRAME_D1,0)
    closep1 = closep.pivot()
    close = PivotDia(cadena,mt5.TIMEFRAME_D1,1)
    clos = close.pivot()
    medi = (media2 / closep1) * 100 - 100
    meidaclose = float('%.2f' % medi)
    media = (clos / media2) * 100 - 100
    meidaclose1 = float('%.2f' % media)

    if velaseurg > 0.01:
        velarre = 'positivo'

    else:
        velarre = 'negativo'

    if buy_sell == 'positivo':
        bajoalto = CierreApertura(cadena,mt5.TIMEFRAME_D1,1,1,'high')
        closealto = bajoalto.cierre_apertura()
        bajo2alto = CierreApertura(cadena,mt5.TIMEFRAME_D1,2,1,'high')
        closealto2 = bajo2alto.cierre_apertura()
        openalto = CierreApertura(cadena,mt5.TIMEFRAME_D1,0,1,'open')
        closopen = openalto.cierre_apertura()
        mediaopen = ( precioeureurg/ closopen) * 100 - 100
        meidac = float('%.2f' % mediaopen)



        if closealto < precioeureurg and closealto2 < precioeureurg and meidac < 0.90 :
            entrada = 'si'


        else:
            entrada = 'no'



    else:
        bajoalto = CierreApertura(cadena,mt5.TIMEFRAME_D1,1,1,'low')
        closealto = bajoalto.cierre_apertura()
        bajo2alto = CierreApertura(cadena,mt5.TIMEFRAME_D1,2,1,'low')
        closealto2 = bajo2alto.cierre_apertura()
        openalto = CierreApertura(cadena,mt5.TIMEFRAME_D1,0,1,'open')
        closopen = openalto.cierre_apertura()
        mediaopen = (precioeureurg/ closopen) * 100 - 100
        meidac = float('%.2f' % mediaopen)



        if closealto > precioeureurg and closealto2 > precioeureurg and meidac > -0.90 :
            entrada = 'si'

        else:
            entrada = 'no'

    medias = (media3 / media2) * 100 - 100
    meidacierre1 = float('%.2f' % medias)

    asr = time.strftime('%b %d %Y %H:%M')
    print(asr)

    print("R1 [H1] {} S1  {}  ".format(meidaclose,meidaclose1))
    print("VELA  [H1] {} [H-1] {} MEDIA MOVIL {} ".format(meidacierre,velarre,meidacierre1,))
    print("ROTURA [?] {}  USD {}".format(entrada,a))
    print(" DIVISA : {}   BUY - SELL : {}  [RSI ] {} [ADX ] {}".format(cadena,buy_sell,closeadxgp,adx0))

    ordersgp = mt5.positions_get(symbol=cadena)
    entradasgp = len(ordersgp)
    print("Total orders  :",entradasgp)

    if entradasgp == 0:

        if localtime().tm_hour in np.linspace(17, 20, num=7):
             print('IT DOES NOT WORK IN THESE HOURS ')

        if localtime().tm_min in np.linspace(50,60,num=11):
            print('IT DOES NOT WORK IN THESE MINUTES ')


        else:
            for i in [1]:

                if buy_sell == 'positivo' and adx0 == 'positivo' and meidacierre >= 0.05 and meidacierre <= 0.30 \
                        and meidacierre1 >= 0.03 and velarre == 'negativo' and entrada == 'si':
                    com = Buysell_Cierre(cadena,2500,200,0.01)
                    compra = com.comprar()
                    n -= 1
                    f -= 1
                    orden = cadena

                elif buy_sell == 'negativo' and adx0 == 'negativo' and meidacierre <= -0.05 and meidacierre >= -0.30 \
                        and meidacierre1 <= -0.03 and velarre == 'positivo' and entrada == 'si':
                    com1 = Buysell_Cierre(cadena,2500,200,0.01)
                    vender = com1.vender()
                    n -= 1
                    f -= 1
                    orden = cadena

    if entradasgp == 1:

        while t > 0:
            time.sleep(2)
            os.system("cls")
            asr = time.strftime('%b %d %Y %H:%M')

            print("VELA  [H1] {} MEDIA MOVIL {}  ".format(meidacierre,meidacierre1))
            print(" DIVISA : {}   BUY - SELL : {}  [RSI ] {}".format(cadena,buy_sell,closeadxgp))

            media02 = Medias(cadena,mt5.TIMEFRAME_M15,0,10,'close')
            media2 = media02.media_moviles()
            media03 = Medias(cadena,mt5.TIMEFRAME_M15,0,2,'close')
            media3 = media03.media_moviles()

            medias = (media3 / media2) * 100 - 100
            meidacierre1 = float('%.2f' % medias)
            print(meidacierre1)

            entradas = Buysell_Cierre(cadena,1500,200,0.01)
            entrar = entradas.entrada()

            print("WAITING FOR THE CLOSURE")

            if entrar > 0.03 and meidacierre1 < -0.02:
                cerrar = Buysell_Cierre(cadena,1500,200,0.01)
                cerrar1 = cerrar.cierre()
                t -= 1
                os.system('Bot_Divisas_USD.py')

            elif entrar < -0.03 and meidacierre1 > 0.02:
                cerrar = Buysell_Cierre(cadena,300,200,0.01)
                cerrar1 = cerrar.cierre()
                t -= 1
                os.system('Bot_Divisas_USD.py')

# import os
# os.system ('2-03-2022.py')
# meidacierre in np.linspace(0.18, 0.22, num=5)
