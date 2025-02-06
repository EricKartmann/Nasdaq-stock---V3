import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Seguimiento empresas Nasdaq",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Nuevo título
st.title("📊 Seguimiento empresas Nasdaq")

# Lista de símbolos de las principales empresas del NASDAQ
top_companies = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'SOFI': 'SoFi Technologies Inc.',
    'BBAI': 'BigBear.ai Holdings Inc.',
    'RKLB': 'Rocket Lab USA Inc.',
    'COHR': 'Coherent Corp.',
    'PL': 'Planet Labs PBC',
    'QCOM': 'Qualcomm Inc.',
    'BNZI': 'Banzai International Inc.',
    'HUBS': 'HubSpot Inc.',
    'RDW': 'Redwire Corporation',
    'QBTS': 'D-Wave Quantum Inc.',
    'VZLA': 'Vizsla Silver Corp.',
    'HUBC': 'Hub Cyber Security Ltd.',
    'PYPL': 'PayPal Holdings Inc.',
    'LUNR': 'Intuitive Machines Inc.',
    'SPIR': 'Spire Global Inc.',
    'ARBE': 'Arbe Robotics Ltd.',
    'RGTI': 'Rigetti Computing Inc.',
    'IPGP': 'IPG Photonics Corporation',
    'LASE': 'Laser Photonics Corporation',
    'LITE': 'Lumentum Holdings Inc.',
    'ARQQ': 'Arqit Quantum Inc.'
}

def resample_to_4h(data):
    """Convierte los datos a intervalos de 4 horas"""
    data_4h = data.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return data_4h

def format_volume(volume):
    """Formatea el volumen en K, M o B según su tamaño"""
    if volume >= 1_000_000_000:
        return f"{volume/1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    return str(volume)

def calculate_rsi(data, periods=14):
    """Calcula el RSI (Relative Strength Index)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calcula el MACD (Moving Average Convergence Divergence)"""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1], macd, signal

def calculate_bollinger_bands(data, window=20):
    """Calcula las Bandas de Bollinger"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]

def calculate_stochastic(data, k_window=14, d_window=3):
    """Calcula el Oscilador Estocástico"""
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    return k.iloc[-1], d.iloc[-1]

def get_premarket_data(symbol):
    """Obtiene datos del pre-mercado"""
    try:
        stock = yf.Ticker(symbol)
        # Obtener datos del pre-mercado (últimas 4 horas antes de la apertura)
        premarket = stock.history(period='1d', interval='1m', prepost=True)
        if not premarket.empty:
            # Filtrar solo datos del pre-mercado
            current_hour = datetime.now().hour
            if current_hour < 9:  # Si estamos antes de la apertura del mercado
                premarket = premarket[premarket.index.hour < 9]
                if not premarket.empty:
                    return {
                        'price': premarket['Close'].iloc[-1],
                        'change': ((premarket['Close'].iloc[-1] - premarket['Open'].iloc[0]) / premarket['Open'].iloc[0]) * 100,
                        'volume': premarket['Volume'].sum()
                    }
    except Exception:
        pass
    return None

def predict_next_day_movement(hist_4h, macd_line, signal_line, premarket_data=None):
    """Predice la tendencia para el próximo día basado en patrones de 4 horas y datos del pre-mercado"""
    # Análisis de tendencia reciente (últimas 24 horas = 6 períodos de 4h)
    recent_trend = hist_4h['Close'].tail(6).pct_change().mean() * 100
    
    # Momentum del MACD
    macd_momentum = (macd_line.tail(6) - signal_line.tail(6)).mean()
    
    # Volatilidad reciente
    recent_volatility = hist_4h['Close'].tail(6).pct_change().std() * 100
    
    # Fuerza de la tendencia base
    trend_strength = abs(recent_trend) / recent_volatility if recent_volatility != 0 else 0
    
    # Ajustar predicción con datos del pre-mercado si están disponibles
    if premarket_data:
        premarket_trend = premarket_data['change']
        
        # Ajustar la tendencia base con el pre-mercado
        if abs(premarket_trend) > 1:  # Si hay un movimiento significativo en el pre-mercado
            if premarket_trend > 0:
                recent_trend = max(recent_trend, 0) + premarket_trend/2  # Dar más peso al pre-mercado
            else:
                recent_trend = min(recent_trend, 0) + premarket_trend/2
            
            # Ajustar la fuerza de la tendencia
            trend_strength = trend_strength * 1.5 if np.sign(recent_trend) == np.sign(premarket_trend) else trend_strength * 0.8
    
    prediction = {
        'Tendencia': "ALCISTA" if recent_trend > 0 else "BAJISTA",
        'Fuerza': "FUERTE" if trend_strength > 1 else "MODERADA" if trend_strength > 0.5 else "DÉBIL",
        'Volatilidad': f"{recent_volatility:.2f}%",
        'Confianza': "ALTA" if trend_strength > 1.5 else "MEDIA" if trend_strength > 0.75 else "BAJA"
    }
    
    # Añadir información del pre-mercado si está disponible
    if premarket_data:
        prediction['Premarket'] = f"{premarket_data['change']:.2f}%"
    
    return prediction

def get_advanced_signals(price, upper_bb, lower_bb, macd, macd_signal, k, d):
    """Analiza señales de indicadores avanzados"""
    signals = []
    
    # Señales de Bandas de Bollinger
    if price > upper_bb:
        signals.append(("VENTA", "Precio sobre banda superior de Bollinger"))
    elif price < lower_bb:
        signals.append(("COMPRA", "Precio bajo banda inferior de Bollinger"))
    
    # Señales de MACD
    if macd > macd_signal and macd > 0:
        signals.append(("COMPRA", "Cruce alcista del MACD"))
    elif macd < macd_signal and macd < 0:
        signals.append(("VENTA", "Cruce bajista del MACD"))
    
    # Señales del Estocástico
    if k > 80 and d > 80:
        signals.append(("VENTA", "Estocástico en zona de sobrecompra"))
    elif k < 20 and d < 20:
        signals.append(("COMPRA", "Estocástico en zona de sobreventa"))
    
    return signals

def get_recommendation(price_change, rsi, volume_change, sma_ratio, price, upper_bb, lower_bb, macd, macd_signal, k, d):
    """Genera una recomendación basada en todos los indicadores técnicos"""
    signals = []
    
    # Señales básicas
    if price_change > 2:
        signals.append(("VENTA", "Subida fuerte de precio"))
    elif price_change < -2:
        signals.append(("COMPRA", "Caída significativa de precio"))
        
    if rsi > 70:
        signals.append(("VENTA", "RSI en sobrecompra"))
    elif rsi < 30:
        signals.append(("COMPRA", "RSI en sobreventa"))
        
    if volume_change > 50:
        signals.append(("COMPRA" if price_change > 0 else "VENTA", "Alto volumen de operaciones"))
        
    if sma_ratio > 1.05:
        signals.append(("COMPRA", "Tendencia alcista fuerte"))
    elif sma_ratio < 0.95:
        signals.append(("VENTA", "Tendencia bajista fuerte"))
    
    # Añadir señales avanzadas
    signals.extend(get_advanced_signals(price, upper_bb, lower_bb, macd, macd_signal, k, d))
    
    if not signals:
        return "MANTENER", "Sin señales claras de trading"
    
    # Contar señales de compra y venta
    buy_signals = len([s for s in signals if s[0] == "COMPRA"])
    sell_signals = len([s for s in signals if s[0] == "VENTA"])
    
    # Generar razón detallada
    all_reasons = [s[1] for s in signals]
    detailed_reason = " | ".join(all_reasons[:3])
    
    if buy_signals > sell_signals:
        return "COMPRA", detailed_reason
    elif sell_signals > buy_signals:
        return "VENTA", detailed_reason
    else:
        return "MANTENER", "Señales mixtas: " + detailed_reason

def calculate_price_targets(hist_4h, current_price, symbol):
    """Calcula niveles de precio objetivo para compra y venta"""
    if symbol not in ['TSLA', 'NVDA']:
        return None, None

    # Calcular niveles técnicos
    atr = calculate_atr(hist_4h)
    sma_20 = hist_4h['Close'].rolling(window=20).mean().iloc[-1]
    recent_high = hist_4h['High'].tail(20).max()
    recent_low = hist_4h['Low'].tail(20).min()
    
    # Calcular niveles de Fibonacci desde el mínimo reciente
    fib_levels = calculate_fibonacci_levels(recent_low, recent_high)
    
    # Niveles de soporte y resistencia
    supports = [recent_low, fib_levels[0.382], sma_20]
    resistances = [recent_high, fib_levels[0.618], fib_levels[0.786]]
    
    # Encontrar el mejor nivel de compra (soporte más cercano por debajo del precio)
    buy_targets = [s for s in supports if s < current_price]
    buy_target = max(buy_targets) if buy_targets else current_price * 0.95
    
    # Encontrar el mejor nivel de venta (resistencia más cercana por encima del precio)
    sell_targets = [r for r in resistances if r > current_price]
    sell_target = min(sell_targets) if sell_targets else current_price * 1.05
    
    # Ajustar por volatilidad (ATR)
    buy_target = round(buy_target - atr * 0.5, 2)
    sell_target = round(sell_target + atr * 0.5, 2)
    
    return buy_target, sell_target

def calculate_atr(data, period=14):
    """Calcula el Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr.iloc[-1]

def calculate_fibonacci_levels(low, high):
    """Calcula niveles de Fibonacci"""
    diff = high - low
    levels = {
        0.236: low + diff * 0.236,
        0.382: low + diff * 0.382,
        0.5: low + diff * 0.5,
        0.618: low + diff * 0.618,
        0.786: low + diff * 0.786
    }
    return levels

def get_stock_data(symbols):
    data = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo")  # Obtener datos del último mes
            
            # Calcular cambios porcentuales
            week_change = ((hist['Close'][-1] - hist['Close'][-5]) / hist['Close'][-5] * 100) if len(hist) >= 5 else 0
            month_change = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100) if len(hist) > 0 else 0
            
            # Obtener datos del pre-mercado
            premarket_data = get_premarket_data(symbol)
            
            # Obtener datos de 7 días con intervalos de 1 hora
            hist_4h = resample_to_4h(hist)
            
            current_price = hist_4h['Close'].iloc[-1]
            previous_close = hist_4h['Close'].iloc[-2]
            
            # Cálculos básicos
            price_change = ((current_price - previous_close) / previous_close) * 100
            rsi = calculate_rsi(hist_4h).iloc[-1]
            volume_change = ((hist_4h['Volume'].iloc[-1] - hist_4h['Volume'].mean()) / hist_4h['Volume'].mean()) * 100
            sma_20 = hist_4h['Close'].rolling(window=20).mean().iloc[-1]
            sma_ratio = current_price / sma_20
            
            # Indicadores avanzados
            macd, macd_signal, macd_line, signal_line = calculate_macd(hist_4h)
            upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(hist_4h)
            k, d = calculate_stochastic(hist_4h)
            
            # Predicción para el próximo día incluyendo datos del pre-mercado
            prediction = predict_next_day_movement(hist_4h, macd_line, signal_line, premarket_data)
            
            # Obtener recomendación actual
            current_rec, current_reason = get_recommendation(
                price_change, rsi, volume_change, sma_ratio,
                current_price, upper_bb, lower_bb, macd, macd_signal, k, d
            )
            
            # Combinar la predicción con la recomendación actual
            if prediction['Tendencia'] == "ALCISTA":
                main_recommendation = "COMPRA"
            elif prediction['Tendencia'] == "BAJISTA":
                main_recommendation = "VENTA"
            else:
                main_recommendation = "MANTENER"
            
            # Incluir datos del pre-mercado en la razón si están disponibles
            premarket_info = f" | Premarket: {prediction.get('Premarket', 'No disponible')}" if 'Premarket' in prediction else ""
            
            # Calcular objetivos de precio para TSLA y NVDA
            buy_target, sell_target = calculate_price_targets(hist_4h, current_price, symbol)
            price_targets = ""
            if buy_target and sell_target:
                price_targets = f" | Objetivo Compra: ${buy_target} | Objetivo Venta: ${sell_target}"

            # Incluir los objetivos de precio en la razón si están disponibles
            recommendation_reason = f"{prediction['Tendencia']} ({prediction['Fuerza']}) | Confianza: {prediction['Confianza']}{premarket_info}{price_targets} | {current_reason}"
            
            # Calcular volumen promedio
            daily_volume = hist_4h['Volume'].tail(6).mean()
            avg_volume = format_volume(int(daily_volume))
            
            data.append({
                'Símbolo': symbol,
                'Empresa': top_companies[symbol],
                'Precio Actual ($)': current_price,
                'Cambio 1 Semana (%)': round(week_change, 2),
                'Cambio 1 Mes (%)': round(month_change, 2),
                'Recomendación': main_recommendation,
                'Razón': recommendation_reason,
                'Cierre Anterior ($)': previous_close,
                'Cambio (%)': f"{price_change:.2f}%",
                'Volumen 24h': avg_volume,
                'RSI': f"{rsi:.1f}",
                'MACD': f"{macd:.2f}",
                'Estocástico %K': f"{k:.1f}"
            })
        except Exception as e:
            st.error(f"Error procesando {symbol}: {str(e)}")
            continue
    
    return pd.DataFrame(data)

# Mostrar DataFrame con nuevo formato
with st.container():
    df = get_stock_data(list(top_companies.keys()))
    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
            height=1000,  # Aumentar altura para mostrar más datos
            column_config={
                'Cambio 1 Semana (%)': st.column_config.NumberColumn(
                    format="%.2f%%",
                    help="Cambio porcentual en la última semana"
                ),
                'Cambio 1 Mes (%)': st.column_config.NumberColumn(
                    format="%.2f%%",
                    help="Cambio porcentual en el último mes"
                )
            }
        )

# Mostrar última actualización
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Agregar botón de actualización
if st.button("🔄 Actualizar Datos"):
    st.rerun()

if __name__ == "__main__":
    main() 