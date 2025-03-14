import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class StockAnalyzer:
    """
    Clase para analizar acciones y generar recomendaciones basadas en indicadores técnicos.
    Esta clase puede ser reutilizada en diferentes aplicaciones.
    """
    
    def __init__(self, strategy="Equilibrada"):
        """
        Inicializa el analizador de acciones con una estrategia predefinida.
        
        Args:
            strategy: Estrategia de trading a utilizar (Equilibrada, Conservadora, Agresiva, etc.)
        """
        self.strategy = strategy
        self.weights = self._get_strategy_weights(strategy)
    
    def _get_strategy_weights(self, strategy):
        """
        Obtiene los pesos para cada indicador según la estrategia seleccionada.
        
        Args:
            strategy: Nombre de la estrategia
            
        Returns:
            Diccionario con los pesos para cada indicador
        """
        # Pesos predefinidos para cada estrategia
        strategies = {
            "Equilibrada": {
                "rsi": 1.0,
                "macd": 1.0,
                "bollinger": 1.0,
                "stochastic": 1.0,
                "nw_envelope": 1.0,
                "price_action": 1.0,
                "volume": 1.0
            },
            "Conservadora": {
                "rsi": 1.5,
                "macd": 0.8,
                "bollinger": 1.5,
                "stochastic": 1.2,
                "nw_envelope": 1.5,
                "price_action": 0.7,
                "volume": 0.5
            },
            "Agresiva": {
                "rsi": 0.7,
                "macd": 1.5,
                "bollinger": 0.8,
                "stochastic": 0.8,
                "nw_envelope": 0.8,
                "price_action": 1.8,
                "volume": 1.5
            },
            "Tendencia": {
                "rsi": 0.8,
                "macd": 1.8,
                "bollinger": 0.6,
                "stochastic": 0.7,
                "nw_envelope": 1.2,
                "price_action": 1.5,
                "volume": 1.3
            },
            "Contratendencia": {
                "rsi": 1.8,
                "macd": 0.7,
                "bollinger": 1.8,
                "stochastic": 1.5,
                "nw_envelope": 1.8,
                "price_action": 0.6,
                "volume": 0.8
            }
        }
        
        # Si la estrategia no existe, usar la equilibrada
        return strategies.get(strategy, strategies["Equilibrada"])
    
    def set_custom_weights(self, weights):
        """
        Establece pesos personalizados para los indicadores.
        
        Args:
            weights: Diccionario con los pesos para cada indicador
        """
        self.weights = weights
        self.strategy = "Personalizada"
    
    def resample_to_4h(self, data):
        """
        Convierte los datos a intervalos de 4 horas.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con datos en intervalos de 4 horas
        """
        data_4h = data.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return data_4h
    
    def calculate_rsi(self, data, periods=14):
        """
        Calcula el RSI (Relative Strength Index).
        
        Args:
            data: DataFrame con datos OHLCV
            periods: Número de períodos para el cálculo
            
        Returns:
            Serie con los valores del RSI
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, short_window=12, long_window=26, signal_window=9):
        """
        Calcula el MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame con datos OHLCV
            short_window: Ventana corta para EMA
            long_window: Ventana larga para EMA
            signal_window: Ventana para la línea de señal
            
        Returns:
            Valor actual del MACD, valor actual de la señal, línea MACD completa, línea de señal completa
        """
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        
        # Calcular el histograma del MACD
        histogram = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], macd, signal, histogram
    
    def calculate_bollinger_bands(self, data, window=20):
        """
        Calcula las Bandas de Bollinger.
        
        Args:
            data: DataFrame con datos OHLCV
            window: Ventana para el cálculo
            
        Returns:
            Banda superior, media móvil, banda inferior
        """
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]
    
    def calculate_stochastic(self, data, k_window=14, d_window=3):
        """
        Calcula el Oscilador Estocástico.
        
        Args:
            data: DataFrame con datos OHLCV
            k_window: Ventana para %K
            d_window: Ventana para %D
            
        Returns:
            Valor actual de %K, valor actual de %D
        """
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_window).mean()
        
        return k.iloc[-1], d.iloc[-1]
    
    def calculate_nadaraya_watson_envelope(self, data, bandwidth=8.0, multiplier=3.0, source='Close'):
        """
        Calcula el indicador Nadaraya-Watson Envelope.
        
        Args:
            data: DataFrame con datos OHLCV
            bandwidth: Ancho de banda para el kernel gaussiano
            multiplier: Multiplicador para el error medio absoluto
            source: Columna de datos a utilizar
            
        Returns:
            Banda superior, banda inferior
        """
        # Función gaussiana para ponderación
        def gaussian(x, h):
            return np.exp(-(x**2)/(h*h*2))
        
        # Obtener los datos de origen
        src = data[source].values
        n = len(src)
        
        # Limitar a 500 barras como máximo
        max_bars = min(500, n)
        
        # Calcular el estimador Nadaraya-Watson
        y_nw = np.zeros(max_bars)
        
        for i in range(max_bars):
            sum_weight = 0
            weighted_sum = 0
            
            for j in range(max_bars):
                # Calcular el peso gaussiano
                weight = gaussian(i - j, bandwidth)
                weighted_sum += src[j] * weight
                sum_weight += weight
            
            # Estimación en el punto i
            y_nw[i] = weighted_sum / sum_weight
        
        # Calcular el error medio absoluto
        mae = np.mean(np.abs(src[:max_bars] - y_nw)) * multiplier
        
        # Calcular las bandas superior e inferior
        upper_band = y_nw[-1] + mae
        lower_band = y_nw[-1] - mae
        
        return upper_band, lower_band
    
    def get_advanced_signals(self, price, upper_bb, lower_bb, macd, macd_signal, k, d, nw_upper=None, nw_lower=None, macd_hist=None, prev_macd_hist=None, rsi=None):
        """
        Genera señales avanzadas basadas en indicadores técnicos.
        
        Args:
            price: Precio actual
            upper_bb: Banda superior de Bollinger
            lower_bb: Banda inferior de Bollinger
            macd: Valor actual del MACD
            macd_signal: Valor actual de la línea de señal
            k: Valor actual de %K
            d: Valor actual de %D
            nw_upper: Banda superior de Nadaraya-Watson
            nw_lower: Banda inferior de Nadaraya-Watson
            macd_hist: Valor actual del histograma del MACD
            prev_macd_hist: Valor anterior del histograma del MACD
            rsi: Valor actual del RSI
            
        Returns:
            Lista de tuplas (señal, razón, peso)
        """
        signals = []
        
        # Señales de Bandas de Bollinger
        if price > upper_bb:
            signals.append(("VENTA", "Precio sobre banda superior de Bollinger", self.weights["bollinger"]))
        elif price < lower_bb:
            signals.append(("COMPRA", "Precio bajo banda inferior de Bollinger", self.weights["bollinger"]))
        
        # Señales de MACD
        if macd > macd_signal and macd > 0:
            signals.append(("COMPRA", "Cruce alcista del MACD", self.weights["macd"]))
        elif macd < macd_signal and macd < 0:
            signals.append(("VENTA", "Cruce bajista del MACD", self.weights["macd"]))
        
        # Señales del Estocástico
        if k > 80 and d > 80:
            signals.append(("VENTA", "Estocástico en zona de sobrecompra", self.weights["stochastic"]))
        elif k < 20 and d < 20:
            signals.append(("COMPRA", "Estocástico en zona de sobreventa", self.weights["stochastic"]))
        
        # Señales de Nadaraya-Watson Envelope
        if nw_upper is not None and nw_lower is not None:
            # Señal de venta: Precio sobre banda superior + inversión tendencia MACD
            if price > nw_upper:
                signals.append(("VENTA", "Precio sobre banda superior de NW Envelope", self.weights["nw_envelope"] * 0.7))
                
                # Señal combinada: Precio sobre NW + inversión tendencia MACD
                if macd_hist is not None and prev_macd_hist is not None:
                    if macd_hist < prev_macd_hist and prev_macd_hist > 0:
                        signals.append(("VENTA", "Precio sobre NW Envelope con inversión de tendencia en MACD", self.weights["nw_envelope"] * 1.8))
            
            # Señal de compra: Precio bajo banda inferior + cambio tendencia MACD + RSI < 30 + estocástico favorable
            elif price < nw_lower:
                signals.append(("COMPRA", "Precio bajo banda inferior de NW Envelope", self.weights["nw_envelope"] * 0.5))
                
                # Señal combinada completa
                if macd_hist is not None and prev_macd_hist is not None and rsi is not None:
                    # Cambio de tendencia en MACD
                    macd_trend_change = macd_hist > prev_macd_hist and prev_macd_hist < 0
                    
                    # RSI en sobreventa
                    rsi_oversold = rsi < 30
                    
                    # Estocástico favorable (en zona de sobreventa)
                    stoch_favorable = k < 20 and d < 20
                    
                    # Señal combinada parcial: NW + MACD
                    if macd_trend_change:
                        signals.append(("COMPRA", "Precio bajo NW Envelope con cambio de tendencia en MACD", self.weights["nw_envelope"] * 1.2))
                    
                    # Señal combinada completa: NW + MACD + RSI + Estocástico
                    if macd_trend_change and rsi_oversold and stoch_favorable:
                        signals.append(("COMPRA", "Señal óptima: NW + MACD + RSI + Estocástico", self.weights["nw_envelope"] * 2.5))
                    # Señal combinada: NW + MACD + RSI
                    elif macd_trend_change and rsi_oversold:
                        signals.append(("COMPRA", "Señal fuerte: NW + MACD + RSI", self.weights["nw_envelope"] * 2.0))
        
        return signals
    
    def get_recommendation(self, price_change, rsi, volume_change, sma_ratio, price, upper_bb, lower_bb, macd, macd_signal, k, d, nw_upper=None, nw_lower=None):
        """
        Genera una recomendación basada en todos los indicadores técnicos.
        
        Args:
            price_change: Cambio porcentual en el precio
            rsi: Valor actual del RSI
            volume_change: Cambio porcentual en el volumen
            sma_ratio: Ratio entre precio actual y SMA
            price: Precio actual
            upper_bb: Banda superior de Bollinger
            lower_bb: Banda inferior de Bollinger
            macd: Valor actual del MACD
            macd_signal: Valor actual de la línea de señal
            k: Valor actual de %K
            d: Valor actual de %D
            nw_upper: Banda superior de Nadaraya-Watson
            nw_lower: Banda inferior de Nadaraya-Watson
            
        Returns:
            Tupla (recomendación, razón)
        """
        signals = []
        
        # Señales básicas
        if price_change > 2:
            signals.append(("VENTA", "Subida fuerte de precio", self.weights["price_action"]))
        elif price_change < -2:
            signals.append(("COMPRA", "Caída significativa de precio", self.weights["price_action"]))
            
        if rsi > 70:
            signals.append(("VENTA", "RSI en sobrecompra", self.weights["rsi"]))
        elif rsi < 30:
            signals.append(("COMPRA", "RSI en sobreventa", self.weights["rsi"]))
            
        if volume_change > 50:
            signals.append(("COMPRA" if price_change > 0 else "VENTA", "Alto volumen de operaciones", self.weights["volume"]))
            
        if sma_ratio > 1.05:
            signals.append(("COMPRA", "Tendencia alcista fuerte", self.weights["price_action"]))
        elif sma_ratio < 0.95:
            signals.append(("VENTA", "Tendencia bajista fuerte", self.weights["price_action"]))
        
        # Añadir señales avanzadas
        signals.extend(self.get_advanced_signals(price, upper_bb, lower_bb, macd, macd_signal, k, d, nw_upper, nw_lower))
        
        if not signals:
            return "MANTENER", "Sin señales claras de trading"
        
        # Calcular puntuación ponderada para señales de compra y venta
        buy_score = sum([s[2] for s in signals if s[0] == "COMPRA"])
        sell_score = sum([s[2] for s in signals if s[0] == "VENTA"])
        
        # Generar razón detallada
        all_reasons = [s[1] for s in signals]
        detailed_reason = " | ".join(all_reasons[:3])
        
        if buy_score > sell_score:
            return "COMPRA", detailed_reason
        elif sell_score > buy_score:
            return "VENTA", detailed_reason
        else:
            return "MANTENER", "Señales mixtas: " + detailed_reason
    
    def analyze_stock(self, symbol, period="1mo"):
        """
        Analiza una acción y genera una recomendación.
        
        Args:
            symbol: Símbolo de la acción
            period: Período de tiempo para los datos históricos
            
        Returns:
            Diccionario con los resultados del análisis
        """
        try:
            # Obtener datos históricos
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {
                    'Símbolo': symbol,
                    'Error': 'No se encontraron datos históricos'
                }
            
            # Calcular cambios porcentuales
            week_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) >= 5 else 0
            month_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100) if len(hist) > 0 else 0
            
            # Obtener datos con intervalos de 4 horas
            hist_4h = self.resample_to_4h(hist)
            
            current_price = hist_4h['Close'].iloc[-1]
            previous_close = hist_4h['Close'].iloc[-2]
            
            # Cálculos básicos
            price_change = ((current_price - previous_close) / previous_close) * 100
            rsi = self.calculate_rsi(hist_4h).iloc[-1]
            volume_change = ((hist_4h['Volume'].iloc[-1] - hist_4h['Volume'].mean()) / hist_4h['Volume'].mean()) * 100
            sma_20 = hist_4h['Close'].rolling(window=20).mean().iloc[-1]
            sma_ratio = current_price / sma_20
            
            # Indicadores avanzados
            macd, macd_signal, macd_line, signal_line, histogram = self.calculate_macd(hist_4h)
            upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(hist_4h)
            k, d = self.calculate_stochastic(hist_4h)
            
            # Calcular Nadaraya-Watson Envelope
            nw_upper, nw_lower = self.calculate_nadaraya_watson_envelope(hist_4h)
            
            # Obtener valores actuales y anteriores del histograma del MACD
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2] if len(histogram) > 1 else None
            
            # Obtener recomendación basada en indicadores técnicos
            nw_signals = []  # Señales específicas de Nadaraya-Watson + MACD
            other_signals = []  # Otras señales
            
            # Señales de Nadaraya-Watson Envelope (prioridad alta)
            if nw_upper is not None and nw_lower is not None:
                # Señal de venta: Precio sobre banda superior + inversión tendencia MACD
                if current_price > nw_upper:
                    # Señal básica
                    nw_signals.append(("VENTA", "NW ENVELOPE: Precio sobre banda superior", self.weights["nw_envelope"] * 0.7))
                    
                    # Señal combinada: Precio sobre NW + inversión tendencia MACD
                    if current_hist is not None and prev_hist is not None:
                        if current_hist < prev_hist and prev_hist > 0:
                            nw_signals.append(("VENTA", "NW ENVELOPE FUERTE: Zona de venta con inversión de tendencia en MACD", self.weights["nw_envelope"] * 1.8))
                
                # Señal de compra: Precio bajo banda inferior + cambio tendencia MACD + RSI < 30 + estocástico favorable
                elif current_price < nw_lower:
                    # Señal básica
                    nw_signals.append(("COMPRA", "NW ENVELOPE: Precio bajo banda inferior", self.weights["nw_envelope"] * 0.5))
                    
                    # Señal combinada completa
                    if current_hist is not None and prev_hist is not None:
                        # Cambio de tendencia en MACD
                        macd_trend_change = current_hist > prev_hist and prev_hist < 0
                        
                        # RSI en sobreventa
                        rsi_oversold = rsi < 30
                        
                        # Estocástico favorable (en zona de sobreventa)
                        stoch_favorable = k < 20 and d < 20
                        
                        # Señal combinada parcial: NW + MACD
                        if macd_trend_change:
                            nw_signals.append(("COMPRA", "NW ENVELOPE: Zona de compra con cambio de tendencia en MACD", self.weights["nw_envelope"] * 1.2))
                        
                        # Señal combinada completa: NW + MACD + RSI + Estocástico
                        if macd_trend_change and rsi_oversold and stoch_favorable:
                            nw_signals.append(("COMPRA", "NW ENVELOPE ÓPTIMO: Zona de compra + MACD + RSI + Estocástico", self.weights["nw_envelope"] * 2.5))
                        # Señal combinada: NW + MACD + RSI
                        elif macd_trend_change and rsi_oversold:
                            nw_signals.append(("COMPRA", "NW ENVELOPE FUERTE: Zona de compra + MACD + RSI", self.weights["nw_envelope"] * 2.0))
            
            # Señales básicas (prioridad media)
            if price_change > 2:
                other_signals.append(("VENTA", "Subida fuerte de precio", self.weights["price_action"]))
            elif price_change < -2:
                other_signals.append(("COMPRA", "Caída significativa de precio", self.weights["price_action"]))
                
            if rsi > 70:
                other_signals.append(("VENTA", "RSI en sobrecompra", self.weights["rsi"]))
            elif rsi < 30:
                other_signals.append(("COMPRA", "RSI en sobreventa", self.weights["rsi"]))
                
            if volume_change > 50:
                other_signals.append(("COMPRA" if price_change > 0 else "VENTA", "Alto volumen de operaciones", self.weights["volume"]))
                
            if sma_ratio > 1.05:
                other_signals.append(("COMPRA", "Tendencia alcista fuerte", self.weights["price_action"]))
            elif sma_ratio < 0.95:
                other_signals.append(("VENTA", "Tendencia bajista fuerte", self.weights["price_action"]))
            
            # Señales de Bandas de Bollinger
            if current_price > upper_bb:
                other_signals.append(("VENTA", "Precio sobre banda superior de Bollinger", self.weights["bollinger"]))
            elif current_price < lower_bb:
                other_signals.append(("COMPRA", "Precio bajo banda inferior de Bollinger", self.weights["bollinger"]))
            
            # Señales de MACD
            if macd > macd_signal and macd > 0:
                other_signals.append(("COMPRA", "Cruce alcista del MACD", self.weights["macd"]))
            elif macd < macd_signal and macd < 0:
                other_signals.append(("VENTA", "Cruce bajista del MACD", self.weights["macd"]))
            
            # Señales del Estocástico
            if k > 80 and d > 80:
                other_signals.append(("VENTA", "Estocástico en zona de sobrecompra", self.weights["stochastic"]))
            elif k < 20 and d < 20:
                other_signals.append(("COMPRA", "Estocástico en zona de sobreventa", self.weights["stochastic"]))
            
            # Combinar todas las señales, priorizando las de NW Envelope
            signals = nw_signals + other_signals
            
            # Calcular puntuación ponderada para señales de compra y venta
            buy_score = sum([s[2] for s in signals if s[0] == "COMPRA"])
            sell_score = sum([s[2] for s in signals if s[0] == "VENTA"])
            
            # Generar razón detallada, priorizando las señales de NW Envelope
            nw_reasons = [s[1] for s in nw_signals]
            other_reasons = [s[1] for s in other_signals]
            
            # Si hay señales de NW, usarlas como principales y asegurarse de que se muestren primero
            if nw_reasons:
                # Tomar hasta 2 razones de NW Envelope
                detailed_reason = " | ".join(nw_reasons[:2])
                
                # Añadir una señal adicional de otras si hay espacio y si hay otras razones
                if other_reasons and len(nw_reasons) < 2:
                    detailed_reason += " | " + other_reasons[0]
            else:
                # Si no hay señales de NW, usar las otras (hasta 3)
                detailed_reason = " | ".join(other_reasons[:3])
            
            # Determinar la fuerza de la señal
            signal_strength = ""
            if buy_score > sell_score:
                recommendation = "COMPRA"
                signal_strength = self._get_signal_strength(buy_score)
            elif sell_score > buy_score:
                recommendation = "VENTA"
                signal_strength = self._get_signal_strength(sell_score)
            else:
                recommendation = "MANTENER"
                detailed_reason = "Señales mixtas: " + detailed_reason
            
            # Añadir fuerza de la señal a la recomendación si existe
            if signal_strength:
                recommendation = f"{recommendation} ({signal_strength})"
            
            # Calcular volumen promedio
            daily_volume = hist_4h['Volume'].tail(6).mean()
            
            # Formatear volumen
            if daily_volume >= 1_000_000_000:
                avg_volume = f"{daily_volume/1_000_000_000:.2f}B"
            elif daily_volume >= 1_000_000:
                avg_volume = f"{daily_volume/1_000_000:.2f}M"
            elif daily_volume >= 1_000:
                avg_volume = f"{daily_volume/1_000:.2f}K"
            else:
                avg_volume = str(int(daily_volume))
            
            # Devolver resultados
            return {
                'Símbolo': symbol,
                'Precio Actual': current_price,
                'Cambio 1 Semana (%)': round(week_change, 2),
                'Cambio 1 Mes (%)': round(month_change, 2),
                'Recomendación': recommendation,
                'Razón': detailed_reason,
                'Cierre Anterior': previous_close,
                'Cambio (%)': round(price_change, 2),
                'Volumen 24h': avg_volume,
                'RSI': round(rsi, 1),
                'MACD': round(macd, 2),
                'Histograma MACD': round(current_hist, 2),
                'Estocástico %K': round(k, 1),
                'NW Envelope': f"{round(nw_upper, 2)}/{round(nw_lower, 2)}"
            }
            
        except Exception as e:
            return {
                'Símbolo': symbol,
                'Error': str(e)
            }
    
    def _get_signal_strength(self, score):
        """
        Determina la fuerza de una señal basada en su puntuación.
        
        Args:
            score: Puntuación de la señal
            
        Returns:
            Cadena que indica la fuerza de la señal
        """
        if score > 5.0:
            return "Muy Fuerte"
        elif score > 3.5:
            return "Fuerte"
        elif score > 2.0:
            return "Moderada"
        else:
            return "Débil"
    
    def analyze_stocks(self, symbols, company_names=None):
        """
        Analiza múltiples acciones y genera recomendaciones.
        
        Args:
            symbols: Lista de símbolos de acciones
            company_names: Diccionario con nombres de empresas (opcional)
            
        Returns:
            DataFrame con los resultados del análisis
        """
        results = []
        
        for symbol in symbols:
            result = self.analyze_stock(symbol)
            
            # Añadir nombre de la empresa si está disponible
            if company_names and symbol in company_names:
                result['Empresa'] = company_names[symbol]
            
            results.append(result)
        
        return pd.DataFrame(results)

# Ejemplo de uso:
if __name__ == "__main__":
    # Crear un analizador con estrategia equilibrada
    analyzer = StockAnalyzer(strategy="Equilibrada")
    
    # Analizar una acción
    result = analyzer.analyze_stock("AAPL")
    print(f"Recomendación para AAPL: {result['Recomendación']}")
    print(f"Razón: {result['Razón']}")
    
    # Cambiar a estrategia agresiva
    analyzer.strategy = "Agresiva"
    analyzer.weights = analyzer._get_strategy_weights("Agresiva")
    
    # Analizar múltiples acciones
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = analyzer.analyze_stocks(symbols)
    print("\nResultados con estrategia agresiva:")
    print(results[['Símbolo', 'Precio Actual', 'Recomendación', 'Razón']]) 