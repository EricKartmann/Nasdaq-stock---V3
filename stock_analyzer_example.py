import pandas as pd
import matplotlib.pyplot as plt
from stock_analyzer import StockAnalyzer

def main():
    """
    Ejemplo de uso del componente StockAnalyzer para analizar acciones
    y generar recomendaciones con diferentes estrategias.
    """
    print("=== Analizador de Acciones con Nadaraya-Watson Envelope ===")
    print("Este programa analiza acciones y genera recomendaciones basadas en indicadores técnicos.")
    
    # Lista de acciones a analizar
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    company_names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc."
    }
    
    # Crear analizadores con diferentes estrategias
    strategies = ["Equilibrada", "Conservadora", "Agresiva", "Tendencia", "Contratendencia"]
    analyzers = {strategy: StockAnalyzer(strategy=strategy) for strategy in strategies}
    
    # Analizar acciones con cada estrategia
    results = {}
    for strategy, analyzer in analyzers.items():
        print(f"\nAnalizando acciones con estrategia {strategy}...")
        results[strategy] = analyzer.analyze_stocks(symbols, company_names)
    
    # Mostrar resultados
    for strategy, df in results.items():
        print(f"\n=== Resultados con estrategia {strategy} ===")
        print(df[['Símbolo', 'Empresa', 'Precio Actual', 'Recomendación', 'Razón']])
    
    # Comparar recomendaciones entre estrategias
    compare_recommendations(results)
    
    # Crear una estrategia personalizada
    print("\n=== Creando estrategia personalizada ===")
    custom_analyzer = StockAnalyzer()
    custom_weights = {
        "rsi": 2.0,  # Mayor peso al RSI
        "macd": 1.2,
        "bollinger": 0.8,
        "stochastic": 0.5,
        "nw_envelope": 1.8,  # Mayor peso al Nadaraya-Watson Envelope
        "price_action": 0.7,
        "volume": 0.5
    }
    custom_analyzer.set_custom_weights(custom_weights)
    
    # Analizar con la estrategia personalizada
    custom_results = custom_analyzer.analyze_stocks(symbols, company_names)
    print("\n=== Resultados con estrategia personalizada ===")
    print(custom_results[['Símbolo', 'Empresa', 'Precio Actual', 'Recomendación', 'Razón']])
    
    # Guardar resultados en Excel
    save_to_excel(results, custom_results)

def compare_recommendations(results):
    """
    Compara las recomendaciones entre diferentes estrategias.
    
    Args:
        results: Diccionario con los resultados de cada estrategia
    """
    # Crear un DataFrame para comparar recomendaciones
    comparison = pd.DataFrame()
    
    for strategy, df in results.items():
        comparison[strategy] = df.set_index('Símbolo')['Recomendación']
    
    print("\n=== Comparación de recomendaciones entre estrategias ===")
    print(comparison)
    
    # Contar recomendaciones por estrategia
    counts = {}
    for strategy, df in results.items():
        counts[strategy] = df['Recomendación'].value_counts()
    
    # Crear gráfico de barras para comparar recomendaciones
    plt.figure(figsize=(12, 6))
    
    for i, strategy in enumerate(results.keys()):
        buy_count = counts[strategy].get('COMPRA', 0)
        hold_count = counts[strategy].get('MANTENER', 0)
        sell_count = counts[strategy].get('VENTA', 0)
        
        x = [i-0.2, i, i+0.2]
        y = [buy_count, hold_count, sell_count]
        
        plt.bar(x, y, width=0.15, label=strategy if i == 0 else "")
    
    plt.xticks(range(len(results)), list(results.keys()))
    plt.ylabel('Número de acciones')
    plt.title('Comparación de recomendaciones por estrategia')
    
    # Añadir leyenda
    plt.legend(['COMPRA', 'MANTENER', 'VENTA'], loc='upper right')
    
    # Guardar gráfico
    plt.savefig('comparacion_estrategias.png')
    print("\nGráfico guardado como 'comparacion_estrategias.png'")

def save_to_excel(strategy_results, custom_results):
    """
    Guarda los resultados en un archivo Excel.
    
    Args:
        strategy_results: Diccionario con los resultados de cada estrategia
        custom_results: DataFrame con los resultados de la estrategia personalizada
    """
    # Crear un escritor de Excel
    writer = pd.ExcelWriter('analisis_acciones.xlsx', engine='xlsxwriter')
    
    # Guardar cada estrategia en una hoja diferente
    for strategy, df in strategy_results.items():
        df.to_excel(writer, sheet_name=strategy, index=False)
    
    # Guardar la estrategia personalizada
    custom_results.to_excel(writer, sheet_name='Personalizada', index=False)
    
    # Guardar el archivo
    writer.close()
    print("\nResultados guardados en 'analisis_acciones.xlsx'")

if __name__ == "__main__":
    main() 