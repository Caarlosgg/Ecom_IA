import pandas as pd
import numpy as np
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
import joblib
import winsound

def run_advanced_processor():
    local_path = "data/raw_data.parquet"
    
    # 1. CARGA DE DATOS (Optimizado para evitar corrupción de archivos)
    if os.path.exists(local_path):
        print("⚡ Cargando desde cache local...")
        df = pd.read_parquet(local_path)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print("🌐 Descargando dataset masivo...")
        if not os.path.exists('data'): os.makedirs('data')
        try:
            df = pd.read_excel(url)
            df = df.dropna(subset=['CustomerID'])
            cols_to_fix = ['InvoiceNo', 'StockCode', 'Description', 'CustomerID']
            for col in cols_to_fix:
                df[col] = df[col].astype(str).fillna('')
            df.to_parquet(local_path, index=False)
        except Exception as e:
            print(f"❌ Error en descarga: {e}")
            return

    # 2. FEATURE ENGINEERING
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Agregación precisa
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': [lambda x: (snapshot_date - x.max()).days, 
                        lambda x: (x.max() - x.min()).days],
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum',
        'StockCode': 'nunique',
        'Quantity': 'sum'
    })
    
    rfm.columns = ['Recency', 'Tenure', 'Frequency', 'Monetary', 'Diversity', 'TotalQuantity']
    
    # Evitamos división por cero en AOV y DailySpend
    rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']
    rfm['DailySpend'] = rfm['Monetary'] / (rfm['Tenure'] + 1)
    
    # 3. NORMALIZACIÓN Y FILTRO (Mejora: Manejo de Outliers antes de la serialización)
    # Aplicamos logaritmo para reducir el skewness
    rfm_log = np.log1p(rfm.clip(lower=0))
    
    # Filtro Z-Score 3.0: Mantenemos solo datos que no sean ruido estadístico
    # Importante: Esto reduce el número de filas, por eso la re-serialización va DESPUÉS.
    mask = (np.abs(stats.zscore(rfm_log)) < 3.0).all(axis=1)
    rfm_log_filtered = rfm_log[mask]
    
    # 4. RE-SERIALIZACIÓN (Tu lógica original intacta)
    # Aquí es donde creamos los nuevos IDs de 1 a N
    rfm_log_filtered = rfm_log_filtered.reset_index(drop=True)
    rfm_log_filtered.index = rfm_log_filtered.index + 1
    rfm_log_filtered.index.name = 'CustomerID'

    # 5. ESCALADO ESTÁNDAR
    # Limpiamos posibles NaNs o Infinitos antes de escalar (Seguridad extra)
    rfm_log_filtered = rfm_log_filtered.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    rfm_scaled = pd.DataFrame(scaler.fit_transform(rfm_log_filtered), 
                              columns=rfm_log_filtered.columns, 
                              index=rfm_log_filtered.index)

    # 6. GUARDADO DE MÉTRICAS ORIGINALES
    # Mejora: Sincronización robusta. Filtramos el RFM original con la misma máscara
    # y reseteamos el índice para que coincida 1:1 con rfm_scaled
    rfm_original = rfm[mask].copy()
    rfm_original = rfm_original.reset_index(drop=True)
    rfm_original.index = rfm_log_filtered.index
    
    # 7. EXPORTACIÓN FINAL
    rfm_scaled.to_csv("data/rfm_processed.csv")
    rfm_original.to_csv("data/rfm_original_metrics.csv")
    joblib.dump(scaler, "data/nexus_scaler.pkl")
    
    print(f"✅ ADN Preparado y Serializado.")
    print(f"📊 Clientes totales: {len(rfm_scaled)}")
    print(f"💾 Archivos guardados en /data")

if __name__ == "__main__":
    run_advanced_processor()
    winsound.MessageBeep()