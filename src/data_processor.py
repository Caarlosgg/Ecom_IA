import pandas as pd
import numpy as np
from scipy import stats
import os
from sklearn.preprocessing import PowerTransformer
import joblib

def run_pro_processor():
    # Configuración de rutas
    RAW_PATH = "data/raw_data.parquet"
    PROCESSED_PATH = "data/rfm_processed.csv"
    ORIGINAL_PATH = "data/rfm_original_metrics.csv"
    SCALER_PATH = "data/nexus_scaler.pkl"
    
    print("🚀 INICIANDO PROTOCOLO DE PROCESAMIENTO ECOM-IA (ULTRA PRO)...")

    # --- 1. CARGA INTELIGENTE Y LIMPIEZA INICIAL ---
    if not os.path.exists('data'): 
        os.makedirs('data')

    if os.path.exists(RAW_PATH):
        print("⚡ Cargando desde cache local (Ultrarrápido)...")
        df = pd.read_parquet(RAW_PATH)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print("🌐 Descargando dataset maestro desde UCI Repository...")
        try:
            # CORRECCIÓN AQUÍ: Forzamos que StockCode y otros sean texto (str) desde el inicio
            # Esto evita el error "Expected bytes, got int"
            df = pd.read_excel(
                url, 
                dtype={'StockCode': str, 'CustomerID': str, 'InvoiceNo': str}
            )
            
            # --- LIMPIEZA PROFUNDA ---
            # 1. Eliminar filas sin Cliente
            df = df.dropna(subset=['CustomerID'])
            
            # 2. Limpieza de IDs (quitar decimales .0 si vinieran como string "12345.0")
            df['CustomerID'] = df['CustomerID'].str.split('.').str[0]
            
            # 3. Eliminar devoluciones (Facturas que empiezan por 'C')
            df = df[~df['InvoiceNo'].str.startswith('C')]
            
            # Guardar cache para la próxima
            df.to_parquet(RAW_PATH, index=False)
        except Exception as e:
            print(f"❌ Error crítico en descarga: {e}")
            print("💡 Consejo: Si el error persiste, descarga el archivo 'Online Retail.xlsx' manualmente y ponlo en la carpeta.")
            return

    # --- 2. INGENIERÍA DE VARIABLES (CREACIÓN DEL ADN) ---
    print("⚙️ Generando métricas avanzadas (RFM + Profundidad)...")
    
    # Conversión segura a números
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    
    # Filtro de Negocio: Solo ventas válidas (>0)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    
    # Fecha de corte
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # AGREGACIÓN POR CLIENTE
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': [lambda x: (snapshot_date - x.max()).days,  # Recency
                        lambda x: (x.max() - x.min()).days],       # Tenure
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalSum': 'sum',                                         # Monetary
        'StockCode': 'nunique',                                    # Diversity
        'Quantity': 'sum'                                          # TotalQuantity
    })
    
    # Aplanar columnas
    rfm.columns = ['Recency', 'Tenure', 'Frequency', 'Monetary', 'Diversity', 'TotalQuantity']
    
    # Variables Derivadas
    rfm['AOV'] = (rfm['Monetary'] / rfm['Frequency']).fillna(0)
    rfm['DailySpend'] = (rfm['Monetary'] / (rfm['Tenure'] + 1)).fillna(0)

    # --- 3. LIMPIEZA ESTADÍSTICA (OUTLIERS) ---
    print("🧹 Eliminando anomalías estadísticas (Z-Score Filter)...")
    rfm_log_temp = np.log1p(rfm)
    
    # Mantenemos solo clientes "normales"
    mask = (np.abs(stats.zscore(rfm_log_temp)) < 3.0).all(axis=1)
    rfm_clean = rfm[mask].copy()
    
    deleted = len(rfm) - len(rfm_clean)
    print(f"   -> Eliminados {deleted} clientes anómalos.")

    # --- 4. SERIALIZACIÓN VISUAL ---
    print("✨ Re-indexando clientes para Dashboard Visual...")
    
    rfm_clean = rfm_clean.reset_index()
    rfm_clean = rfm_clean.rename(columns={'CustomerID': 'Real_ID'})
    
    # Índice serial limpio (1 a N)
    rfm_clean.index = rfm_clean.index + 1
    rfm_clean.index.name = 'Cliente_ID'

    # --- 5. NORMALIZACIÓN MATEMÁTICA "PRO" (Yeo-Johnson) ---
    print("🧪 Aplicando transformación Yeo-Johnson...")
    
    cols_to_scale = ['Recency', 'Tenure', 'Frequency', 'Monetary', 'Diversity', 'TotalQuantity', 'AOV', 'DailySpend']
    
    pt = PowerTransformer(method='yeo-johnson')
    rfm_transformed = pt.fit_transform(rfm_clean[cols_to_scale])
    
    rfm_scaled = pd.DataFrame(rfm_transformed, 
                              columns=cols_to_scale, 
                              index=rfm_clean.index)

    # --- 6. GUARDADO DE ARTEFACTOS ---
    print("💾 Guardando sistema de archivos sincronizado...")
    
    rfm_scaled.to_csv(PROCESSED_PATH)
    rfm_clean.to_csv(ORIGINAL_PATH)
    joblib.dump(pt, SCALER_PATH)
    
    print("-" * 60)
    print(f"✅ PROCESO COMPLETADO EXITOSAMENTE")
    print(f"📊 Clientes listos: {len(rfm_scaled)}")
    print(f"🆔 Formato Visual: IDs del 1 al {len(rfm_scaled)}")
    print("-" * 60)

if __name__ == "__main__":
    run_pro_processor()