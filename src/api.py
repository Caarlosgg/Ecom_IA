from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# 1. Inicializar la App
app = FastAPI(title="EcomIA Nexus API", version="3.0", description="API de Segmentación de Clientes en Tiempo Real")

# 2. Cargar el Cerebro (Solo una vez al inicio)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

try:
    scaler = joblib.load(os.path.join(DATA_DIR, "nexus_scaler.pkl"))
    pca = joblib.load(os.path.join(DATA_DIR, "nexus_pca.pkl"))
    model = joblib.load(os.path.join(DATA_DIR, "nexus_model.pkl"))
    logic_map = joblib.load(os.path.join(DATA_DIR, "nexus_logic.pkl"))
    name_map = joblib.load(os.path.join(DATA_DIR, "nexus_map.pkl"))
    print("✅ Sistema EcomIA cargado en memoria.")
except Exception as e:
    print(f"❌ Error crítico cargando modelos: {e}")

# 3. Definir el formato de los datos de entrada (Validación estricta)
class ClientData(BaseModel):
    recency: int
    tenure: int
    frequency: int
    monetary: float
    diversity: int
    quantity: int

# 4. El Endpoint (La puerta de entrada)
@app.post("/predict_segment")
def predict(client: ClientData):
    """
    Recibe datos crudos de un cliente y devuelve su Segmento y Estrategia.
    """
    try:
        # Ingeniería de variables (Igual que en el dashboard)
        # Protección contra división por cero
        aov = client.monetary / client.frequency if client.frequency > 0 else 0
        daily_spend = client.monetary / (client.tenure + 1) if client.tenure > 0 else 0

        # Crear Vector (Orden EXACTO del entrenamiento)
        # ['Recency', 'Tenure', 'Frequency', 'Monetary', 'Diversity', 'TotalQuantity', 'AOV', 'DailySpend']
        features = pd.DataFrame([[
            client.recency, client.tenure, client.frequency, client.monetary,
            client.diversity, client.quantity, aov, daily_spend
        ]], columns=['Recency', 'Tenure', 'Frequency', 'Monetary', 'Diversity', 'TotalQuantity', 'AOV', 'DailySpend'])

        # Inferencia
        scaled_features = scaler.transform(features)
        pca_features = pca.transform(scaled_features)
        cluster_id = model.predict(pca_features)[0]
        
        # Traducción
        final_id = logic_map.get(cluster_id, cluster_id)
        segment_name = name_map.get(final_id, "Desconocido")

        # Estrategia (Pequeña base de datos de acciones)
        strategies = {
            "💎 Diamante": "Trato VIP y Acceso Anticipado",
            "🥇 Oro": "Programa de Puntos y Cross-Sell",
            "🥈 Plata": "Cupones por volumen",
            "🥉 Bronce": "Descuentos agresivos de recuperación"
        }

        return {
            "segment": segment_name,
            "strategy": strategies.get(segment_name, "Analizar"),
            "metrics": {
                "aov": round(aov, 2),
                "daily_spend": round(daily_spend, 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Para correrlo: uvicorn src.api:app --reload