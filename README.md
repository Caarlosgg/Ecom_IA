# 💎 EcomIA | Enterprise Customer Intelligence System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/AI-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)

**EcomIA** es una solución integral de Inteligencia Artificial para la segmentación avanzada de clientes en E-commerce. Diseñada con una arquitectura de microservicios, combina un pipeline de entrenamiento científico robusto, un dashboard estratégico para la toma de decisiones y una API REST para integración en tiempo real.

---

## 🚀 Características Principales

* **🧠 Motor de IA Científico:** Implementación de **K-Means Clustering** optimizado, apoyado por **PCA** (Reducción de Dimensionalidad) y **Isolation Forest** para la limpieza automática de anomalías y ruido en los datos.
* **🛡️ Lógica de Negocio Blindada:** Algoritmo de ranking jerárquico ponderado (80% Importancia Monetaria) que garantiza una segmentación comercialmente coherente y libre de alucinaciones (Diamante > Oro > Plata > Bronce).
* **📊 Dashboard "Glassmorphism":** Interfaz visual interactiva desarrollada en **Streamlit**, con métricas en tiempo real, gráficos 3D y simulador de escenarios ROI.
* **🔌 API de Alto Rendimiento:** Microservicio **FastAPI** desplegable que permite consultar el segmento y la estrategia de un cliente en milisegundos desde cualquier plataforma externa.
* **⚙️ MLOps Pipeline:** Script de orquestación (`run_pipeline.py`) para automatizar el ciclo de vida del dato: ETL, Re-entrenamiento y Despliegue.

---

## 📂 Arquitectura del Proyecto

El proyecto sigue una estructura modular escalable de grado industrial:

```text
ECOM_IA/
├── data/                    # Data Lake y Model Registry
│   ├── plots/               # Evidencia visual generada por la IA (PNG)
│   ├── final_segments.csv   # Dataset maestro segmentado
│   ├── nexus_*.pkl          # Artefactos serializados del modelo (Scalers, PCA, KMeans)
│   ├── raw_data.parquet     # Cache de datos crudos optimizada
│   └── rfm_processed.csv    # Datos pre-procesados (Yeo-Johnson)
├── src/                     # Núcleo del Sistema
│   ├── api.py               # Servidor FastAPI (Endpoint de Inferencia)
│   ├── data_processor.py    # ETL: Limpieza, RFM Engineering y Z-Score
│   └── model_trainer.py     # IA: Isolation Forest + PCA + K-Means
├── dashboard.py             # Interfaz de Usuario (Frontend Streamlit)
├── run_pipeline.py          # Orquestador de Automatización
├── requirements.txt         # Dependencias del entorno
└── README.md                # Documentación técnica
```

---

## 🛠️ Instalación y Despliegue

### Prerrequisitos
* Python 3.9 o superior
* Git

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/EcomIA.git](https://github.com/TU_USUARIO/EcomIA.git)
cd ECOM_IA
```

### 2. Configurar el entorno virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🚦 Guía de Uso

El sistema opera en tres modos distintos según la necesidad:

### A. Modo Automatización (Pipeline)
Para procesar nuevos datos crudos y re-entrenar el cerebro de la IA de cero:
```bash
python run_pipeline.py
```
*Este comando ejecuta secuencialmente la limpieza, ingeniería de variables y entrenamiento, actualizando los archivos `.pkl` en la carpeta `data/`.*

### B. Modo Visualización (Dashboard)
Para abrir la herramienta de análisis estratégico y simulación:
```bash
streamlit run dashboard.py
```

### C. Modo Producción (API)
Para levantar el servidor de inferencia (para conectar con Shopify, WooCommerce, etc.):
```bash
uvicorn src.api:app --reload
```
*El servidor escuchará peticiones en `http://127.0.0.1:8000`.*

---

## 🔌 Documentación de la API

La API expone un endpoint POST para predicciones en tiempo real. Documentación interactiva disponible en `/docs` una vez iniciado el servidor.

**Endpoint:** `POST /predict_segment`

**Ejemplo de Request (JSON):**
```json
{
  "recency": 12,
  "tenure": 450,
  "frequency": 15,
  "monetary": 2500.00,
  "diversity": 8,
  "quantity": 120
}
```

**Ejemplo de Response:**
```json
{
  "segment": "💎 Diamante",
  "strategy": "Trato VIP y Acceso Anticipado",
  "metrics": {
    "aov": 166.67,
    "daily_spend": 5.54
  }
}
```

---

## 🧬 Metodología Científica

El modelo clasifica a los clientes en 4 clusters estrictos basados en su comportamiento transaccional:

| Segmento | Perfil del Cliente | Acción Recomendada |
| :--- | :--- | :--- |
| **💎 Diamante** | **Élite:** Gasto muy alto, compra reciente y frecuente. | Atención VIP, Eventos Exclusivos. |
| **🥇 Oro** | **Leal:** Gasto alto y recurrente. Pilar del negocio. | Programas de Fidelización, Cross-Sell. |
| **🥈 Plata** | **Promedio:** Gasto medio, frecuencia ocasional. | Nurturing, Cupones por volumen. |
| **🥉 Bronce** | **Riesgo:** Gasto bajo o inactividad prolongada. | Campañas agresivas de reactivación. |

**Pipeline Técnico Detallado:**
1.  **Ingeniería de Variables:** Creación de métricas RFM (Recency, Frequency, Monetary) + Profundidad (Diversity, Quantity, AOV).
2.  **Limpieza Avanzada:** Filtro estadístico Z-Score + **Isolation Forest** (Contaminación 2%) para eliminar ruido y outliers antes del entrenamiento.
3.  **Pre-procesamiento:** Transformación **Yeo-Johnson** para normalizar distribuciones asimétricas y Escalado MinMax.
4.  **Reducción:** **PCA** (Principal Component Analysis) proyectando 8 dimensiones a 3 componentes latentes (reteniendo >90% varianza explicada).
5.  **Clustering:** **K-Means** con inicialización robusta (`n_init=50`, `max_iter=500`) y scoring de confianza basado en distancias euclidianas.


Desarrollado como parte de un sistema experto de Inteligencia de Negocio.

**Stack:** Python | Streamlit | FastAPI | Docker Ready
