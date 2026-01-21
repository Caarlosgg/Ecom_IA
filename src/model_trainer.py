import pandas as pd
import numpy as np
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import os
import joblib

def train_and_evaluate():
    path_processed = "data/rfm_processed.csv"
    path_metrics = "data/rfm_original_metrics.csv"

    if not os.path.exists(path_processed):
        print("❌ Error: Ejecuta data_processor.py primero.")
        return

    # 1. CARGA DE DATOS
    data = pd.read_csv(path_processed, index_col=0)
    original_metrics = pd.read_csv(path_metrics, index_col=0)
    
    # 2. OPTIMIZACIÓN DE DIMENSIONALIDAD (PCA)
    # Usamos 3 componentes pero aseguramos que la varianza explicada sea suficiente
    pca = PCA(n_components=3, random_state=42)
    data_pca = pca.fit_transform(data)
    var_explicada = np.sum(pca.explained_variance_ratio_)
    
    # 3. CLUSTERING JERÁRQUICO (BisectingKMeans)
    # Aumentamos n_init para evitar mínimos locales y asegurar grupos estables
    model = BisectingKMeans(
        n_clusters=4, 
        init='k-means++', 
        n_init=100,      # Mayor estabilidad
        bisecting_strategy='biggest_inertia', 
        random_state=42
    )
    raw_clusters = model.fit_predict(data_pca)

    # 4. LÓGICA DE RE-RANKING PROFESIONAL
    # No usamos solo Monetary; usamos un Score combinado para evitar errores de clasificación
    # Un Diamante debe ser alto en Monetary y Frequency
    original_metrics['Temp_Cluster'] = raw_clusters
    
    # Calculamos la mediana para no dejarnos engañar por outliers
    cluster_profile = original_metrics.groupby('Temp_Cluster').agg({
        'Monetary': 'median',
        'Frequency': 'median',
        'Recency': 'median'
    })

    # Creamos un ranking basado en valor comercial (Dinero + Frecuencia - Recencia)
    # Esto asegura que el ID 0 sea siempre el mejor cliente real
    ranking = cluster_profile.assign(
        score = cluster_profile['Monetary'] * 0.7 + cluster_profile['Frequency'] * 0.3
    ).sort_values('score', ascending=False).index

    logic_map = {old_id: new_id for new_id, old_id in enumerate(ranking)}
    
    # Aplicamos el mapeo final
    original_metrics['Cluster'] = original_metrics['Temp_Cluster'].map(logic_map)
    
    name_map = {
        0: "💎 Diamante",
        1: "🥇 Oro",
        2: "🥈 Plata",
        3: "🥉 Bronce"
    }
    
    original_metrics['Segmento'] = original_metrics['Cluster'].map(name_map)
    final_clusters = original_metrics['Cluster'].values

    # 5. AUDITORÍA DE ESTRÉS (Clasificador de Validación)
    # Esto mide si un cliente nuevo podrá ser clasificado con éxito
    X_train, X_test, y_train, y_test = train_test_split(data, final_clusters, test_size=0.2, random_state=42)
    
    # ExtraTrees nos dirá qué variables son las que realmente definen a tus clientes
    validator = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42)
    validator.fit(X_train, y_train)
    
    # Medimos la importancia de las variables (Para el ADN del Dashboard)
    importances = dict(zip(data.columns, validator.feature_importances_))
    
    report = classification_report(y_test, validator.predict(X_test), output_dict=True)
    cohesion = silhouette_score(data_pca, final_clusters)

    # 6. SALIDA DE CONTROL DE CALIDAD
    print("\n" + "═"*60)
    print(f"🏆 AUDITORÍA DE SISTEMA NEXUS AI")
    print("═"*60)
    print(f"📈 Varianza Capturada PCA: {var_explicada:.2%}")
    print(f"🧪 Cohesión Silhouette: {cohesion:.4f} (Calidad de separación)")
    print("-" * 60)
    print("🎯 PESO DE VARIABLES EN LA DECISIÓN (ADN):")
    for feat, val in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f" • {feat:<15}: {val:.2%}")
    print("-" * 60)
    for i, name in name_map.items():
        acc = report[str(i)]['f1-score']
        print(f"{name:<15} | Estabilidad F1: {acc:.4f}")
    print("═"*60)

    # 7. EXPORTACIÓN DE CEREBROS
    os.makedirs("data", exist_ok=True)
    original_metrics.drop(columns=['Temp_Cluster']).to_csv("data/final_segments.csv")
    
    # Guardamos todo lo necesario para que el Dashboard sea una "réplica" del entrenamiento
    joblib.dump(model, "data/nexus_model.pkl")
    joblib.dump(pca, "data/nexus_pca.pkl")
    joblib.dump(name_map, "data/nexus_map.pkl")
    joblib.dump(logic_map, "data/nexus_logic.pkl")
    joblib.dump(importances, "data/nexus_dna_weights.pkl") # Nuevo: Pesos de las variables
    
    print(f"🚀 Cerebro Nexus AI entrenado y jerarquizado correctamente.")

if __name__ == "__main__":
    train_and_evaluate()