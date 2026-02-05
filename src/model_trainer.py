import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías Científicas Avanzadas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances

# Configuración visual para reportes
plt.style.use('dark_background')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def train_advanced_model():
    print("\n" + "█"*80)
    print("🧠  NEXUS AI TRAINER v3.0 | MODO: CIENTÍFICO AVANZADO")
    print("█"*80 + "\n")
    
    # -------------------------------------------------------------------------
    # 1. CARGA Y VALIDACIÓN
    # -------------------------------------------------------------------------
    PATH_PROCESSED = "data/rfm_processed.csv"
    PATH_ORIGINAL = "data/rfm_original_metrics.csv"
    
    if not os.path.exists(PATH_PROCESSED):
        print("❌ Error: Faltan datos procesados.")
        return

    print("📂 Ingestando datos...")
    data_ai = pd.read_csv(PATH_PROCESSED, index_col=0)
    data_human = pd.read_csv(PATH_ORIGINAL, index_col=0)
    print(f"   -> Dataset Inicial: {len(data_ai)} clientes")

    # -------------------------------------------------------------------------
    # 2. DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST) - NUEVO!
    # -------------------------------------------------------------------------
    # Antes de segmentar, eliminamos el ruido. Clientes con comportamientos matemáticamente
    # absurdos o extremos que distorsionan los promedios.
    print("\n🛡️  Ejecutando Protocolo de Limpieza (Isolation Forest)...")
    
    iso = IsolationForest(contamination=0.02, random_state=42) # Eliminamos el 2% más raro
    outliers = iso.fit_predict(data_ai)
    
    # Filtramos los datos (1 = Normal, -1 = Outlier)
    clean_mask = outliers != -1
    
    data_ai_clean = data_ai[clean_mask]
    data_human_clean = data_human[clean_mask]
    
    removed = len(data_ai) - len(data_ai_clean)
    print(f"   -> Outliers detectados y eliminados: {removed}")
    print(f"   -> Dataset Limpio para Entrenamiento: {len(data_ai_clean)} clientes")

    # -------------------------------------------------------------------------
    # 3. PCA OPTIMIZADO
    # -------------------------------------------------------------------------
    print("\n🔭 Proyectando Espacio Vectorial (PCA)...")
    pca = PCA(n_components=3, random_state=42)
    data_pca = pca.fit_transform(data_ai_clean)
    
    var_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"   -> Retención de Información: {var_ratio:.2%}")

    # -------------------------------------------------------------------------
    # 4. K-MEANS DE ALTA PRECISIÓN
    # -------------------------------------------------------------------------
    print("\n🤖 Entrenando Núcleo de Segmentación (K=4)...")
    
    # max_iter=500 y n_init=50 para asegurar convergencia absoluta
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=50, max_iter=500, random_state=42)
    raw_clusters = kmeans.fit_predict(data_pca)
    
    # Métricas de Calidad
    sil = silhouette_score(data_pca, raw_clusters)
    print(f"   -> Índice Silhouette (Cohesión): {sil:.4f} (Excelente > 0.35)")

    # -------------------------------------------------------------------------
    # 5. CÁLCULO DE PROBABILIDAD DE PERTENENCIA - NUEVO!
    # -------------------------------------------------------------------------
    print("\n📐 Calculando Distancias y Scores de Confianza...")
    
    # Obtenemos los centroides en el espacio PCA
    centers = kmeans.cluster_centers_
    
    # Calculamos la distancia de cada punto a su centroide asignado
    # Esto nos dice "cuán representativo" es el cliente de su grupo
    distances = euclidean_distances(data_pca, centers)
    
    # Seleccionamos la distancia al cluster asignado
    min_distances = [distances[i, c] for i, c in enumerate(raw_clusters)]
    
    # Normalizamos (Score 0-100): Más cerca del centro = Más puntuación
    # Usamos una transformación exponencial para suavizar
    confidence_scores = 100 * (1 - (min_distances / np.max(min_distances)))
    
    # Añadimos métricas temporales al dataframe humano
    data_human_clean = data_human_clean.copy()
    data_human_clean['Temp_Cluster'] = raw_clusters
    data_human_clean['Confidence_Score'] = confidence_scores

    # -------------------------------------------------------------------------
    # 6. RANKING JERÁRQUICO ESTRICTO (DINERO = REY)
    # -------------------------------------------------------------------------
    print("⚖️  Aplicando Lógica de Negocio (Money-First)...")
    
    profile = data_human_clean.groupby('Temp_Cluster').agg({
        'Monetary': 'median',
        'Frequency': 'median',
        'Recency': 'median'
    })
    
    scaler_rank = MinMaxScaler()
    profile['Recency_Inv'] = -profile['Recency']
    
    rank_mat = scaler_rank.fit_transform(profile[['Monetary', 'Frequency', 'Recency_Inv']])
    
    # PESOS: 80% Dinero, 10% Frecuencia, 10% Recencia
    profile['Score'] = np.dot(rank_mat, [0.8, 0.1, 0.1])
    
    ranking_ids = profile.sort_values('Score', ascending=False).index
    logic_map = {old: new for new, old in enumerate(ranking_ids)}
    
    name_map = {0: "💎 Diamante", 1: "🥇 Oro", 2: "🥈 Plata", 3: "🥉 Bronce"}

    # -------------------------------------------------------------------------
    # 7. FEATURE IMPORTANCE (RANDOM FOREST ESPÍA)
    # -------------------------------------------------------------------------
    print("🧬 Secuenciando ADN del modelo...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(data_ai_clean, raw_clusters)
    feats = dict(zip(data_ai_clean.columns, clf.feature_importances_))
    feats = dict(sorted(feats.items(), key=lambda x: x[1], reverse=True))

    # -------------------------------------------------------------------------
    # 8. GENERACIÓN DE EVIDENCIA VISUAL - NUEVO!
    # -------------------------------------------------------------------------
    print("🎨 Generando mapa visual de clusters...")
    
    # Mapeamos nombres para el gráfico
    plot_df = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2', 'PCA3'])
    plot_df['Cluster'] = pd.Series(raw_clusters).map(logic_map).map(name_map)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=plot_df, palette='viridis', alpha=0.7)
    plt.title('Distribución Espacial de Clientes (Vista PCA)')
    plt.xlabel('Componente Principal 1 (Valor)')
    plt.ylabel('Componente Principal 2 (Comportamiento)')
    
    if not os.path.exists('data/plots'): os.makedirs('data/plots')
    plt.savefig('data/plots/clusters_visual.png')
    print("   -> Gráfico guardado en: data/plots/clusters_visual.png")

    # -------------------------------------------------------------------------
    # 9. GUARDADO FINAL
    # -------------------------------------------------------------------------
    print("\n💾 Persistiendo Cerebro Digital...")
    
    data_human_clean['Cluster'] = data_human_clean['Temp_Cluster'].map(logic_map)
    data_human_clean['Segmento'] = data_human_clean['Cluster'].map(name_map)
    data_human_clean.drop(columns=['Temp_Cluster'], inplace=True)
    
    data_human_clean.to_csv("data/final_segments.csv")
    joblib.dump(kmeans, "data/nexus_model.pkl")
    joblib.dump(pca, "data/nexus_pca.pkl")
    joblib.dump(name_map, "data/nexus_map.pkl")
    joblib.dump(logic_map, "data/nexus_logic.pkl")
    joblib.dump(feats, "data/nexus_dna_weights.pkl")
    
    print("\n" + "="*80)
    print(f"✅ ENTRENAMIENTO FINALIZADO | {len(data_human_clean)} Clientes Indexados")
    print("="*80)
    
    # REPORTE DE VERDAD
    summary = data_human_clean.groupby('Segmento')[['Monetary', 'Frequency', 'Recency', 'Confidence_Score']].median()
    order = ["💎 Diamante", "🥇 Oro", "🥈 Plata", "🥉 Bronce"]
    
    print(f"{'SEGMENTO':<12} | {'GASTO (€)':<12} | {'FREQ':<6} | {'RECENCY':<10} | {'CONFIDENCIA':<10}")
    print("-" * 65)
    for seg in order:
        if seg in summary.index:
            row = summary.loc[seg]
            print(f"{seg:<12} | {row['Monetary']:>9,.0f} € | {row['Frequency']:>6.0f} | {row['Recency']:>7.0f} d | {row['Confidence_Score']:>9.1f}%")
    print("-" * 65 + "\n")

if __name__ == "__main__":
    train_advanced_model()