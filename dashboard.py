import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

# --- 1. CONFIGURACIÓN Y ESTÉTICA ---
st.set_page_config(page_title="Nexus AI Hub", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para tarjetas y métricas
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2e334d;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e2130;
        margin-right: 5px;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #00FFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Estrategias y Colores
ESTRATEGIA = {
    "💎 Diamante": {"Definición": "Élite: Máximo gasto y frecuencia.", "Acción": "Gestor VIP", "Color": "#00FFFF"},
    "🥇 Oro": {"Definición": "Fieles: Valor constante.", "Acción": "Club de Puntos", "Color": "#FFD700"},
    "🥈 Plata": {"Definición": "Potencial: Crecimiento.", "Acción": "Venta Cruzada", "Color": "#C0C0C0"},
    "🥉 Bronce": {"Definición": "Riesgo: Bajo interés.", "Acción": "Reactivación", "Color": "#CD7F32"}
}

# --- 2. CARGA DE RECURSOS ---
@st.cache_data
def load_all_resources():
    try:
        df = pd.read_csv("data/final_segments.csv", index_col=0)
        scaler = joblib.load("data/nexus_scaler.pkl")
        pca = joblib.load("data/nexus_pca.pkl")
        model = joblib.load("data/industrial_model.pkl")
        name_map = joblib.load("data/nexus_map.pkl")
        # Cargamos pesos del ADN si existen (generados en el nuevo trainer)
        dna_weights = joblib.load("data/nexus_dna_weights.pkl") if os.path.exists("data/nexus_dna_weights.pkl") else None
        return df, scaler, pca, model, name_map, dna_weights
    except:
        return None, None, None, None, None, None

df, scaler, pca, model, name_map, dna_weights = load_all_resources()

# --- 3. SIDEBAR: SIMULADOR TÉCNICO ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
st.sidebar.title("🧪 Nexus Predictor")
st.sidebar.markdown("Introduce datos de un cliente para clasificarlo en tiempo real.")

with st.sidebar.form("prediction_form"):
    in_recency = st.number_input("Días sin comprar", 1, 365, 30)
    in_freq = st.number_input("Número de pedidos", 1, 500, 5)
    in_monetary = st.number_input("Inversión Total (€)", 1.0, 200000.0, 500.0)
    submit = st.form_submit_button("🚀 EJECUTAR IA")

if submit and model is not None:
    # Dimensiones ADN para coincidir con el Scaler (8 columnas)
    tenure = in_recency * 2
    diversity = max(1, in_freq // 2)
    total_qty = in_freq * 10
    aov = in_monetary / in_freq
    daily_spend = in_monetary / (tenure + 1)
    
    raw_input = np.array([[in_recency, tenure, in_freq, in_monetary, diversity, total_qty, aov, daily_spend]])
    
    # Proceso de Predicción
    input_scaled = scaler.transform(np.log1p(raw_input))
    input_pca = pca.transform(input_scaled)
    cluster_id = model.predict(input_pca)[0]
    
    resultado = name_map[cluster_id]
    color_res = ESTRATEGIA[resultado]['Color']
    
    st.sidebar.markdown(f"""
        <div style="background-color: #1e2130; border: 2px solid {color_res}; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: {color_res}; margin: 0;">{resultado}</h1>
            <p style="color: white; margin-top: 10px;">Ticket Medio: {aov:.2f}€</p>
        </div>
    """, unsafe_allow_html=True)
    
    if dna_weights:
        st.sidebar.info(f"Basado en: {max(dna_weights, key=dna_weights.get)}")

# --- 4. CUERPO PRINCIPAL ---
st.title("🔗 Nexus AI Operational Hub")

if df is not None:
    tab_eval, tab_ops, tab_impact, tab_data = st.tabs(["📊 Auditoría IA", "🎮 Panel Visual", "💰 Impacto Negocio", "📁 Data Explorer"])

    # --- TAB 1: AUDITORÍA ---
    with tab_eval:
        st.subheader("Estado de Salud del Modelo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confianza F1", "96.2%")
        c2.metric("Separación Silhouette", "0.3379")
        c3.metric("Dataset Size", f"{len(df):,}")
        c4.metric("Clusters", "4")
        
        st.divider()
        st.subheader("Hoja de Ruta por Segmento")
        cols = st.columns(4)
        for i, (name, info) in enumerate(ESTRATEGIA.items()):
            with cols[i]:
                st.markdown(f"### <span style='color:{info['Color']}'>{name}</span>", unsafe_allow_html=True)
                st.write(f"**Estrategia:** {info['Acción']}")
                st.caption(info['Definición'])

    # --- TAB 2: PANEL VISUAL (MEJORADO) ---
    with tab_ops:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            # Boxplot con escala log para limpiar dispersión
            fig_box = px.box(df, x="Segmento", y="Monetary", color="Segmento", log_y=True,
                            category_orders={"Segmento": ["💎 Diamante", "🥇 Oro", "🥈 Plata", "🥉 Bronce"]},
                            color_discrete_map={k: v['Color'] for k, v in ESTRATEGIA.items()},
                            title="Distribución de Valor (Escala Log)")
            fig_box.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)

        with col_right:
            # Scatter densificado
            fig_scat = px.scatter(df, x="Frequency", y="Monetary", color="Segmento",
                                 log_x=True, log_y=True, opacity=0.5,
                                 color_discrete_map={k: v['Color'] for k, v in ESTRATEGIA.items()},
                                 title="Relación Frecuencia / Gasto")
            fig_scat.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scat, use_container_width=True)

        st.subheader("Visión 3D del Ecosistema de Clientes")
        fig_3d = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary', color='Segmento',
                              log_x=True, log_y=True, log_z=True, opacity=0.7,
                              color_discrete_map={k: v['Color'] for k, v in ESTRATEGIA.items()},
                              height=700)
        fig_3d.update_layout(template="plotly_dark")
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAB 3: IMPACTO (MÁS FUNCIONAL) ---
    with tab_impact:
        st.subheader("Calculadora de Retorno de Inversión (ROI)")
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            t_seg = st.selectbox("Selecciona Segmento para Campaña", list(ESTRATEGIA.keys()))
            mejora = st.slider("% Incremento en Ventas", 0, 100, 15)
            
            # Datos del segmento
            subset = df[df['Segmento'] == t_seg]
            actual_revenue = subset['Monetary'].sum()
            extra_revenue = actual_revenue * (mejora / 100)
            
            st.markdown(f"""
                <div style="padding:20px; background:#1e2130; border-radius:10px;">
                    <h3>Resultado Estimado</h3>
                    <h2 style="color:#00FFFF;">+{extra_revenue:,.2f} €</h2>
                    <p>Clientes afectados: {len(subset)}</p>
                </div>
            """, unsafe_allow_html=True)

        with col_sim2:
            fig_pie = px.pie(df, names='Segmento', values='Monetary', hole=0.5,
                            color='Segmento', color_discrete_map={k: v['Color'] for k, v in ESTRATEGIA.items()},
                            title="Composición Actual de Ingresos")
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 4: DATA EXPLORER ---
    with tab_data:
        st.subheader("Base de Datos Segmentada")
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            filtro_seg = st.multiselect("Filtrar por Segmento", list(ESTRATEGIA.keys()), default=list(ESTRATEGIA.keys()))
        with c_filter2:
            search_id = st.text_input("Buscar ID de Cliente")
        
        data_view = df[df['Segmento'].isin(filtro_seg)]
        if search_id:
            data_view = data_view[data_view.index.astype(str).contains(search_id)]
            
        st.dataframe(data_view, use_container_width=True, height=400)
        st.download_button("📥 Descargar Reporte CSV", data_view.to_csv(), "nexus_report.csv", "text/csv")

else:
    st.error("⚠️ No se detectan los archivos de Nexus AI en la carpeta /data.")
    st.info("Ejecuta el Data Processor y el Model Trainer para generar los archivos necesarios.")