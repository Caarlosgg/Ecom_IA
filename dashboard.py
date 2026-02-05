import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL "GOLD MASTER" (CSS PROFESIONAL)
# ==============================================================================
st.set_page_config(
    page_title="EcomIA | Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💎"
)

# Estilos CSS Avanzados para dar aspecto de App Profesional
st.markdown("""
    <style>
    /* Fondo Principal */
    .main { background-color: #0E1117; }
    h1, h2, h3, h4 { color: #FAFAFA; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Métricas (Tarjetas superiores) */
    div[data-testid="metric-container"] {
        background-color: #1A1C24;
        border: 1px solid #30333F;
        padding: 15px; border-radius: 8px; color: #E0E0E0;
        transition: transform 0.2s, border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover { 
        border-color: #00FFFF; 
        transform: translateY(-2px); 
    }
    div[data-testid="metric-container"] label { color: #AAA; font-size: 0.85rem; }
    
    /* Tarjetas de Definición de Grupos */
    .group-card {
        background-color: #15171E;
        border-radius: 10px;
        padding: 20px;
        border-top: 5px solid;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Sidebar (Barra lateral) */
    [data-testid="stSidebar"] { background-color: #15171E; border-right: 1px solid #30333F; }
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab"] { 
        background-color: #1A1C24; 
        color: #AAA; 
        border-radius: 5px 5px 0 0; 
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #262730 !important; 
        color: #FFFFFF !important; 
        border-top: 3px solid #00FFFF; 
        border-bottom: none;
    }
    
    /* Tablas */
    [data-testid="stDataFrame"] { border: 1px solid #30333F; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. DEFINICIONES DE NEGOCIO (K=4)
# ==============================================================================
SEGMENT_DEFINITIONS = {
    "💎 Diamante": {
        "Color": "#00FFFF", "Icon": "💎", 
        "Title": "Top Performers", 
        "Desc": "Clientes de altísimo valor. Alta frecuencia y gasto.",
        "Action": "Atención VIP & Upselling"
    },
    "🥇 Oro": {
        "Color": "#FFD700", "Icon": "🥇", 
        "Title": "Clientes Leales", 
        "Desc": "Compradores recurrentes y sólidos.",
        "Action": "Fidelización (Puntos)"
    },
    "🥈 Plata": {
        "Color": "#C0C0C0", "Icon": "🥈", 
        "Title": "Potencial", 
        "Desc": "Compras ocasionales. Margen de crecimiento.",
        "Action": "Nurturing & Cross-Sell"
    },
    "🥉 Bronce": {
        "Color": "#CD7F32", "Icon": "🥉", 
        "Title": "Bajo Valor / Riesgo", 
        "Desc": "Inactivos recientes o compras pequeñas.",
        "Action": "Reactivación Agresiva"
    }
}

# Función auxiliar para colores transparentes en gráficos
def hex_to_rgba(hex_code, opacity):
    hex_code = hex_code.lstrip('#')
    return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {opacity})"

# ==============================================================================
# 3. CARGA DE SISTEMA ROBUSTA (ANTI-ERRORES)
# ==============================================================================
@st.cache_data
def load_system():
    # Lista de archivos requeridos
    files = ["data/final_segments.csv", "data/nexus_scaler.pkl", "data/nexus_pca.pkl", 
             "data/nexus_model.pkl", "data/nexus_map.pkl", "data/nexus_logic.pkl", "data/nexus_dna_weights.pkl"]
    
    # Verificación de existencia
    if any(not os.path.exists(f) for f in files):
        return None, "❌ Faltan archivos del sistema. Por favor, ejecuta 'model_trainer.py' primero."
    
    try:
        # Carga del CSV
        df = pd.read_csv("data/final_segments.csv", index_col=0)
        
        # --- CORRECCIÓN CRÍTICA DE ID ---
        # Si el CSV tiene índice, lo reseteamos para que el ID sea una columna normal
        df = df.reset_index()
        
        # Buscamos si existe 'Real_ID' (creado por data_processor) o 'CustomerID'
        # Y aseguramos que sea Texto (String) para que no falle el buscador
        if 'Real_ID' in df.columns:
            df['Real_ID'] = df['Real_ID'].astype(str)
        elif 'CustomerID' in df.columns:
            df = df.rename(columns={'CustomerID': 'Real_ID'})
            df['Real_ID'] = df['Real_ID'].astype(str)
        else:
            # Si no hay ID, creamos uno ficticio para que no falle el gráfico
            df['Real_ID'] = df.index.astype(str)

        # Carga de Cerebros IA (Modelos PKL)
        scaler = joblib.load("data/nexus_scaler.pkl")
        pca = joblib.load("data/nexus_pca.pkl")
        model = joblib.load("data/nexus_model.pkl")
        name_map = joblib.load("data/nexus_map.pkl")
        logic_map = joblib.load("data/nexus_logic.pkl")
        dna_weights = joblib.load("data/nexus_dna_weights.pkl")
        
        return (df, scaler, pca, model, name_map, logic_map, dna_weights), None
        
    except Exception as e:
        return None, f"Error interno leyendo archivos: {str(e)}"

# Ejecutamos carga
assets, error_msg = load_system()

# Si hay error, detenemos todo y mostramos mensaje claro
if error_msg:
    st.error(error_msg)
    st.stop()

# Desempaquetamos los activos
df, scaler, pca, model, name_map, logic_map, dna_weights = assets

# Detectamos la variable más importante para el Sidebar
top_feature = max(dna_weights, key=dna_weights.get) if dna_weights else "General"


# ==============================================================================
# 4. SIDEBAR: PREDICTOR INTELIGENTE (SIN ERRORES DE MÍNIMOS)
# ==============================================================================
st.sidebar.title("🔮 EcomIA Predictor")
st.sidebar.caption(f"🧠 Modelo Activo: **4 Clusters** | 🔑 Clave: **{top_feature}**")

with st.sidebar.form("prediction_form"):
    st.markdown("### 1. Variables Temporales")
    # Recency: Mínimo 1
    in_recency = st.number_input("Días sin comprar (Recency)", min_value=1, max_value=2000, value=30, help="Días desde la última compra.")
    
    # Tenure: Debe ser mayor que Recency.
    # LOGICA ANTI-CRASH: El valor por defecto no puede ser menor que el min_value
    min_tenure_val = max(1, in_recency)
    default_tenure = max(min_tenure_val, 365)
    in_tenure = st.number_input("Antigüedad (Días)", min_value=min_tenure_val, max_value=5000, value=default_tenure)

    st.markdown("### 2. Variables Monetarias")
    in_freq = st.number_input("Pedidos Totales", min_value=1, max_value=20000, value=5)
    in_monetary = st.number_input("Gasto Total (€)", min_value=1.0, max_value=5000000.0, value=500.0)

    st.markdown("### 3. Profundidad")
    
    # --- CORRECCIÓN ERROR "StreamlitValueBelowMinError" ---
    # Calculamos valores por defecto, pero aseguramos con max(1, ...) que NUNCA sean 0.
    calc_diversity = int(in_freq * 1.5)
    safe_diversity = max(1, calc_diversity) # <--- BLINDAJE
    in_diversity = st.number_input("Productos Únicos", min_value=1, max_value=10000, value=safe_diversity)
    
    calc_qty = int(in_monetary / 20)
    safe_qty = max(1, calc_qty) # <--- BLINDAJE
    in_qty = st.number_input("Unidades Totales", min_value=1, max_value=500000, value=safe_qty)

    st.markdown("---")
    # Botón de Enviar Formulario
    btn_predict = st.form_submit_button("🔎 CLASIFICAR CLIENTE", use_container_width=True)

# Lógica de Predicción (Se ejecuta al pulsar el botón)
if btn_predict:
    # 1. Variables Derivadas (Matemáticas)
    calc_aov = in_monetary / in_freq
    calc_daily = in_monetary / (in_tenure + 1)
    
    # 2. DataFrame Exacto (8 variables en el orden del entrenamiento)
    raw_input = pd.DataFrame([[in_recency, in_tenure, in_freq, in_monetary, in_diversity, in_qty, calc_aov, calc_daily]],
                             columns=['Recency', 'Tenure', 'Frequency', 'Monetary', 'Diversity', 'TotalQuantity', 'AOV', 'DailySpend'])
    
    # 3. Inferencia (Yeo-Johnson -> PCA -> K-Means)
    # Scaler maneja la transformación, no hace falta logaritmo manual
    vector_scaled = scaler.transform(raw_input)
    vector_pca = pca.transform(vector_scaled)
    
    # Predicción del Cluster ID
    raw_cluster_id = model.predict(vector_pca)[0]
    
    # Mapeo a Nombre de Negocio
    final_cluster_id = logic_map.get(raw_cluster_id, raw_cluster_id)
    segment_name = name_map.get(final_cluster_id, "Desconocido")
    
    # 4. Resultado Visual
    info = SEGMENT_DEFINITIONS.get(segment_name, {"Color": "#888", "Icon": "?", "Action": "N/A"})
    
    st.sidebar.markdown(f"""
    <div style="margin-top:15px; padding:15px; border:2px solid {info['Color']}; border-radius:10px; background:linear-gradient(135deg, #15171E, {info['Color']}11);">
        <h5 style="color:#AAA; margin:0; font-size:10px; text-transform:uppercase;">CLASIFICACIÓN IA</h5>
        <h2 style="color:{info['Color']}; margin:5px 0; text-shadow:0 0 15px {info['Color']}44;">
            {info['Icon']} {segment_name}
        </h2>
        <div style="margin-top:10px; font-size:12px; color:#DDD;">
            <strong>Estrategia:</strong><br>{info.get('Action', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feedback de métricas
    col_sb1, col_sb2 = st.sidebar.columns(2)
    col_sb1.metric("Ticket Medio", f"{calc_aov:.0f}€")
    col_sb2.metric("Gasto Diario", f"{calc_daily:.2f}€")

    # Radar Chart
    fig_radar = go.Figure(go.Scatterpolar(
        r=[1-(min(in_recency,365)/365), min(in_freq/50,1), min(in_monetary/5000,1)],
        theta=['Recency', 'Frequency', 'Monetary'],
        fill='toself', line_color=info['Color'], fillcolor=hex_to_rgba(info['Color'], 0.2)
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), 
                           paper_bgcolor='rgba(0,0,0,0)', height=150, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    st.sidebar.plotly_chart(fig_radar, use_container_width=True)


# ==============================================================================
# 5. CUERPO PRINCIPAL DEL DASHBOARD
# ==============================================================================
st.title("🚀 EcomIA | Dashboard Operativo")
st.markdown(f"**Dataset:** {len(df):,} Clientes | **Estado:** En línea 🟢")

# Pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🌎 Visión Global", 
    "📈 Análisis Profundo", 
    "💰 Calculadora ROI", 
    "💾 Explorador de Datos"
])

# --- TAB 1: VISIÓN GLOBAL ---
with tab1:
    # -----------------------------------------------------------
    # RADIOGRAFÍA DE CLUSTERS (TABLA DE VERDAD)
    # -----------------------------------------------------------
    st.subheader("📊 Radiografía de Clusters (Medianas Reales)")
    st.info("💡 Usa estos valores como referencia. Muestran al 'Cliente Típico' de cada grupo.")
    
    # Calculamos MEDIANAS reales de los datos (Más estable que Mín/Máx)
    stats_df = df.groupby('Segmento').agg({
        'Monetary': lambda x: f"{x.median():,.0f}€",
        'Frequency': lambda x: f"{int(x.median())} pedidos",
        'Recency': lambda x: f"{int(x.median())} días"
    })
    # Reordenamos para que salga Diamante primero
    ordered_index = ["💎 Diamante", "🥇 Oro", "🥈 Plata", "🥉 Bronce"]
    stats_df = stats_df.reindex(ordered_index)
    
    st.table(stats_df) # Usamos st.table para máxima claridad
    st.divider()
    # -----------------------------------------------------------

    # Tarjetas de Definición
    cols = st.columns(4)
    for i, key in enumerate(ordered_index):
        data = SEGMENT_DEFINITIONS.get(key)
        if data:
            with cols[i]:
                st.markdown(f"""
                <div class="group-card" style="border-color: {data['Color']};">
                    <h3 style="color: {data['Color']}; margin:0;">{data['Icon']} {key.split(' ')[1]}</h3>
                    <p style="color: #FFF; font-weight:bold; font-size:0.9rem; margin-bottom:5px;">{data['Title']}</p>
                    <p style="color: #AAA; font-size:0.8rem; height: 45px;">{data['Desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()

    # KPIs Globales
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ingresos Totales", f"{df['Monetary'].sum()/1000:.1f}k €")
    k2.metric("Ticket Medio", f"{df['Monetary'].mean():.2f} €")
    k3.metric("Frecuencia Media", f"{df['Frequency'].mean():.1f}")
    k4.metric("Tasa Churn", f"{(len(df[df['Recency']>90])/len(df))*100:.1f}%", delta_color="inverse")

    # Gráficos
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Distribución")
        seg_counts = df['Segmento'].value_counts()
        fig_pie = px.pie(values=seg_counts, names=seg_counts.index, hole=0.6, 
                         color=seg_counts.index, 
                         color_discrete_map={k: v['Color'] for k, v in SEGMENT_DEFINITIONS.items()})
        fig_pie.update_layout(template="plotly_dark", showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Mapa 3D de Clientes")
        # --- CORRECCIÓN ERROR HOVER_DATA ---
        # Usamos 'Real_ID' explícitamente
        fig_3d = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary', color='Segmento',
                              log_x=True, log_y=True, log_z=True, opacity=0.6,
                              color_discrete_map={k: v['Color'] for k, v in SEGMENT_DEFINITIONS.items()},
                              hover_data=['Real_ID', 'Segmento'])
        fig_3d.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 2: ANÁLISIS PROFUNDO ---
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Antigüedad vs Gasto")
        fig = px.scatter(df, x="Tenure", y="Monetary", color="Segmento", log_x=True, log_y=True,
                        color_discrete_map={k: v['Color'] for k, v in SEGMENT_DEFINITIONS.items()}, 
                        hover_data=['Real_ID'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Dispersión de Valor")
        fig = px.box(df, x="Segmento", y="Monetary", color="Segmento", log_y=True,
                    category_orders={"Segmento": ordered_index},
                    color_discrete_map={k: v['Color'] for k, v in SEGMENT_DEFINITIONS.items()})
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: CALCULADORA ROI ---
with tab3:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 💰 Simulador de Campaña")
        target_seg = st.selectbox("Segmento Objetivo", ordered_index)
        lift_pct = st.slider("% Mejora esperada en Ventas", 1, 50, 15)
        
        # Cálculos
        base_rev = df[df['Segmento']==target_seg]['Monetary'].sum()
        uplift_rev = base_rev * (lift_pct/100)
        col_hex = SEGMENT_DEFINITIONS[target_seg]['Color']
        
        st.markdown(f"""
        <div style="padding:20px; border:1px solid {col_hex}; border-radius:10px; background:#1A1C24; margin-top:20px;">
            <h4 style="color:#AAA; margin:0;">Beneficio Extra</h4>
            <h2 style="color:{col_hex}; margin:5px 0; font-size: 2.5rem;">+{uplift_rev:,.0f} €</h2>
            <div style="font-size: 0.9rem; color: #DDD;">
                Base Actual: {base_rev:,.0f} €
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.subheader("Proyección Visual")
        dat = pd.DataFrame({'Escenario': ['Actual', 'Proyectado'], 'Ventas': [base_rev, base_rev+uplift_rev]})
        fig_bar = px.bar(dat, x='Escenario', y='Ventas', color='Escenario', 
                         color_discrete_sequence=['#444', col_hex])
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 4: EXPLORADOR DE DATOS ---
with tab4:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: f_segs = st.multiselect("Filtrar Segmentos", ordered_index, default=ordered_index)
    with c2: min_euro = st.number_input("Gasto Mínimo (€)", 0, 100000, 0)
    with c3: search_text = st.text_input("Buscar por ID")
    
    # Filtrado Seguro
    mask = (df['Segmento'].isin(f_segs)) & (df['Monetary'] >= min_euro)
    if search_text:
        mask &= df['Real_ID'].str.contains(search_text)
        
    df_filtered = df[mask]
    
    st.markdown(f"**Resultados:** {len(df_filtered)} clientes.")
    st.dataframe(df_filtered.style.format({'Monetary': '{:.2f}€', 'AOV': '{:.2f}€'}), use_container_width=True, height=500)
    
    st.download_button("Descargar CSV", df_filtered.to_csv(index=False), "ecomia_clientes.csv")