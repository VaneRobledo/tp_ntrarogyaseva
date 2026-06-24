import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import base64


# 1. CONFIGURACIÓN E INTERFAZ
st.set_page_config(page_title="Análisis Programa NTR Vaidya Seva", layout="centered")

css_path = Path(__file__).resolve().parent / 'styles.css'
if css_path.exists():
    with open(css_path, 'r', encoding='utf-8') as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Conviertes tus imágenes locales
img_fase1_base64 = get_image_base64("fase_1.png")
img_fase2_base64 = get_image_base64("fase_2.png")

# Reemplaza con el ID real de tu archivo de Google Drive
FILE_ID = "AN_ID_LARGO_Y_COMPLEJO"
URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"


@st.cache_data  # Es muy importante usar caché para un archivo de 183 MB
def load_data():
    return pd.read_csv(URL)


df = load_data()
geo_ap = gpd.read_file('AndhraPradesh_Districts.geojson')
gep_tg = gpd.read_file('telangana.geojson')
geo_ap = geo_ap.dissolve(by='district_name', as_index=False)

# --- TÍTULO PRINCIPAL ---
st.markdown(
"""
<header class="hero">
    <h1 class="bold-title">Programa NTR Vaidya Seva</h1>
    <p class="tagline">Análisis de Gestión Financiera</p>
</header>
""",
unsafe_allow_html=True,
)

estados = {'Andhra Pradesh': '#9fc5e8', 'Telangana': '#9dc1b0'}
PALETA_MAESTRA = {
    'Primario_Lavanda': '#8fa1cd',
    'Secundario_Menta': '#cae6d4',
    'Neutro_Oscuro': '#5a5a5a',
    'Neutro_Claro': '#e2e5ec',
    'Alerta_Terracota': '#dd9993',
    'Alerta_Naranjita': '#ffd0b3',
    'Alerta_Salmon': '#ffd8c8'
}

PALETA_DISTRITOS = {
    'Krishna': '#9fc5e8',
    'Guntur': '#f4d7b5',
    'Hyderabad': '#8fa1cd',
    'Visakhapatnam': '#f3bccc',
    'Nellore': '#cdeee1',
    'East Godavari': '#c27ba0',
    'Chittoor': '#dd9993',
    'Kurnool': '#9dc1b0',
    'Rangareddy': '#9fc5e8',        # Azul suave (derivado del primario)
    'Vizianagaram': '#cdeee1',        # Verde menta claro (derivado del secundario)
    'West Godavari': '#c27ba0',  # Variante de la gama del morado apagado
    'Karimnagar': '#9dc1b0',        # Verde oliva / Salvia
    'Srikakulam': '#f7bfba'
}

# Configuración base para fuentes ejecutivas de Plotly
TEMPLATE_CORPORATIVO = dict(
    layout=go.Layout(
        title_font=dict(color=PALETA_MAESTRA['Neutro_Oscuro'], family="Arial", size=20, weight=900),
        font=dict(color=PALETA_MAESTRA['Neutro_Oscuro'], family="Arial", size=16, weight=700),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='#f0f0f0', linecolor='lightgrey', tickfont=dict(color='dimgrey')),
        yaxis=dict(gridcolor='#f0f0f0', linecolor='lightgrey', tickfont=dict(color='dimgrey'))
    )
)

# ==============================================================================
# PIPELINE DE PROCESAMIENTO PREVIO (TARIFAS TEÓRICAS Y SCORE NABH EXTRAPOLADO)
# ==============================================================================
cirugias_a_graficar = [
    'Angioplastia coronaria con stent',
    'Angioplastia con stent adicional',
    'Cirugía de bypass coronario',
    'Bypass coronario con balón de contrapulsación (BCIAO)',
    'Sustitución de válvula mitral con prótesis'
]

# Forzar limpieza en la columna de control
df['Momento_Autorizacion'] = df['Momento_Autorizacion'].astype(str).str.strip()

# Extracción de tarifas teóricas sobre el flujo estándar
df_quantiles = df[df['SURGERY'].isin(cirugias_a_graficar)].groupby(['SURGERY', 'Momento_Autorizacion'], observed=False)['DOLARES_CLAIM'].median().reset_index()
df_tarifas_teoricas = df_quantiles[df_quantiles['Momento_Autorizacion'] == 'Autorización Previa (Flujo Estándar)'].copy()

if df_tarifas_teoricas.empty:
    df_tarifas_teoricas = df_quantiles[df_quantiles['Momento_Autorizacion'].str.contains('Previa', na=False)].copy()

df_tarifas_teoricas.columns = ['SURGERY', 'Momento_Autorizacion', 'tarifa_base_teorica']
df_tarifas_teoricas['tarifa_nabh_teorica'] = df_tarifas_teoricas['tarifa_base_teorica'] * 1.02

map_base = dict(zip(df_tarifas_teoricas['SURGERY'], df_tarifas_teoricas['tarifa_base_teorica']))
map_nabh = dict(zip(df_tarifas_teoricas['SURGERY'], df_tarifas_teoricas['tarifa_nabh_teorica']))

df['tarifa_base'] = df['SURGERY'].map(map_base)
df['tarifa_nabh'] = df['SURGERY'].map(map_nabh)
df['ratio_cobro'] = df['DOLARES_CLAIM'] / df['tarifa_base']

# Algoritmo de clasificación de Hospitales NABH Estimado
hospital_scoring = df.groupby('HOSP_NAME')['ratio_cobro'].quantile(0.75).reset_index()
hospital_scoring['TIENE_NABH'] = (hospital_scoring['ratio_cobro'] > 1.01).astype(int)
mapeo_nabh_distrito = dict(zip(hospital_scoring['HOSP_NAME'], hospital_scoring['TIENE_NABH']))
df['TIENE_NABH'] = df['HOSP_NAME'].map(mapeo_nabh_distrito)

# Limpieza institucional de nombres de hospitales
hospital_names_clean = {
    'usha cardiac centre limited': 'Usha Cardiac Centre Limited',
    'pinnamaneni institue of medical sciences': 'Pinnamaneni Institute of Medical Sciences',
    'praveen cardiac centre': 'Praveen Cardiac Centre',
    'help hospitals pvt ltd': 'Help Hospitals Pvt Ltd',
    'andhrahospitalsvijayawadapvtltd': 'Andhra Hospitals Vijayawada Pvt Ltd',
    'dr.ramesh cardiac and multispeciality hospital ltd': 'Dr. Ramesh Cardiac and Multispeciality Hospital Ltd',
    'aayush nri lepl healthcare pvt ltd': 'Aayush NRI LEPL Healthcare Pvt Ltd',
    'sentini hospitals private limited': 'Sentini Hospitals Private Limited',
    'kamineni hospital': 'Kamineni Hospital',
    'heart care centre': 'Heart Care Centre',
    'sunrise hospitals': 'Sunrise Hospitals'
}
df['HOSP_NAME'] = df['HOSP_NAME'].replace(hospital_names_clean)

# --- SECCIÓN 1: MÉTRICAS (KPIs) ---
cols = st.columns(4)
registros = len(df)
cardiaco = (df[df['CATEGORY_NAME']=='Cirugía Cardíaca y Cardiotorácica']['DOLARES_CLAIM'].sum()/df['DOLARES_CLAIM'].sum()*100).round(2)
telangana = (df[(df['HOSP_STATE']=='Telangana')&(df['CATEGORY_NAME']=='Cirugía Cardíaca y Cardiotorácica')]['DOLARES_CLAIM'].sum()/df[df['HOSP_STATE'] == 'Telangana']['DOLARES_CLAIM'].sum()*100).round(1)
limite = (df[(df['dias_liquidacion']>90)&(df['CATEGORY_NAME']=='Cirugía Cardíaca y Cardiotorácica')]['ID'].count()/df[df['CATEGORY_NAME']=='Cirugía Cardíaca y Cardiotorácica']['ID'].count()*100).round(0)

metrics = [
    ("Registros analizados", f"{registros:,.0f}", "cirugías y reclamos en 2017"),
    ("Gasto en cirugía cardíaca", f"{cardiaco:.1f}%", "del gasto total del programa"),
    ("Fondos a Telangana", f"{telangana:.1f}%", "de la fuga total, vía cardíacas"),
    ("Tiempo de liquidación", f"{limite:.0f}%", "de cirugías cardíacas demoran más de tres meses")
]

for col, (title, value, texto) in zip(cols, metrics):
    with col:
        st.markdown(f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="stat-value">{value}</div>
                <div class="text-card-body">{texto}</div>
            </div>
        """, unsafe_allow_html=True)



tab_contexto, tab_infraest, tab_cardio, tab_telangana, tab_liquidacion, tab_simulador = st.tabs([
    'Introducción', 'Diagnóstico General', 'Cirugía Cardíaca y Cardiotorácica',
    'Movilidad sanitaria', 'Análisis de Tiempos de Liquidación', 'Simulador de Tiempos de Demora'
])

with tab_contexto:
    """## **1. Introducción y Marco Operativo del Programa**"""
    st.html(
        """
        <div class="text-column-container">
            <div class="dashboard-card">
                <p class="text-card-body">
                    El programa NTR Vaidya Seva es la iniciativa insignia del gobierno de Andhra Pradesh diseñada para brindar atención médica gratuita a ciudadanos de bajos recursos para enfermedades críticas. El dataset analizado abarca aproximadamente 480,000 registros e incluye información sobre pacientes (demografía, casta, residencia), cirugías (especialidad, tipo), hospitales (ubicación, sector público/privado) y datos financieros (preautorización y reclamos). 
                    El funcionamiento del sistema digital se articula en dos etapas principales que garantizan la transparencia y el control del proceso médico-financiero:
                </p>
            </div>
        </div>
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.html(
                f"""
                <div class="text-card">
                    <h4 class="text-card-title">FASE 1: PRE AUTORIZACIÓN ELECTRÓNICA</h4>
                    <div class="card-image-container">
                        <img src="data:image/png;base64,{img_fase1_base64}" alt="Secuencia 1">
                    </div>
                </div>
                """
            )
    with col2:
        st.html(
                f"""
                <div class="text-card">
                    <h4 class="text-card-title">Fase 2: Gestión de Reclamos y Liquidación</h4>
                    <div class="card-image-container">
                        <img src="data:image/png;base64,{img_fase2_base64}" alt="Secuencia 1">
                    </div>
                </div>
                """
            )

with tab_infraest:
        """## **2. Diagnóstico Analítico y Hallazgos Estadísticos**"""
        col1, col2 = st.columns([60, 40])  
        with col1:
            district_standardization_map = {
                'anantapur': 'Anantapur', 'ananthapuramu': 'Anantapur', 'vishakhapatnam': 'Visakhapatnam',
                'nellore': 'Nellore', 'sri potti sriramulu nellore': 'Nellore', 'Ysr Kadapa': 'Ysr Kadapa',
                'YSR Kadapa': 'YSR Kadapa', 'chittoor': 'Chittoor', 'east godavari': 'East Godavari',
                'guntur': 'Guntur', 'krishna': 'Krishna', 'kurnool': 'Kurnool', 'prakasam': 'Prakasam',
                'srikakulam': 'Srikakulam', 'vizianagaram': 'Vizianagaram', 'vijayanagaram': 'Vizianagaram',
                'west godavari': 'West Godavari', 'hyderabad': 'Hyderabad', 'rangareddy': 'Rangareddy',
                'ranga reddy': 'Rangareddy', 'khammam': 'Khammam', 'warangal': 'Warangal',
                'karimnagar': 'Karimnagar', 'nizamabad': 'Nizamabad', 'mahbubnagar': 'Mahbubnagar'
            }

            df_ap_patients = df[df['HOSP_STATE'] == 'Andhra Pradesh'].copy()
            df_districts_for_map = df_ap_patients['DISTRICT_NAME'].map(district_standardization_map).fillna(df_ap_patients['DISTRICT_NAME'].str.title())

            patient_district_counts = df_districts_for_map.value_counts().reset_index()
            patient_district_counts.columns = ['District', 'Count']

            geo_ap_temp = geo_ap.copy()
            geo_ap_temp['District'] = geo_ap_temp['district_name'].str.lower().map(district_standardization_map).fillna(geo_ap_temp['district_name'].str.title())

            combined_geo_df = geo_ap_temp
            combined_geo_df = combined_geo_df.merge(patient_district_counts, on='District', how='left')
            combined_geo_df['Count'] = combined_geo_df['Count'].fillna(0)
            total_ap_patients = df_ap_patients['ID'].nunique()
            combined_geo_df['Porcentaje de Pacientes'] = (combined_geo_df['Count'] / total_ap_patients * 100).round(2)
        
            fig = px.choropleth_map(
                combined_geo_df, geojson=combined_geo_df.geometry, locations=combined_geo_df.index, height=400,
                color='Count', hover_name='District', hover_data={'Count': True, 'Porcentaje de Pacientes': ':.0f%'},
                custom_data=['Porcentaje de Pacientes', 'Count'], color_continuous_scale="Blues", zoom=5,
                center={"lat": 16, "lon": 79.5}, labels={'Count': 'Número de Pacientes'}, opacity=0.5,
                title='Número de Pacientes por Distrito en Andhra Pradesh'
            )
            fig.update_traces(hovertemplate="<b>%{hovertext}</b><br><br>Número de Pacientes=%{customdata[1]:.0s}<br>Porcentaje de Pacientes=%{customdata[0]:.0f}%<extra></extra>")
            fig.update_layout(
                font_color="lightslategray", font_family='Arial', title_font_color="dimgrey",
                title=dict(font=dict(size=20, weight=900)), font=dict(size=16, weight=700), title_x=0.2, title_y=0.95,
                margin=dict(t=60, l=40, r=40, b=40)
            )
            st.plotly_chart(fig, width='stretch', key='chart-mapa')
          
        with col2:
            grouped_by_district_type = df.groupby('HOSP_DISTRICT').agg(
                total_surgeries_district_type=('SURGERY', 'count'),
                total_claims_district_type=('DOLARES_CLAIM', 'sum'),
                sum_hospitals=('HOSP_NAME', 'nunique')
            ).reset_index()

            total_program_surgeries = grouped_by_district_type['total_surgeries_district_type'].sum()
            total_program_claims = grouped_by_district_type['total_claims_district_type'].sum()

            grouped_by_district_type['percent_of_total_surgeries'] = (grouped_by_district_type['total_surgeries_district_type'] / total_program_surgeries * 100).round(2)
            grouped_by_district_type['percent_of_total_claims'] = (grouped_by_district_type['total_claims_district_type'] / total_program_claims * 100).round(2)

            distritos_80 = ['Guntur', 'Krishna', 'Visakhapatnam', 'East Godavari', 'Chittoor', 'Nellore', 'Hyderabad', 'Kurnool', 'West Godavari']
            df_melt = grouped_by_district_type[grouped_by_district_type['HOSP_DISTRICT'].isin(distritos_80)].copy()

            df_melt_long = df_melt.melt(
                id_vars=['HOSP_DISTRICT', 'total_surgeries_district_type', 'total_claims_district_type'],
                value_vars=['percent_of_total_surgeries', 'percent_of_total_claims'],
                var_name='Type_Of_Percent', value_name='percent_Of_Cases'
            )

            df_melt_long['legend_label'] = df_melt_long['Type_Of_Percent'].map({
                'percent_of_total_surgeries': 'Cirugías (%)', 'percent_of_total_claims': 'Reclamo (%)'
            })

            df_melt_long['Valor_Absoluto'] = df_melt_long.apply(
                lambda r: f"{r['total_surgeries_district_type']:,} cirugías" if 'surgeries' in r['Type_Of_Percent'] else f"${r['total_claims_district_type']:,.2f}", axis=1
            )

            fig1 = px.bar(
                df_melt_long.sort_values(by='percent_Of_Cases', ascending=True),
                y='HOSP_DISTRICT', x='percent_Of_Cases', color='legend_label', barmode='group',
                color_discrete_sequence=[PALETA_MAESTRA['Primario_Lavanda'], PALETA_MAESTRA['Secundario_Menta']],
                labels={'percent_Of_Cases': 'Porcentaje del Total', 'HOSP_DISTRICT': 'Distrito'}, custom_data=['Valor_Absoluto']
            )
            fig1.update_traces(
                texttemplate='%{x:.1f}%', textposition='outside', textfont=dict(color='dimgrey', size=10),
                hovertemplate="<b>Distrito:</b> %{y}<br><b>Proporción:</b> %{x:.2f}%<br><b>Muestra Absoluta:</b> %{customdata[0]}<extra></extra>"
            )
            fig1.update_layout(
                template=TEMPLATE_CORPORATIVO,
                height=400,
                title="<b>Distribución Porcentual por Distrito</b><br><sub>Total Reclamado vs. Total de Cirugías</sub>",
                legend_title="", xaxis=dict(ticksuffix="%"), margin=dict(t=60, l=40, r=40, b=40), title_x=0.2, title_y=0.95,
                font_color="lightslategray", font_family='Arial', title_font=dict(size=20, weight=900), title_font_color="dimgrey",
                font=dict(size=16, weight=700), legend=dict(font=dict(color='dimgrey'))
            )
            st.plotly_chart(fig1, width='stretch', key='chart-gasto')
        with st.container(height=20, border=False):
            pass # El contenedor está vacío para generar espacio  
        col3, col4 = st.columns(2)  
        with col3:
            df_hosp_type = df.groupby(['HOSP_TYPE', 'HOSP_DISTRICT']).size().reset_index(name='count')
            district_stats = df.groupby('HOSP_DISTRICT').agg(
                total_cirugias=('HOSP_NAME', 'count'),
                total_hospitales=('HOSP_NAME', 'nunique')
            ).reset_index()

            top10_districts = district_stats.nlargest(10, 'total_cirugias')['HOSP_DISTRICT'].tolist()
            df_filtered = df_hosp_type[df_hosp_type['HOSP_DISTRICT'].isin(top10_districts)].copy()

            df_pivot = df_filtered.pivot(index='HOSP_DISTRICT', columns='HOSP_TYPE', values='count').fillna(0).reindex(top10_districts)
            df_perc = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
            df_perc = df_perc.reset_index()

            df_perc = df_perc.merge(district_stats, on='HOSP_DISTRICT')

            # Orden inverso para que el top del volumen aparezca arriba en la vista horizontal de Plotly
            df_perc = df_perc.iloc[::-1]

            fig2 = go.Figure()
            hosp_types = [c for c in df_pivot.columns]
            colores_stack = [PALETA_MAESTRA['Primario_Lavanda'], PALETA_MAESTRA['Secundario_Menta']]

            for i, h_type in enumerate(hosp_types):
                fig2.add_trace(go.Bar(
                    y=df_perc['HOSP_DISTRICT'], x=df_perc[h_type], name=h_type,
                    orientation='h', marker_color=colores_stack[i % len(colores_stack)],
                    customdata=np.stack((df_perc[f'{h_type}'], df_perc['total_cirugias'],df_perc['total_hospitales']), axis=-1),
                    texttemplate='%{x:.1f}%', textposition='inside', insidetextanchor='middle',
                    textfont=dict(color='dimgrey', size=10, weight='bold'),
                    hovertemplate=f"<b>Distrito:</b> %{{y}}<br><b>Tipo:</b> {h_type}<br><b>Porcentaje:</b> %{{x:.2f}}%<br><b>Hospitales:</b> %{{customdata[2]:,}}<br><b>Total Muestra:</b> %{{customdata[1]:,}}<extra></extra>"
                ))

            fig2.update_layout(
                template=TEMPLATE_CORPORATIVO, barmode='stack',
                height=400,
                title="<b>Distribución Porcentual Tipo de Hospital por Distrito</b><br><sub>Top 10 Distritos por Cantidad de Cirugías</sub>",
                xaxis=dict(ticksuffix="%"), legend_title="", margin=dict(t=60, l=40, r=40, b=40),title_x=0.2, title_y=0.95,
                font_color="lightslategray",
                font_family='Arial',
                title_font=dict(size=20, weight=900),
                title_font_color="dimgrey",
                font=dict(size=16, weight=700),
                legend=dict(font=dict(color='dimgrey'))
            )
            st.plotly_chart(fig2, width='stretch', key='chart-infra')
        with col4:
            total_patients_category = df['ID'].nunique()
            category_summary = df.groupby('CATEGORY_NAME').agg(
                patient_count=('ID', 'nunique'),
                total_claim_amount=('DOLARES_CLAIM', 'sum')
            ).reset_index()

            category_summary['percentage'] = (category_summary['patient_count'] / total_patients_category * 100).round(2)
            total_program_claim_amount = category_summary['total_claim_amount'].sum()
            category_summary['percentage_of_claim_amount'] = (category_summary['total_claim_amount'] / total_program_claim_amount * 100).round(2)

            category_summary_by_claim = category_summary.sort_values(by='percentage_of_claim_amount', ascending=False)
            category_summary_by_claim['acumulado'] = category_summary_by_claim['percentage_of_claim_amount'].cumsum()

            df_melt_cat = category_summary_by_claim[category_summary_by_claim['acumulado'].shift(1).fillna(0) < 80].copy()
            df_melt_cat_long = df_melt_cat.melt(
                id_vars=['CATEGORY_NAME', 'patient_count', 'total_claim_amount'],
                value_vars=['percentage', 'percentage_of_claim_amount'],
                var_name='Type_Of_Percent', value_name='percent_Of_Cases'
            )
            df_melt_cat_long['legend_label'] = df_melt_cat_long['Type_Of_Percent'].map({'percentage': 'Cirugías (%)', 'percentage_of_claim_amount': 'Reclamo (%)'})
            df_melt_cat_long['Valor_Absoluto'] = df_melt_cat_long.apply(
                lambda r: f"{r['patient_count']:,} cirugías" if 'percentage' == r['Type_Of_Percent']
                else f"${r['total_claim_amount']:,.2f}", axis=1
            )

            fig3 = px.bar(
                df_melt_cat_long.sort_values(by='percent_Of_Cases', ascending=True),
                y='CATEGORY_NAME', x='percent_Of_Cases',
                color='legend_label', barmode='group',
                color_discrete_sequence=[PALETA_MAESTRA['Primario_Lavanda'], PALETA_MAESTRA['Secundario_Menta']],
                labels={'percent_Of_Cases': 'Porcentaje del Total', 'CATEGORY_NAME': ''},
                custom_data=['Valor_Absoluto']
            )
            fig3.update_traces(
                texttemplate='%{x:.1f}%', textposition='outside', textfont=dict(color='dimgrey', size=10),
                hovertemplate="<b>Especialidad:</b> %{y}<br><b>Proporción:</b> %{x:.2f}%<br><b>Muestra Absoluta:</b> %{customdata[0]}<extra></extra>"
            )
            fig3.update_layout(
                template=TEMPLATE_CORPORATIVO,
                height=400,
                title="<b>Distribución Porcentual por Especialidad</b><br><sub>Total Reclamado vs. Total de Cirugías (Especialidades que acumulan el 80% del Monto)</sub>",
                legend_title="", bargap=0.15, xaxis=dict(ticksuffix="%"), margin=dict(t=60, l=40, r=40, b=40),title_x=0.2, title_y=0.95,
                legend=dict(font=dict(color='dimgrey')),
                font_color="lightslategray",
                font_family='Arial',
                title_font=dict(size=20, weight=900),
                title_font_color="dimgrey",
                font=dict(size=16, weight=700),

            )
            st.plotly_chart(fig3, width='stretch', key='chart-especialidad')
    
with tab_cardio:
    """## **3. Análisis Especializado: Cirugía Cardíaca y Cardiotorácica**"""
    target_category = "Cirugía Cardíaca y Cardiotorácica"

    # Filter the DataFrame for the target category
    df_cardio = df[df["CATEGORY_NAME"] == target_category].copy()

    grouped_cardio_location = (
        df_cardio.groupby(["HOSP_LOCATION", "HOSP_DISTRICT"])
        .agg(
            cantidad_SURGERY=("SURGERY", "count"),
            media_DOLARES_CLAIM=("DOLARES_CLAIM", "mean"),
        )
        .reset_index()
        .round(2)
    )

    # Calculate the total number of surgeries in df_cardio for percentage calculation
    total_surgeries_cardio = df_cardio["SURGERY"].count()

    # Calculate the percentage of total surgeries for each location
    grouped_cardio_location["Porcentaje del Total"] = (
        grouped_cardio_location["cantidad_SURGERY"] / total_surgeries_cardio * 100
    ).round(2)

    # Sort by cantidad_SURGERY in descending order for plotting
    grouped_cardio_location_sorted = grouped_cardio_location.sort_values(
        by="cantidad_SURGERY", ascending=False
    ).head(10)

    # --- CORRECCIÓN 1: Creamos la columna con las etiquetas deseadas en el DataFrame ---
    grouped_cardio_location_sorted["Y_LABEL"] = [
        f"{row['HOSP_LOCATION']}"  # Puedes usar '<br>' en vez de '\n' si prefieres salto de línea HTML en Plotly
        for _, row in grouped_cardio_location_sorted.iterrows()
    ]

    # Guardamos la lista exacta para mantener el orden de categorías
    y_labels_order = grouped_cardio_location_sorted["Y_LABEL"].tolist()

    fig4 = px.bar(
        grouped_cardio_location_sorted,
        y="Y_LABEL",  # <--- CORRECCIÓN 2: Graficamos la columna con el formato idéntico
        x="Porcentaje del Total",
        color="HOSP_DISTRICT",
        color_discrete_map=PALETA_DISTRITOS,
        labels={"Porcentaje del Total": "Porcentaje del Total (%)", "Y_LABEL": ""},
        custom_data=["cantidad_SURGERY", "media_DOLARES_CLAIM", "HOSP_DISTRICT"],
    )

    fig4.update_traces(
        texttemplate="%{x:.1f}%",
        textposition="outside",
        textfont=dict(color="dimgrey", size=10),
        hovertemplate="<b>Municipio:</b> %{y}<br><b>Distrito:</b> %{customdata[2]}<br><b>Porcentaje:</b> %{x:.2f}%<br><b>Cirugías:</b> %{customdata[0]:,}<br><b>Media Reclamo:</b> %{customdata[1]:$,.2f}<extra></extra>",
    )

    fig4.update_layout(
        template=TEMPLATE_CORPORATIVO,
        title=f"<b>Distribución Porcentual de Cirugías por Municipio Hospitalario</b><br><sub>Top 10 de Municipios en {target_category}</sub>",
        legend_title="Distrito Sede",
        height=400,
        xaxis=dict(ticksuffix="%"),
        title_x=0.075, title_y=0.95,
        font_color="lightslategray",
        font_family='Arial',
        title_font=dict(size=20, weight=900),
        title_font_color="dimgrey",
        font=dict(size=16, weight=700),
        # Ahora 'categoryarray' coincide exactamente con los valores del eje Y
        yaxis={
            "categoryorder": "array",
            "categoryarray": y_labels_order,
            "autorange": "reversed",
        },
        legend=dict(font=dict(color='dimgrey'))
    )
    st.plotly_chart(fig4, width='stretch', key='chart-cardio1')
    with st.container(height=20, border=False):
        pass # El contenedor está vacío para generar espacio    
    col1, col2 = st.columns(2)
    with col1:
        # 1. Filtrar por categoría, agrupar por cirugía y sumar claims directamente
        surgery_claims = (
            df[df['CATEGORY_NAME'] == target_category]
            .groupby('SURGERY')['DOLARES_CLAIM']
            .sum()
            .reset_index(name='Total_DOLARES_CLAIM')
        )

        # 2. Ordenar de mayor a menor según el gasto (Proporción de Claims)
        surgery_claims = surgery_claims.sort_values(by='Total_DOLARES_CLAIM', ascending=False).reset_index(drop=True)

        # 3. Calcular porcentaje individual y acumulado
        total_claim_cardio = surgery_claims['Total_DOLARES_CLAIM'].sum()
        surgery_claims['Percentage'] = (surgery_claims['Total_DOLARES_CLAIM'] / total_claim_cardio) * 100
        surgery_claims['Cumulative_Percentage'] = surgery_claims['Percentage'].cumsum()

        # 4. Filtrar cirugías que acumulan hasta el 80% del total
        # Evaluamos el acumulado de la fila anterior (.shift) para incluir la categoría que rompe el umbral del 80%
        df_pareto_80 = surgery_claims[surgery_claims['Cumulative_Percentage'].shift(1).fillna(0) < 80]

        # Recrear el gráfico en Plotly
        fig_pareto = px.bar(
            data_frame=df_pareto_80,
            y='SURGERY',
            x='Percentage',
            orientation='h',
            color_discrete_sequence=[PALETA_MAESTRA['Primario_Lavanda']],
            labels={
                'Percentage': 'Porcentaje del Gasto Total (%)',
                'SURGERY': 'Cirugía'
            },
            custom_data=['Total_DOLARES_CLAIM', 'Percentage']
        )

        fig_pareto.update_traces(
            texttemplate='%{x:.1f}%',
            textposition='outside',
            textfont=dict(color='dimgrey', size=10),
            hovertemplate="<b>Cirugía:</b> %{y}<br><b>Porcentaje:</b> %{customdata[1]:.2f}%<br><b>Monto Total:</b> %{customdata[0]:$,.2f}<extra></extra>"
        )

        fig_pareto.update_layout(
            template=TEMPLATE_CORPORATIVO,
            yaxis_title='',
            height=400,
            title=f"<b>Porcentaje del Gasto Total por Cirugía</b><br><sub>{target_category} que acumulan el 80% de los Reclamos en Dólares</sub>",
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(categoryorder='total ascending'), # To ensure the largest bars are at the top
            margin=dict(t=60, l=40, r=40, b=40),
            title_x=0.25, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700),
        )
        st.plotly_chart(fig_pareto, width='stretch', key='chart-cardio2')

    with col2: 
        surgeries_to_filter = ['Angioplastia coronaria con stent', 'Cirugía de bypass coronario', 'Angioplastia con stent adicional', 'Sustitución de válvula mitral con prótesis', 'Bypass coronario con balón de contrapulsación (BCIAO)']
        df_cardio_selected_surgeries = df[df['SURGERY'].isin(surgeries_to_filter)].sort_values(by='SURGERY', ascending=False)

        # Convert to Plotly
        fig_box = px.box(
            data_frame=df_cardio_selected_surgeries,
            x='DOLARES_CLAIM',
            y='SURGERY', points=False,
            color_discrete_sequence=[PALETA_DISTRITOS['Krishna']], # Using the exact color from the original seaborn plot
            labels={
                'DOLARES_CLAIM': 'Reclamo en Dólares',
                'SURGERY': 'Cirugía'
            },
            title=f"<b>Distribución de Monto de Reclamo por Cirugía</b><br><sub>{target_category} que acumulan el 80% de los Reclamos en Dólares</sub>"
        )

        fig_box.update_layout(
            template=TEMPLATE_CORPORATIVO,
            xaxis_title='Reclamo en Dólares',
            yaxis_title='',
            # Adjusting margins for better title and label visibility
            margin=dict(t=60, l=40, r=40, b=40),
            # Ensure y-axis order matches the seaborn plot (descending alphabetical for SURGERY)
            yaxis={'categoryorder': 'array', 'categoryarray': df_cardio_selected_surgeries['SURGERY'].drop_duplicates().tolist()},
            title_x=0.25, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            height=400,
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700),
        )

        fig_box.update_traces(
            marker_line_color=PALETA_MAESTRA['Neutro_Oscuro'], # Corresponding to seaborn's linecolor
            selector=dict(type='box'),
            hovertemplate="<b>Cirugía:</b> %{y}<br><b>Monto Reclamado:</b> %{x:$,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_box, width='stretch', key='chart-cardio3')
        with st.container(height=20, border=False):
            pass # El contenedor está vacío para generar espacio    
        # Flujo de Pacientes por distrito

with tab_telangana:
    """## **4. Dinámicas Territoriales y Movilidad Sanitaria**"""
    col3, col4 = st.columns([60,40])
    with col3:
        unified_district_mapping = {
            'Anantapur': 'Ananthapuramu', 'Vishakhapatnam': 'Visakhapatnam', 'Nellore': 'Sri Potti Sriramulu Nellore',
            'Ysr Kadapa': 'YSR Kadapa', 'Chittoor': 'Chittoor', 'East Godavari': 'East Godavari', 'Guntur': 'Guntur',
            'Krishna': 'Krishna', 'Kurnool': 'Kurnool', 'Prakasam': 'Prakasam', 'Srikakulam': 'Srikakulam',
            'Vizianagaram': 'Vizianagaram', 'West Godavari': 'West Godavari', 'Vijayanagaram': 'Vizianagaram'
        }

        surgeries_to_filter = ['Angioplastia coronaria con stent', 'Cirugía de bypass coronario', 'Angioplastia con stent adicional', 'Sustitución de válvula mitral con prótesis', 'Bypass coronario con balón de contrapulsación (BCIAO)']
        df_cardio_selected_surgeries = df[df['SURGERY'].isin(surgeries_to_filter)].copy()

        df_temp = df_cardio_selected_surgeries.copy()
        df_temp['DISTRICT_NAME_CLEANED'] = df_temp['DISTRICT_NAME'].replace(unified_district_mapping)
        df_temp['HOSP_DISTRICT_CLEANED'] = df_temp['HOSP_DISTRICT'].replace(unified_district_mapping)

        pivot_flujos = pd.crosstab(df_temp['DISTRICT_NAME_CLEANED'], df_temp['HOSP_DISTRICT_CLEANED'], normalize='index') * 100
        raw_districts_to_exclude = ['Khammam', 'Karimnagar', 'Rangareddy']
        pivot_flujos_filtered = pivot_flujos[~pivot_flujos.index.isin(raw_districts_to_exclude)]
        pivot_flujos_filtered = pivot_flujos_filtered.loc[:, ~pivot_flujos_filtered.columns.isin(raw_districts_to_exclude)]

        distritos_ap = ['Ananthapuramu', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna', 'Kurnool', 'Prakasam', 'Sri Potti Sriramulu Nellore', 'Srikakulam', 'Visakhapatnam', 'Vizianagaram', 'West Godavari', 'YSR Kadapa']
        eje_x_orden_estricto = distritos_ap.copy() + ["Hyderabad"]
        eje_y_orden_estricto = distritos_ap.copy()

        df_balloon = pivot_flujos_filtered.stack().reset_index(name='Porcentaje')
        df_balloon = df_balloon[df_balloon['Porcentaje'] > 5].copy()
        df_balloon['HOSP_DISTRICT_CLEANED'] = df_balloon['HOSP_DISTRICT_CLEANED'].astype(str).str.strip()
        df_balloon['DISTRICT_NAME_CLEANED'] = df_balloon['DISTRICT_NAME_CLEANED'].astype(str).str.strip()

        mapa_x = {distrito: i for i, distrito in enumerate(eje_x_orden_estricto)}
        mapa_y = {distrito: i for i, distrito in enumerate(eje_y_orden_estricto)}
        df_balloon['X_coor'] = df_balloon['HOSP_DISTRICT_CLEANED'].map(mapa_x)
        df_balloon['Y_coor'] = df_balloon['DISTRICT_NAME_CLEANED'].map(mapa_y)
        df_balloon = df_balloon.dropna(subset=['X_coor', 'Y_coor'])

        df_balloon['Color_Hex'] = df_balloon['HOSP_DISTRICT_CLEANED'].apply(
            lambda d: PALETA_MAESTRA['Alerta_Terracota'] if "Hyderabad" in str(d) else PALETA_MAESTRA['Primario_Lavanda']
        )
        df_balloon['Tipo_Flujo'] = df_balloon['HOSP_DISTRICT_CLEANED'].apply(
            lambda d: 'Derivación Externa (Telangana)' if "Hyderabad" in str(d) else 'Retención/Tráfico Interno AP'
        )
        fig5 = px.scatter(
            df_balloon, x='HOSP_DISTRICT_CLEANED', y='DISTRICT_NAME_CLEANED',
            size='Porcentaje', color='Tipo_Flujo',
            color_discrete_map={
                'Derivación Externa (Telangana)': PALETA_MAESTRA['Alerta_Terracota'],
                'Retención/Tráfico Interno AP': PALETA_MAESTRA['Primario_Lavanda']
            },
            labels={'HOSP_DISTRICT_CLEANED': 'Distrito del Hospital (Destino)', 'DISTRICT_NAME_CLEANED': 'Residencia del Paciente (Origen)'},
            size_max=25
        )
        fig5.update_yaxes(categoryorder='array', categoryarray=eje_y_orden_estricto[::-1])
        fig5.update_xaxes(categoryorder='array', categoryarray=eje_x_orden_estricto)
        fig5.update_traces(
            hovertemplate="<b>Origen:</b> %{y}<br><b>Destino:</b> %{x}<br><b>Porcentaje de Flujo:</b> %{marker.size:.1f}%<extra></extra>"
        )
        fig5.update_layout(
            template=TEMPLATE_CORPORATIVO,
            title="<b>Matriz de Flujo de Pacientes por Distrito</b>",
            legend_title="", 
            height=400,
            margin=dict(t=60, l=40, r=40, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.75,
                font=dict(color='dimgrey')
            ),
            title_x=0.1, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700)
        )
        st.plotly_chart(fig5, width='stretch', key='chart-flujo')

    with col4:
        # Filter df_balloon for self-retention cases (origin district == hospital district)
        self_retention_data = df_balloon[df_balloon['DISTRICT_NAME_CLEANED'] == df_balloon['HOSP_DISTRICT_CLEANED']].copy()

        # Prepare geo_ap for merging (standardize district names if necessary, similar to previous choropleth)
        geo_ap_temp = geo_ap.copy()
        geospatial_district_names = geo_ap_temp['district_name'].str.title().unique()

        # Map the cleaned district names to ensure consistency for merging
        self_retention_data['District'] = self_retention_data['DISTRICT_NAME_CLEANED'].map(lambda x: x.title() if x.title() in geospatial_district_names else x)

        # Rename 'district_name' to 'District' in geo_ap_temp to match self_retention_data for merging
        geo_ap_temp['District'] = geo_ap_temp['district_name'].str.title()

        # Merge self-retention data with the geojson DataFrame
        combined_geo_df_retention = geo_ap_temp.merge(self_retention_data[['District', 'Porcentaje']], on='District', how='left')
        combined_geo_df_retention['Porcentaje'] = combined_geo_df_retention['Porcentaje'].fillna(0).round(2)

        # Plot the choropleth map
        fig_mapa = px.choropleth_map(
            combined_geo_df_retention,
            geojson=combined_geo_df_retention.geometry,
            locations=combined_geo_df_retention.index,
            color='Porcentaje',
            hover_name='District',
            hover_data={'Porcentaje': ':.0f%'}, # Format percentage for hover
            color_continuous_scale="Blues",
            zoom=5,
            center={"lat": 16, "lon": 79.5},
            labels={'Porcentaje': 'Retención (%)'},
            opacity=0.7,
        )
        fig_mapa.update_traces(hovertemplate="<b>%{hovertext}</b><br>Porcentaje de Pacientes=%{customdata[0]:.0f}%<extra></extra>")
        fig_mapa.update_layout(
            template=TEMPLATE_CORPORATIVO,
            height=400,
            title="<b>Porcentaje de Retención de Pacientes por Distrito</b><br><sub>Cirugías Cardíacas Seleccionadas</sub>",
            font_color="lightslategray",
            font_family='Arial',
            title_font_color="dimgrey",
            title_x=0.1, title_y=0.95,
            margin=dict(t=60, l=40, r=40, b=40),
            font=dict(size=16, weight=700),
            title_font=dict(size=20, weight=900),
            legend=dict(font=dict(color='dimgrey'))
        )
        st.plotly_chart(fig_mapa, width='stretch', key='chart-derivacion')

    col1, col2 = st.columns([60,40])
    with col1: 

        df_crosstab_state_surgery = pd.crosstab(df_cardio_selected_surgeries['HOSP_STATE'], df_cardio_selected_surgeries['SURGERY'], normalize='columns') * 100
        df_plot_crosstab = df_crosstab_state_surgery.reset_index().melt(id_vars='HOSP_STATE', var_name='SURGERY', value_name='Percentage')

        # Obtenemos orden desde el Pareto previo
        surgery_claims = df_cardio_selected_surgeries.groupby('SURGERY')['DOLARES_CLAIM'].sum().reset_index(name='Total')
        surgery_order = surgery_claims.sort_values(by='Total', ascending=False)['SURGERY'].tolist()

        estados_colores = {'Andhra Pradesh': '#9fc5e8', 'Telangana': '#9dc1b0'}

        fig6 = px.bar(
            df_plot_crosstab, y='SURGERY', x='Percentage', color='HOSP_STATE',
            barmode='relative', color_discrete_map=estados_colores,
            labels={'Percentage': 'Porcentaje (%)', 'SURGERY': 'Cirugía Realizada'}
        )
        fig6.update_yaxes(categoryorder='array', categoryarray=surgery_order)
        fig6.update_traces(
            texttemplate='%{x:.1f}%', textposition='inside', textfont=dict(color='dimgrey', size=9.5),
            hovertemplate="<b>Cirugía:</b> %{y}<br><b>Estado Hosp:</b> %{sidebar}<br><b>Participación Local:</b> %{x:.2f}%<extra></extra>"
        )
        fig6.update_layout(
            template=TEMPLATE_CORPORATIVO, bargap=0.15, xaxis=dict(ticksuffix="%"),
            title="<b>Distribución Porcentual de Cirugías por Estado</b><br><sub>Proporción de impacto sobre el total de la demanda sectorizada</sub>",
            legend_title="", margin=dict(t=60, l=40, r=40, b=40), yaxis=dict(autorange='reversed'),
            yaxis_title='',
            title_x=0.35, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700),
            legend=dict(font=dict(color='dimgrey'))
        )
        st.plotly_chart(fig6, width='stretch', key='chart-estados')


    with col2: 

        fuga_dinero = df[df['HOSP_STATE'] == 'Telangana']['DOLARES_CLAIM'].sum()
        fuga_dinero2 = df_cardio_selected_surgeries[df_cardio_selected_surgeries['HOSP_STATE'] == 'Telangana']['DOLARES_CLAIM'].sum()
        porc_fuga_dinero = (fuga_dinero2 * 100) / fuga_dinero

        categorias_fuga = ['Otras Especialidades<br>Exportadas', '5 Cirugías Cardíacas<br>Seleccionadas']
        porcentajes_fuga = [100 - porc_fuga_dinero, porc_fuga_dinero]

        fig7 = px.bar(
            y=categorias_fuga, x=porcentajes_fuga, color=categorias_fuga,
            color_discrete_map={
                'Otras Especialidades<br>Exportadas': PALETA_MAESTRA['Neutro_Claro'],
                '5 Cirugías Cardíacas<br>Seleccionadas': PALETA_MAESTRA['Alerta_Terracota']
            },
            orientation='h', labels={'x': 'Porcentaje (%)', 'y': 'Segmento Médico'}
        )
        fig7.update_traces(
            texttemplate='%{x:.2f}%', textposition='outside', textfont=dict(color='dimgrey', size=11, weight='bold'),
            hovertemplate="<b>Segmento:</b> %{y}<br><b>Participación Presupuestaria:</b> %{x:.2f}%<extra></extra>"
        )
        fig7.update_layout(
            template=TEMPLATE_CORPORATIVO, showlegend=False, xaxis=dict(ticksuffix="%"),
            title="<b>Concentración del Presupuesto de Fuga Geográfica</b><br><sub>Participación de las cirugías cardíacas dentro del total transferido a Telangana</sub>",
            margin=dict(t=60, l=40, r=40, b=40), yaxis=dict(autorange='reversed'),
            yaxis_title='',
            title_x=0.2, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700),
        )
        st.plotly_chart(fig7, width='stretch', key='chart-comparacion')

with tab_liquidacion:
    """## **5. Evaluación de Eficiencia: Tiempos de Liquidación y Cuellos de Botella**"""
    c1, c2 = st.columns([3, 1])
    with c1:
        category_claim_total = df.groupby('CATEGORY_NAME')['DOLARES_CLAIM'].sum()
        top8_claim_categories = category_claim_total.nlargest(8).index.tolist()
        df_filtered_tramos = df[df['CATEGORY_NAME'].isin(top8_claim_categories)].copy()

        bins_demora = [-np.inf, 90, 180, 365, np.inf]
        labels_demora = ['Hasta 3 meses (Flujo Estándar)', '3 a 6 meses (Demora Moderada)', '6 a 12 meses (Demora Alta)', 'Más de 1 año (Demora Crítica)']
        df_filtered_tramos['tramo_demora'] = pd.cut(df_filtered_tramos['dias_liquidacion'], bins=bins_demora, labels=labels_demora)

        df_matrix_tramos = pd.crosstab(df_filtered_tramos['CATEGORY_NAME'], df_filtered_tramos['tramo_demora'], normalize='index').mul(100)
        df_matrix_tramos = df_matrix_tramos.reindex(index=top8_claim_categories, columns=labels_demora).reset_index()

        # Ordenar por volumen de demoras altas para consistencia con tu código
        df_matrix_tramos = df_matrix_tramos.sort_values(by='6 a 12 meses (Demora Alta)', ascending=True)

        fig8 = go.Figure()
        colores_alerta = [PALETA_MAESTRA['Neutro_Claro'], PALETA_MAESTRA['Alerta_Naranjita'], PALETA_MAESTRA['Alerta_Salmon'], PALETA_MAESTRA['Alerta_Terracota']]

        for i, tramo in enumerate(labels_demora):
            fig8.add_trace(go.Bar(
                y=df_matrix_tramos['CATEGORY_NAME'], x=df_matrix_tramos[tramo], name=tramo,
                orientation='h', marker_color=colores_alerta[i],
                texttemplate='%{x:.1f}%', textposition='inside', insidetextanchor='middle',
                textfont=dict(color='dimgrey', size=9.5, weight='bold'),
                hovertemplate=f"<b>Especialidad:</b> %{{y}}<br><b>Tramo:</b> {tramo}<br><b>Porcentaje:</b> %{{x:.2f}}%<extra></extra>"
            ))

        fig8.update_layout(
            template=TEMPLATE_CORPORATIVO, barmode='stack', xaxis=dict(ticksuffix="%"),
            title="<b>Distribución Estructurada del Tiempo de Liquidación por Especialidad</b><br><sub>Impacto de la demora burocrática en las Top 8 Especialidades del programa</sub>",
            legend_title="Tramo de Demora", margin=dict(t=60, l=40, r=20, b=40),
            yaxis_title='',
            height=400,
            title_x=0.15, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700),
            legend=dict(font=dict(color='dimgrey'))
        
        )
        st.plotly_chart(fig8, width='stretch', key='chart-liquidacion1')
    with c2:
        st.html(
                """
                <div class="text-column-container">
                    <div class="dashboard-card">
                        <h4 class="text-card-title">Grupo A (Alta Eficiencia):</h4>
                        <p class="text-card-body">
                            Cirugías Genitourinarias (80.4%), Oncología Radioterápica (77.7%), 
                            Politraumatismo (77.2%) y Nefrología (73.5%) liquidan mayoritariamente 
                            en menos de 3 meses. Su estandarización y protocolos predecibles agilizan la auditoría.
                        </p>
                    </div>
                    <div class="dashboard-card">
                        <h4 class="text-card-title">Grupo B (Cuello de Botella):</h4>
                        <p class="text-card-body">
                            Pediatría y Cirugía Cardíaca presentan retrasos críticos; menos del 40% 
                            cumple el tiempo estándar, concentrándose la mayoría entre los 3 y 12 meses.
                        </p>
                    </div>
                </div>
                """
            )
    
    st.divider()
    """## **6. Estrategia Predictiva para la Gestión de Riesgos**"""
    col1, col2 = st.columns(2)
    with col1:
        df_temp = df_cardio_selected_surgeries.copy()

        # Calculate the number of cases where dias_liquidacion > 180 for each surgery
        df_temp['long_liquidation'] = (df_temp['dias_liquidacion'] > 180).astype(int)

        # Group by SURGERY and calculate the percentage
        percentage_long_liquidation_by_surgery = df_temp.groupby('SURGERY')['long_liquidation'].mean() * 100

        # Convert the series to a DataFrame for Plotly
        plot_df_long_liquidation = percentage_long_liquidation_by_surgery.reset_index()
        plot_df_long_liquidation.columns = ['SURGERY', 'Percentage']

        # Sort for visualization
        plot_df_long_liquidation = plot_df_long_liquidation.sort_values(by='Percentage', ascending=True)

        fig_liquidation_percentage = px.bar(
            data_frame=plot_df_long_liquidation,
            x='Percentage',
            y='SURGERY',
            orientation='h',
            color_discrete_sequence=[PALETA_MAESTRA['Secundario_Menta']],
            labels={
                'Percentage': 'Porcentaje de Casos (%)',
                'SURGERY': 'Cirugía'
            },
            title="<b>Porcentaje de Casos con Liquidación > 180 Días por Cirugía</b><br><sub>Análisis de Demora Crítica en Cirugías Cardíacas Seleccionadas</sub>"
        )

        fig_liquidation_percentage.update_traces(
            texttemplate='%{x:.1f}%',
            textposition='outside',
            textfont=dict(color='dimgrey', size=10),
            hovertemplate="<b>Cirugía:</b> %{y}<br><b>Porcentaje > 180 días:</b> %{x:.2f}%<extra></extra>"
        )

        fig_liquidation_percentage.update_layout(
            template=TEMPLATE_CORPORATIVO,
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(categoryorder='total ascending',title=''), # To ensure the largest bars are at the top
            margin=dict(t=60, l=40, r=40, b=40), # Adjust margin for longer labels
            title_x=0.1, title_y=0.95,
            font_color="lightslategray",
            font_family='Arial',
            title_font=dict(size=20, weight=900),
            title_font_color="dimgrey",
            font=dict(size=16, weight=700),
        )
        st.plotly_chart(fig_liquidation_percentage, width='stretch', key='chart-liqcirugias')

    with col2:
        df_cardio_selected_surgeries_long_liquidation = df_cardio_selected_surgeries[df_cardio_selected_surgeries['dias_liquidacion'] > 180]

        # Count long liquidation cases per district
        long_liquidation_counts = df_cardio_selected_surgeries_long_liquidation['HOSP_LOCATION'].value_counts()

        # Count total cases per district
        total_cases_counts = df_cardio_selected_surgeries['HOSP_LOCATION'].value_counts()

        # Calculate the proportion as a Series
        proportion_long_liquidation_series = (long_liquidation_counts / total_cases_counts).fillna(0).sort_values(ascending=False)

        # Convert the Series to a DataFrame
        proportion_long_liquidation = proportion_long_liquidation_series.reset_index()
        proportion_long_liquidation.columns = ['HOSP_LOCATION', 'Proportion_Long_Liquidation']

        # Multiply by 100 to convert to a 0-100 percentage scale
        proportion_long_liquidation['Proportion_Long_Liquidation'] = proportion_long_liquidation['Proportion_Long_Liquidation'] * 100

        # Get a mapping of HOSP_LOCATION to HOSP_DISTRICT from df_cardio_selected_surgeries
        district_state_map = df_cardio_selected_surgeries[['HOSP_LOCATION', 'HOSP_DISTRICT']].drop_duplicates(subset=['HOSP_LOCATION'], keep='first').set_index('HOSP_LOCATION')['HOSP_DISTRICT']

        # Add the HOSP_DISTRICT column to the proportion_long_liquidation DataFrame
        proportion_long_liquidation['HOSP_DISTRICT'] = proportion_long_liquidation['HOSP_LOCATION'].map(district_state_map)
        plot_data = proportion_long_liquidation.head(10)

        fig_plotly = px.bar(
            data_frame=plot_data,
            x='Proportion_Long_Liquidation',
            y='HOSP_LOCATION',
            color='HOSP_DISTRICT',
            orientation='h',
            color_discrete_map=PALETA_DISTRITOS, # 'estados' is defined globally with colors for Andhra Pradesh and Telangana
            labels={
                'Proportion_Long_Liquidation': 'Proporción (%)',
                'HOSP_LOCATION': 'Municipio Hospitalario',
                'HOSP_DISTRICT': 'Distrito'
            },
            custom_data=['Proportion_Long_Liquidation', 'HOSP_DISTRICT']
        )

        fig_plotly.update_traces(
            texttemplate='%{x:.1f}%', # Format as percentage
            textposition='outside',
            textfont=dict(color='dimgrey', size=10),
            hovertemplate="<b>Municipio:</b> %{y}<br><b>Proporción de liquidación larga:</b> %{x:.1f}%<br><b>Distrito:</b> %{customdata[1]}<extra></extra>"
        )

        fig_plotly.update_layout(
            template=TEMPLATE_CORPORATIVO,
            title={
                'text': "<b>Proporción de Días de Liquidación mayor a 6 meses por Municipio Hospitalario</b><br><sub>Top 10 Municipio con mayor proporción de casos de larga liquidación de Cirugías Cardíacas</sub>",
                'y': 0.95, 'x': 0.1, 'yanchor': 'top', 'font': dict(size=20, color='dimgray')
            },
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(categoryorder='total ascending'), # To ensure the largest bars are at the top
            margin = dict(t=60, l=40, r=40, b=40),
            legend_title_text='Estado',
            
        )
        st.plotly_chart(fig_plotly, width='stretch', key='chart-liquidacion3')
    with st.container(height=20, border=False):
        pass # El contenedor está vacío para generar espacio
    df_plot_temp_fallecido = df_cardio_selected_surgeries.copy()
    df_plot_temp_fallecido['fallecido_label'] = df_plot_temp_fallecido['fallecido'].map({1: 'Sí', 0: 'No'})

    fig9 = px.box(
        data_frame=df_plot_temp_fallecido,
        x='dias_liquidacion', points=False,
        color='fallecido_label',
        color_discrete_map={
            'Sí': PALETA_MAESTRA['Alerta_Terracota'],
            'No': PALETA_MAESTRA['Neutro_Claro']
        },
        category_orders={'fallecido_label': ['Sí', 'No']},
        labels={'dias_liquidacion': 'Días de liquidación', 'fallecido_label': '¿Fallecido?'},
        title=f"<b>Distribución de días de liquidación según mortalidad</b><br><sub>¿En los casos de muerte del paciente se acelera o demora el proceso de liquidación del pago?</sub>",
        hover_data={'fallecido_label': True, 'dias_liquidacion': True}
    )

    fig9.update_layout(
        template=TEMPLATE_CORPORATIVO,
        xaxis_type='log',
        xaxis=dict(tickformat='.0f'),
        legend_title_text='¿Fallecido?',
        margin=dict(t=60, l=40, r=40, b=40),
        boxmode='group',
        boxgroupgap=0.2,
        boxgap=0.1,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        yaxis_title='',
        title_x=0.1, title_y=0.95,
        font_color="lightslategray",
        font_family='Arial',
        title_font=dict(size=20, weight=900),
        title_font_color="dimgrey",
        font=dict(size=16, weight=700),
        legend=dict(font=dict(color='dimgrey'))

    )

    fig9.update_traces(
        marker_line_color='silver', 
        marker_line_width=0.6, 
        selector=dict(type='box')
    )
    st.plotly_chart(fig9, width='stretch', key='chart-liquidacion2')

with tab_simulador:
    st.title("Simulador de Tiempos de Demora")
    st.markdown("---")
    st.markdown("### Regresión por Cuantiles y Clasificación de Riesgo con LightGBM")
    st.write("Seleccioná las características reales para calcular el rango estimado de días de liquidación y alertas de demora.")

    # 2. CARGA DE MODELOS Y DATA MAESTRA
    @st.cache_resource
    def cargar_recursos():
        m10 = joblib.load("modelo_q10.pkl")
        m50 = joblib.load("modelo_q50.pkl")
        m90 = joblib.load("modelo_q90.pkl")
        m_clf = joblib.load("modelo_clasificador_lgb.pkl")
        u_opt = joblib.load("umbral_optimo_lgb.pkl")
        df_rel = joblib.load("df_relaciones_desplegables.pkl")
        return m10, m50, m90, m_clf, u_opt, df_rel

    try:
        model_10, model_50, model_90, model_clf, umbral_optimo, df_rel = cargar_recursos()
    except Exception as e:
        st.error(f"⚠️ No se pudieron cargar los archivos .pkl: {e}")
        st.stop()

    columnas_reales = df_rel.columns.tolist()

    COL_CIRUGIA = 'SURGERY'
    COL_MUNICIPIO = 'HOSP_LOCATION'

    if COL_MUNICIPIO not in columnas_reales:
        st.error(f"❌ La columna '{COL_MUNICIPIO}' no existe en tu archivo pkl.")
        st.stop()

    # ==============================================================================
    # 3. INTERFAZ DE ENTRADA DINÁMICA
    # ==============================================================================
    lista_cirugias = sorted(df_rel[COL_CIRUGIA].dropna().unique().tolist())
    cirugia = st.selectbox("1. Seleccioná la Cirugía Médica", lista_cirugias)

    # Filtrar municipios disponibles según la cirugía (o dejar todos los del pkl si no están condicionados)
    lista_municipios = sorted(df_rel[df_rel[COL_CIRUGIA] == cirugia][COL_MUNICIPIO].dropna().unique().tolist())
    if not lista_municipios: # Fallback por si esa cirugía no tiene registros en el pkl
        lista_municipios = sorted(df_rel[COL_MUNICIPIO].dropna().unique().tolist())
        
    municipio = st.selectbox("2. Seleccioná el Municipio del Hospital", lista_municipios)

    st.markdown("---")
    st.markdown("#### Contexto del Paciente")
    
    fallecido = st.radio("¿El paciente falleció?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    st.markdown(" ")
    boton_calcular = st.button("Predecir Tiempos de Demora y Evaluar Riesgo", type="primary")

    # ==============================================================================
    # 4. EXTRACCIÓN Y PREDICCIÓN COMBINADA (REGRESIÓN + CLASIFICACIÓN)
    # ==============================================================================
    if boton_calcular:
        # Extraer la tasa de mortalidad correspondiente al municipio seleccionado
        df_muni_info = df_rel[df_rel[COL_MUNICIPIO] == municipio]
        if df_muni_info.empty:
            st.error("Error al recuperar los datos internos para el municipio seleccionado.")
            st.stop()
            
        registro = df_muni_info.iloc[0]
        
        # Construcción del vector con las 4 variables predictoras exactas
        input_data = pd.DataFrame([{
            'SURGERY': cirugia,
            'HOSP_LOCATION': municipio,
            'fallecido': int(fallecido),
            'TASA_MORTALIDAD_INTERNA_HOSP': float(registro['TASA_MORTALIDAD_INTERNA_HOSP'])
        }])
        
        # Forzar el formato categórico para LightGBM nativo
        for col in ['SURGERY', 'HOSP_LOCATION']:
            input_data[col] = input_data[col].astype('category')
            
        # A. Ejecución de Modelos de Regresión Cuantílica
        pred_piso = max(0, round(model_10.predict(input_data)[0], 1))
        pred_mediana = max(0, round(model_50.predict(input_data)[0], 1))
        pred_techo = max(0, round(model_90.predict(input_data)[0], 1))
        
        # B. Ejecución del Clasificador con Umbral Óptimo de Youden
        prob_demora_critica = model_clf.predict_proba(input_data)[0][1]
        es_riesgo_critico = prob_demora_critica >= umbral_optimo

        # ==============================================================================
        # 5. RENDERIZADO DE RESULTADOS
        # ==============================================================================
        
        # --- SEMÁFORO DE RIESGO DE NEGOCIO ---
        st.markdown("### ⚠️ Evaluación de Alerta Burocrática (Umbral Crítico 180 Días)")
        if es_riesgo_critico:
            st.markdown(
                f"""
                <div style="background-color: #dd9993; padding: 18px; border-radius: 8px; border-left: 6px solid #a83232; margin-bottom: 20px;">
                    <b style="color: #5a5a5a; font-size: 18px;">ALERTA: Alto Riesgo de Demora Crítica</b><br>
                    <span style="color: #5a5a5a;">El modelo estima una probabilidad de <b>{prob_demora_critica:.1%}</b> (superior al umbral óptimo de {umbral_optimo:.1%}). 
                    Este trámite presenta patrones severos de retraso institucional y requiere asignación de vía rápida.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #cae6d4; padding: 18px; border-radius: 8px; border-left: 6px solid #2d6a4f; margin-bottom: 20px;">
                    <b style="color: #5a5a5a; font-size: 18px;">Flujo Operativo Seguro (Bajo Control)</b><br>
                    <span style="color: #5a5a5a;">Probabilidad de retraso crítico calculada en <b>{prob_demora_critica:.1%}</b> (por debajo del riesgo de demora). 
                    Se estima un procesamiento administrativo estándar.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Renderizado de Tarjetas Métricas de Regresión
        st.markdown("### 📈 Rango de Demora Numérica Estimada")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Piso Optimista (Q10)", value=f"{pred_piso} días")
        with c2:
            st.metric(label="Mediana Esperada (Q50)", value=f"{pred_mediana} días")
        with c3:
            st.metric(label="Techo Crítico (Q90)", value=f"{pred_techo} días")
            
        brecha = round(pred_techo - pred_piso, 1)
        st.info(f"💡 **Ventana de Incertidumbre:** El rango de duda es de **{brecha} días** para este caso específico.")