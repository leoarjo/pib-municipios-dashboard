"""Parte 2 - Análises territoriais (mapas)."""
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from db import get_available_years, get_pib_ride, enriquecer_qualitativas
from utils import fmt_brl_full, fmt_brl_reais

st.set_page_config(page_title="Parte 2 - Mapas", page_icon="🗺️", layout="wide")
st.title("🗺️ Parte 2 - Análises Territoriais")
st.caption("Mapas da RIDE-DF com os indicadores do PIB")

anos = get_available_years()
ano = st.session_state.get("ano", max(anos))
ano = st.selectbox("Ano de referência", anos, index=anos.index(ano))
st.session_state["ano"] = int(ano)

df = enriquecer_qualitativas(get_pib_ride(ano)).dropna(subset=["latitude", "longitude"])

variavel = st.selectbox(
    "Variável para colorir/dimensionar os marcadores",
    ["vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao",
     "vl_bruto_total", "vl_pib", "vl_pib_per_capta"],
    index=0,
)

# ---------------- Mapa scatter (Plotly) ------------------
st.subheader(f"Mapa interativo - {variavel} ({ano})")
fig = px.scatter_mapbox(
    df,
    lat="latitude", lon="longitude",
    size=np.clip(df[variavel], 1, None),  # evita valores negativos/zero
    color=variavel,
    color_continuous_scale="Viridis",
    hover_name="nome_municipio",
    hover_data={
        "uf": True,
        "vl_agropecuaria": ":,.0f",
        "vl_pib": ":,.0f",
        "setor_predominante": True,
        "porte": True,
        "latitude": False,
        "longitude": False,
    },
    zoom=6, height=600,
)
fig.update_layout(mapbox_style="open-street-map",
                  margin={"r": 0, "t": 30, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

# ---------------- Mapa Folium com popups ricos ------------------
st.subheader("Mapa detalhado (Folium) - clique nos marcadores")
centro_lat = df["latitude"].mean()
centro_lon = df["longitude"].mean()

m = folium.Map(location=[centro_lat, centro_lon], zoom_start=7, tiles="OpenStreetMap")
cluster = MarkerCluster().add_to(m)

# cor por setor predominante
cores = {
    "Agropecuária": "green",
    "Indústria": "orange",
    "Serviços": "blue",
    "Administração Pública": "red",
}
for _, r in df.iterrows():
    html = f"""
        <b>{r['nome_municipio']}</b> ({r['uf']})<br>
        Setor predominante: <b>{r['setor_predominante']}</b><br>
        Porte: <b>{r['porte']}</b><br><br>
        VAB Agropecuária: {fmt_brl_full(r['vl_agropecuaria'])}<br>
        VAB Indústria: {fmt_brl_full(r['vl_industria'])}<br>
        VAB Serviços: {fmt_brl_full(r['vl_servicos'])}<br>
        PIB: {fmt_brl_full(r['vl_pib'])}<br>
        PIB per capita: {fmt_brl_reais(r['vl_pib_per_capta'])}
    """
    folium.CircleMarker(
        location=[r["latitude"], r["longitude"]],
        radius=6 + np.log1p(max(r[variavel], 0)) / 2,
        popup=folium.Popup(html, max_width=300),
        tooltip=r["nome_municipio"],
        color=cores.get(r["setor_predominante"], "gray"),
        fill=True, fill_opacity=0.75,
    ).add_to(cluster)

st_folium(m, width=None, height=550, returned_objects=[])

# ---------------- Evolução territorial ------------------
st.subheader("Evolução territorial do VAB Agropecuária")
df_all = get_pib_ride().dropna(subset=["latitude", "longitude"])
fig2 = px.scatter_mapbox(
    df_all.sort_values("ano"),
    lat="latitude", lon="longitude",
    size=np.clip(df_all["vl_agropecuaria"], 1, None),
    color="vl_agropecuaria",
    animation_frame="ano",
    hover_name="nome_municipio",
    color_continuous_scale="YlGn",
    zoom=6, height=600,
    title="VAB Agropecuária por município ao longo do tempo",
)
fig2.update_layout(mapbox_style="open-street-map",
                   margin={"r": 0, "t": 30, "l": 0, "b": 0})
st.plotly_chart(fig2, use_container_width=True)
