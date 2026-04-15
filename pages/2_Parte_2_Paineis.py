"""Parte 2 - Painéis de informações sobre o PIB da RIDE-DF."""
import pandas as pd
import plotly.express as px
import streamlit as st

from db import get_available_years, get_pib_ride, enriquecer_qualitativas
from utils import fmt_brl_mil, fmt_brl_reais

st.set_page_config(page_title="Parte 2 - Painéis", page_icon="📈", layout="wide")
st.title("📈 Parte 2 - Painéis de Informações")
st.caption("Retrato fiel do PIB dos municípios da RIDE-DF no período analisado")

anos = get_available_years()
ano = st.session_state.get("ano", max(anos))
ano = st.selectbox("Ano de referência (retrato)", anos, index=anos.index(ano))
st.session_state["ano"] = int(ano)

df = enriquecer_qualitativas(get_pib_ride(ano))

# ---------------- KPIs ------------------
total_pib   = df["vl_pib"].sum()
total_agro  = df["vl_agropecuaria"].sum()
total_ind   = df["vl_industria"].sum()
total_serv  = df["vl_servicos"].sum()
total_adm   = df["vl_administracao"].sum()
maior_agro  = df.loc[df["vl_agropecuaria"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("PIB total (R$)", fmt_brl_mil(total_pib))
c2.metric("VAB Agropecuária (R$)", fmt_brl_mil(total_agro))
c3.metric("PIB per capita médio (R$)", fmt_brl_reais(df["vl_pib_per_capta"].mean()))
c4.metric(
    "Líder em Agropecuária",
    f"{maior_agro['nome_municipio']}/{maior_agro['uf']}",
    fmt_brl_mil(maior_agro["vl_agropecuaria"]),
)

st.caption("ℹ️ Valores das variáveis VAB/PIB originalmente em R$ mil (IBGE); "
           "exibidos aqui convertidos para R$ bilhões (bi) ou milhões (mi).")

st.divider()

# helper: converte coluna de R$ mil para R$ bilhões (uso em gráficos)
def _bi(series: pd.Series) -> pd.Series:
    return series * 1_000 / 1_000_000_000   # R$ mil → R$ bi


# ---------------- Composição setorial ------------------
st.subheader(f"1. Composição setorial do VAB da RIDE-DF ({ano})")
comp = pd.DataFrame({
    "Setor": ["Agropecuária", "Indústria", "Serviços", "Administração Pública"],
    "Valor (R$ bi)": [
        round(_bi(pd.Series([total_agro])).iloc[0], 2),
        round(_bi(pd.Series([total_ind ])).iloc[0], 2),
        round(_bi(pd.Series([total_serv])).iloc[0], 2),
        round(_bi(pd.Series([total_adm ])).iloc[0], 2),
    ],
})
comp["%"] = (comp["Valor (R$ bi)"] / comp["Valor (R$ bi)"].sum() * 100).round(2)

colA, colB = st.columns([1, 1])
colA.plotly_chart(
    px.pie(comp, values="Valor (R$ bi)", names="Setor", hole=0.45,
           title="Participação setorial no VAB (RIDE-DF)"),
    use_container_width=True,
)
colB.dataframe(
    comp.style.format({"Valor (R$ bi)": "{:.2f}", "%": "{:.2f}"}),
    use_container_width=True, hide_index=True,
)

# ---------------- Top municípios ------------------
st.subheader(f"2. Top 10 municípios por variável ({ano})")
var = st.selectbox(
    "Selecione a variável",
    ["vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao",
     "vl_bruto_total", "vl_pib", "vl_pib_per_capta"],
)
top10 = df.nlargest(10, var)[["nome_municipio", "uf", var]].copy()

# Formatar valor exibido no hover/eixo
if var != "vl_pib_per_capta":
    top10["valor_fmt"] = top10[var].apply(fmt_brl_mil)
    top10["valor_bi"] = _bi(top10[var])
    fig = px.bar(
        top10, x="valor_bi", y="nome_municipio", color="uf", orientation="h",
        title=f"Top 10 por {var} ({ano}) — R$ bilhões",
        labels={"valor_bi": "R$ bi", "nome_municipio": "Município"},
        hover_data={"valor_fmt": True, "valor_bi": False},
    )
else:
    top10["valor_fmt"] = top10[var].apply(fmt_brl_reais)
    fig = px.bar(
        top10, x=var, y="nome_municipio", color="uf", orientation="h",
        title=f"Top 10 por {var} ({ano}) — R$",
        labels={var: "R$", "nome_municipio": "Município"},
    )
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

# ---------------- Distribuição por UF ------------------
st.subheader("3. Distribuição por UF")
por_uf = df.groupby("uf")[[
    "vl_agropecuaria", "vl_industria", "vl_servicos",
    "vl_administracao", "vl_pib",
]].sum().reset_index()

# Converter para R$ bi
for col in ["vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao", "vl_pib"]:
    por_uf[col] = _bi(por_uf[col])

por_uf_long = por_uf.melt(id_vars="uf", var_name="variavel", value_name="valor (R$ bi)")
fig2 = px.bar(
    por_uf_long, x="uf", y="valor (R$ bi)", color="variavel", barmode="group",
    title=f"Soma das variáveis por UF — RIDE-DF {ano} (R$ bilhões)",
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- Setor predominante ------------------
st.subheader("4. Setor econômico predominante por município")
sp = df["setor_predominante"].value_counts().reset_index()
sp.columns = ["Setor predominante", "Nº de municípios"]
cA, cB = st.columns([1, 1])
cA.plotly_chart(
    px.bar(sp, x="Setor predominante", y="Nº de municípios", color="Setor predominante",
           title="Quantidade de municípios por setor predominante"),
    use_container_width=True,
)
cB.dataframe(
    df[["nome_municipio", "uf", "setor_predominante", "porte"]]
    .sort_values(["setor_predominante", "nome_municipio"])
    .reset_index(drop=True),
    use_container_width=True, height=350,
)

st.divider()

# ---------------- Série histórica ------------------
st.subheader("5. Série histórica (2010-2021) - RIDE-DF")
df_hist = get_pib_ride()  # todos os anos
agg = df_hist.groupby("ano")[[
    "vl_agropecuaria", "vl_industria", "vl_servicos",
    "vl_administracao", "vl_pib"
]].sum().reset_index()

# Converter para R$ bi
for col in ["vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao", "vl_pib"]:
    agg[col] = _bi(agg[col])

fig3 = px.line(
    agg.melt(id_vars="ano", var_name="variável", value_name="R$ bilhões"),
    x="ano", y="R$ bilhões", color="variável", markers=True,
    title="Evolução anual dos agregados da RIDE-DF (R$ bilhões correntes)",
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Evolução do VAB Agropecuária por UF")
agro_uf = df_hist.groupby(["ano", "uf"])["vl_agropecuaria"].sum().reset_index()
agro_uf["vl_agropecuaria_bi"] = _bi(agro_uf["vl_agropecuaria"])
fig4 = px.line(
    agro_uf, x="ano", y="vl_agropecuaria_bi", color="uf", markers=True,
    title="VAB Agropecuária - RIDE-DF por UF (R$ bilhões)",
    labels={"vl_agropecuaria_bi": "R$ bilhões"},
)
st.plotly_chart(fig4, use_container_width=True)
