"""Parte 1 - Captação e Avaliação da Qualidade dos Dados."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from db import get_available_years, get_pib_ride

st.set_page_config(page_title="Parte 1 - Qualidade", page_icon="🧪", layout="wide")
st.title("🧪 Parte 1 - Qualidade dos Dados")
st.caption("Estatísticas descritivas, *missing values*, *outliers* e anomalias")

# --------------------------------------------------------------
# Ano
# --------------------------------------------------------------
anos = get_available_years()
ano = st.session_state.get("ano", max(anos))
ano = st.selectbox("Ano de referência", anos, index=anos.index(ano))
st.session_state["ano"] = int(ano)

df = get_pib_ride(ano)

st.info(
    f"**Ano:** {ano}  |  **Municípios RIDE-DF:** {df.shape[0]}  |  "
    f"**Variáveis:** {df.shape[1]}"
)

NUM_COLS = [
    "vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao",
    "vl_bruto_total", "vl_subsidios", "vl_pib", "vl_pib_per_capta",
]

# --------------------------------------------------------------
# 1. Estatísticas descritivas
# --------------------------------------------------------------
st.header("1. Estatísticas Descritivas")
desc = df[NUM_COLS].describe().T
desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(2)
desc["skew"] = df[NUM_COLS].skew().round(3)
desc["kurtosis"] = df[NUM_COLS].kurtosis().round(3)
st.dataframe(desc.style.format("{:,.2f}"), use_container_width=True)

st.markdown(
    """
> *CV (%) alto e assimetria (skew) positiva indicam distribuições fortemente
> concentradas em municípios maiores - padrão típico de variáveis
> monetárias regionais.*
"""
)

# --------------------------------------------------------------
# 2. Missing values
# --------------------------------------------------------------
st.header("2. Valores Ausentes (Missing)")
missing = pd.DataFrame(
    {
        "n_missing": df.isna().sum(),
        "%_missing": (df.isna().sum() / len(df) * 100).round(2),
    }
)
missing = missing[missing["n_missing"] > 0]
if missing.empty:
    st.success("Nenhum valor ausente encontrado nas variáveis analisadas. ✅")
else:
    st.warning("Foram encontrados valores ausentes:")
    st.dataframe(missing, use_container_width=True)

# --------------------------------------------------------------
# 3. Duplicatas
# --------------------------------------------------------------
st.header("3. Registros Duplicados")
dup = df.duplicated(subset=["ano", "codigo_municipio_dv"]).sum()
st.metric("Duplicatas por (ano, município)", int(dup))
if dup == 0:
    st.success("Não há duplicidade de chave (ano × município). ✅")

# --------------------------------------------------------------
# 4. Outliers (IQR)
# --------------------------------------------------------------
st.header("4. Identificação de Outliers (regra IQR 1.5×)")

def iqr_outliers(s: pd.Series) -> tuple[int, float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((s < low) | (s > high)).sum())
    return n, low, high

out_tbl = pd.DataFrame(
    {
        c: iqr_outliers(df[c].dropna()) for c in NUM_COLS
    },
    index=["n_outliers", "lim_inf", "lim_sup"],
).T.astype({"n_outliers": int})
st.dataframe(out_tbl.style.format({"lim_inf": "{:,.2f}", "lim_sup": "{:,.2f}"}),
             use_container_width=True)

variavel = st.selectbox("Visualizar boxplot de:", NUM_COLS, index=0)
fig = px.box(
    df, y=variavel, points="all", color="uf",
    hover_data=["nome_municipio", "uf"],
    title=f"Boxplot - {variavel} (RIDE-DF, {ano})",
)
st.plotly_chart(fig, use_container_width=True)

# Top-5 outliers da variável escolhida
q1, q3 = df[variavel].quantile([0.25, 0.75])
iqr = q3 - q1
mask = (df[variavel] < q1 - 1.5 * iqr) | (df[variavel] > q3 + 1.5 * iqr)
if mask.any():
    st.markdown(f"**Municípios outliers em `{variavel}`:**")
    st.dataframe(
        df.loc[mask, ["nome_municipio", "uf", variavel]]
        .sort_values(variavel, ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.success(f"Nenhum outlier em `{variavel}` pela regra IQR.")

# --------------------------------------------------------------
# 5. Histogramas
# --------------------------------------------------------------
st.header("5. Distribuição das Variáveis")

# Regra de Sturges: k = ceil(log2(n) + 1) — mais adequado para n=34
_n_bins = int(np.ceil(np.log2(len(df)) + 1))

cols = st.columns(2)
for i, c in enumerate(NUM_COLS):
    fig = px.histogram(
        df, x=c,
        nbins=_n_bins,
        title=c,
        marginal="box",
        labels={c: c, "count": "Frequência"},
    )
    fig.update_traces(
        marker_line_width=1,
        marker_line_color="rgba(0,0,0,0.45)",
    )
    fig.update_layout(bargap=0.05)
    cols[i % 2].plotly_chart(fig, use_container_width=True)

st.caption(
    f"Bins calculados pela regra de Sturges: k = ⌈log₂(n) + 1⌉ = {_n_bins} "
    f"(n = {len(df)} municípios). Valores em R\$ mil (IBGE)."
)

# --------------------------------------------------------------
# 6. Anomalias lógicas / validações de integridade
# --------------------------------------------------------------
st.header("6. Anomalias e Validações de Integridade")

check = pd.DataFrame(
    {
        "Regra": [
            "Algum valor negativo em colunas monetárias",
            "VAB total ≠ soma dos 4 setores (tolerância 0,5 %)",
            "PIB < VAB total (não deveria ocorrer: PIB = VAB + Impostos)",
            "PIB per capita ≤ 0",
        ],
        "Nº de registros": [
            int((df[NUM_COLS].fillna(0) < 0).any(axis=1).sum()),
            int(
                (
                    (
                        df[["vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao"]]
                        .sum(axis=1)
                        - df["vl_bruto_total"]
                    ).abs()
                    / df["vl_bruto_total"].replace(0, np.nan)
                    > 0.005
                ).sum()
            ),
            int((df["vl_pib"] < df["vl_bruto_total"]).sum()),
            int((df["vl_pib_per_capta"] <= 0).sum()),
        ],
    }
)
st.dataframe(check, use_container_width=True, hide_index=True)

st.markdown(
    """
### 📝 Conclusões da Parte 1

- A base está **consistente** para o recorte RIDE-DF: sem duplicatas,
  sem valores negativos em colunas monetárias e PIB ≥ VAB conforme identidade contábil.
- Todas as variáveis monetárias apresentam forte **assimetria à direita**,
  influenciada principalmente pelo **Distrito Federal** (Brasília),
  que concentra a maior parte do PIB da região e aparece como
  outlier em quase todas as variáveis.
- Recomenda-se considerar **transformação logarítmica** ou tratamento
  robusto de outliers na etapa de modelagem (Parte 2).
"""
)
