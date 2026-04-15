"""Parte 2 - Correlações e Associações."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from db import get_available_years, get_pib_ride, enriquecer_qualitativas

st.set_page_config(page_title="Parte 2 - Correlações", page_icon="🔗", layout="wide")
st.title("🔗 Parte 2 - Correlações e Associações")

anos = get_available_years()
ano = st.session_state.get("ano", max(anos))
ano = st.selectbox("Ano de referência", anos, index=anos.index(ano))
st.session_state["ano"] = int(ano)

df = enriquecer_qualitativas(get_pib_ride(ano))

NUM_COLS = [
    "vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao",
    "vl_bruto_total", "vl_subsidios", "vl_pib", "vl_pib_per_capta",
]
QUAL_COLS = ["uf", "setor_predominante", "porte"]

# =========================================================================
# 1. QUANTITATIVA × QUANTITATIVA (correlação)
# =========================================================================
st.header("1. Correlação entre variáveis quantitativas")

col_m, col_l = st.columns([1, 2])
metodo = col_m.radio("Método", ["pearson", "spearman"], horizontal=True)
usar_log = col_l.checkbox(
    "Aplicar transformação log1p antes de correlacionar (recomendado)",
    value=True,
    help="Variáveis monetárias brutas são fortemente assimétricas e dominadas por "
         "Brasília. O log1p reduz o efeito de escala e revela relações estruturais "
         "mais significativas.",
)

df_corr = np.log1p(df[NUM_COLS]) if usar_log else df[NUM_COLS]
corr = df_corr.corr(method=metodo)

fig = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, aspect="auto",
    title=f"Matriz de correlação ({metodo}{'  |  log1p' if usar_log else ''}) - RIDE-DF {ano}",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
> **Leitura:** valores próximos de 1 (vermelho escuro) ⇒ forte relação linear/monotônica
> positiva; próximos de -1 (azul escuro) ⇒ forte relação negativa.
"""
)

with st.expander("ℹ️ Por que as correlações nos valores brutos chegam a 0,99–1,00?"):
    st.markdown(
        """
        Quando se aplica Pearson (ou Spearman) nos **valores absolutos em R\$ mil**,
        duas fontes de distorção se combinam:

        1. **Efeito de escala / tamanho:** municípios maiores tendem a ter valores altos
           em *todos* os setores. Brasília, com economia ~200× maior que a média dos
           demais, cria um ponto extremo que "puxa" os coeficientes para 1.
        2. **Relações definitórias:** `vl_bruto_total` = agropecuária + indústria +
           serviços + administração (por definição); `vl_pib` ≈ `vl_bruto_total` +
           impostos − subsídios. Correlações próximas de 1 entre esses pares são
           esperadas e não refletem uma relação causal ou estrutural.

        A transformação **log1p** comprime a escala, reduz a influência de Brasília e
        revela as correlações estruturais entre os setores — que é o que interessa
        analiticamente.
        """
    )

with st.expander("Scatter matrix (pares de variáveis)"):
    fig2 = px.scatter_matrix(
        df_corr.assign(uf=df["uf"]), dimensions=NUM_COLS, color="uf",
        height=900, title="Scatter matrix das variáveis quantitativas",
    )
    fig2.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig2, use_container_width=True)

# Correlações com o alvo
st.subheader("Correlação das variáveis com o alvo `vl_agropecuaria`")
alvo_corr = (
    corr["vl_agropecuaria"]
    .drop("vl_agropecuaria")
    .sort_values(key=abs, ascending=False)
    .rename("corr")
    .to_frame()
)
st.dataframe(alvo_corr.style.format("{:.3f}"), use_container_width=True)

# =========================================================================
# 2. QUALITATIVA × QUALITATIVA (associação - Chi² e V de Cramér)
# =========================================================================
st.header("2. Associação entre variáveis qualitativas")

def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    tab = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(tab)
    n = tab.values.sum()
    r, k = tab.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else np.nan
    return v, p

res = []
for i, a in enumerate(QUAL_COLS):
    for b in QUAL_COLS[i + 1:]:
        v, p = cramers_v(df[a], df[b])
        res.append({"Variável A": a, "Variável B": b,
                    "V de Cramér": round(v, 3), "p-valor (Chi²)": round(p, 4)})
st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)

st.markdown(
    """
> **V de Cramér** varia de 0 a 1: quanto mais próximo de 1, mais forte
> a associação. **p < 0,05** indica associação estatisticamente significativa.
"""
)

# Tabela de contingência escolhida
c1, c2 = st.columns(2)
a = c1.selectbox("Variável A", QUAL_COLS, index=0)
b = c2.selectbox("Variável B", [q for q in QUAL_COLS if q != a], index=0)
tab = pd.crosstab(df[a], df[b], margins=True, margins_name="Total")
st.dataframe(tab, use_container_width=True)
st.plotly_chart(
    px.imshow(
        pd.crosstab(df[a], df[b]),
        text_auto=True, color_continuous_scale="Blues",
        title=f"Tabela de contingência: {a} × {b}",
    ),
    use_container_width=True,
)

# =========================================================================
# 3. QUANTITATIVA × QUALITATIVA (ANOVA + boxplot)
# =========================================================================
st.header("3. Quantitativa × Qualitativa (ANOVA)")
qt = st.selectbox("Variável quantitativa", NUM_COLS, index=0)
ql = st.selectbox("Variável qualitativa", QUAL_COLS, index=1)

# Excluir grupos com n = 1 (ex.: DF tem apenas Brasília) — ANOVA e boxplot
grupos_contagem = df.groupby(ql)[qt].count()
grupos_validos = grupos_contagem[grupos_contagem > 1].index
excluidos = grupos_contagem[grupos_contagem <= 1].index.tolist()

df_anova = df[df[ql].isin(grupos_validos)]

if excluidos:
    st.caption(
        f"⚠️ Grupo(s) excluídos do ANOVA e do boxplot por terem apenas 1 observação "
        f"(sem variância): **{', '.join(excluidos)}**. "
        "Um grupo com n = 1 viola a premissa do ANOVA e não permite interpretar "
        "mediana, quartis ou dispersão em boxplot."
    )

grupos = [g[qt].dropna().values for _, g in df_anova.groupby(ql)]
f_stat, p_val = stats.f_oneway(*grupos)

# Eta² calculado sobre o subconjunto válido
grand_mean = df_anova[qt].mean()
ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in grupos)
ss_total = ((df_anova[qt] - grand_mean) ** 2).sum()
eta2 = ss_between / ss_total if ss_total else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Estatística F", f"{f_stat:.3f}")
c2.metric("p-valor", f"{p_val:.4f}")
c3.metric("η² (eta²)", f"{eta2:.3f}")

fig3 = px.box(df_anova, x=ql, y=qt, color=ql, points="all",
              title=f"{qt} por {ql} (ANOVA p={p_val:.4f})",
              hover_data=["nome_municipio", "uf"])
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    """
> Se **p < 0,05**, há evidência de que as médias da variável quantitativa
> diferem entre os grupos. **η²** indica a parcela da variância explicada
> pela variável qualitativa.
"""
)

# Tabela resumo por grupo (apenas grupos válidos)
st.subheader(f"Resumo de `{qt}` por `{ql}`")
resumo = df_anova.groupby(ql)[qt].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
st.dataframe(resumo, use_container_width=True)
