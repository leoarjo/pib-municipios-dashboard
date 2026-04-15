"""Parte 2 - Modelos de Regressão Linear (simples e múltipla)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from db import enriquecer_qualitativas, get_available_years, get_pib_ride

st.set_page_config(page_title="Parte 2 - Regressão", page_icon="📐", layout="wide")
st.title("📐 Parte 2 - Modelos de Regressão Linear")
st.caption("Alvo: `vl_agropecuaria` (Valor Adicionado Bruto da Agropecuária)")

anos = get_available_years()
ano = st.session_state.get("ano", max(anos))
ano = st.selectbox("Ano de referência", anos, index=anos.index(ano))
st.session_state["ano"] = int(ano)

df = enriquecer_qualitativas(get_pib_ride(ano)).dropna(
    subset=["vl_agropecuaria", "vl_industria", "vl_servicos",
            "vl_administracao", "vl_pib_per_capta"]
)

usar_log = st.checkbox(
    "Aplicar transformação log1p nas variáveis monetárias "
    "(recomendado - distribuições muito assimétricas)", value=True,
)

def prep(s: pd.Series) -> pd.Series:
    return np.log1p(s) if usar_log else s

TARGET = "vl_agropecuaria"

st.info(f"Ano: **{ano}** | Observações: **{len(df)}** | "
        f"Transformação: **{'log1p' if usar_log else 'nenhuma'}**")

# =========================================================================
# 1. REGRESSÃO LINEAR SIMPLES
# =========================================================================
st.header("1. Regressão Linear Simples")

preditores_simples = ["vl_industria", "vl_servicos", "vl_administracao",
                      "vl_bruto_total", "vl_pib", "vl_pib_per_capta"]
x_var = st.selectbox("Variável preditora (X)", preditores_simples, index=0)

X = prep(df[[x_var]])
y = prep(df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
modelo_s = LinearRegression().fit(X_train, y_train)
pred_train = modelo_s.predict(X_train)
pred_test = modelo_s.predict(X_test)

c1, c2, c3, c4 = st.columns(4)
c1.metric("β₀ (intercepto)", f"{modelo_s.intercept_:.4f}")
c2.metric(f"β₁ ({x_var})", f"{modelo_s.coef_[0]:.4f}")
c3.metric("R² treino", f"{r2_score(y_train, pred_train):.3f}")
c4.metric("R² teste", f"{r2_score(y_test, pred_test):.3f}")

c1, c2, c3 = st.columns(3)
c1.metric("RMSE teste", f"{np.sqrt(mean_squared_error(y_test, pred_test)):.3f}")
c2.metric("MAE teste", f"{mean_absolute_error(y_test, pred_test):.3f}")
r, p = stats.pearsonr(X[x_var], y)
c3.metric("Correlação de Pearson (X,y)", f"{r:.3f}  (p={p:.4f})")

# Gráfico regressão
fig = px.scatter(
    x=X[x_var], y=y,
    labels={"x": f"{'log1p' if usar_log else ''}({x_var})",
            "y": f"{'log1p' if usar_log else ''}({TARGET})"},
    trendline="ols", title="Reta de regressão (OLS)",
)
st.plotly_chart(fig, use_container_width=True)

# Diagnóstico de resíduos
residuos = y_test - pred_test
colA, colB = st.columns(2)
colA.plotly_chart(
    px.scatter(x=pred_test, y=residuos,
               labels={"x": "Predito", "y": "Resíduo"},
               title="Resíduos vs Predito (teste)").add_hline(y=0, line_dash="dash"),
    use_container_width=True,
)
colB.plotly_chart(
    px.histogram(residuos, nbins=20, title="Distribuição dos resíduos"),
    use_container_width=True,
)

# Shapiro-Wilk (normalidade dos resíduos)
sw_stat, sw_p = stats.shapiro(residuos)
st.caption(
    f"**Shapiro-Wilk (normalidade dos resíduos):** W={sw_stat:.3f}, p={sw_p:.4f} — "
    + ("resíduos compatíveis com normalidade ✅" if sw_p > 0.05
       else "indício de não-normalidade ⚠️")
)

# Summary completo via statsmodels
with st.expander("📑 Summary estatístico (statsmodels OLS)"):
    Xsm = sm.add_constant(X_train)
    ols = sm.OLS(y_train, Xsm).fit()
    st.text(ols.summary().as_text())

st.divider()

# =========================================================================
# 2. REGRESSÃO LINEAR MÚLTIPLA
# =========================================================================
st.header("2. Regressão Linear Múltipla")

default_feats = ["vl_industria", "vl_servicos", "vl_administracao", "vl_pib_per_capta"]
feats_num = st.multiselect(
    "Variáveis numéricas preditoras",
    ["vl_industria", "vl_servicos", "vl_administracao",
     "vl_bruto_total", "vl_subsidios", "vl_pib", "vl_pib_per_capta"],
    default=default_feats,
)
feats_cat = st.multiselect(
    "Variáveis qualitativas (one-hot)",
    ["uf", "setor_predominante", "porte"],
    default=["uf"],
)

if not feats_num and not feats_cat:
    st.warning("Selecione pelo menos uma variável preditora.")
    st.stop()

Xm_num = prep(df[feats_num]) if feats_num else pd.DataFrame(index=df.index)
Xm_cat = (
    pd.get_dummies(df[feats_cat], drop_first=True, dtype=float)
    if feats_cat else pd.DataFrame(index=df.index)
)
Xm = pd.concat([Xm_num, Xm_cat], axis=1)
ym = prep(df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    Xm, ym, test_size=0.3, random_state=42
)
modelo_m = LinearRegression().fit(X_train, y_train)
pred_train = modelo_m.predict(X_train)
pred_test = modelo_m.predict(X_test)

c1, c2, c3, c4 = st.columns(4)
c1.metric("R² treino", f"{r2_score(y_train, pred_train):.3f}")
c2.metric("R² teste", f"{r2_score(y_test, pred_test):.3f}")
# R² ajustado (treino)
n, p = X_train.shape
r2_adj = 1 - (1 - r2_score(y_train, pred_train)) * (n - 1) / (n - p - 1)
c3.metric("R² ajustado (treino)", f"{r2_adj:.3f}")
c4.metric("RMSE teste", f"{np.sqrt(mean_squared_error(y_test, pred_test)):.3f}")

# Coeficientes
coefs = pd.DataFrame({
    "variavel": ["(intercepto)"] + list(Xm.columns),
    "coeficiente": [modelo_m.intercept_] + list(modelo_m.coef_),
})
st.subheader("Coeficientes estimados")
st.dataframe(coefs.style.format({"coeficiente": "{:.4f}"}),
             use_container_width=True, hide_index=True)

# VIF (multicolinearidade) apenas para numéricos
if len(feats_num) >= 2:
    st.subheader("VIF - Fator de Inflação de Variância (multicolinearidade)")
    Xn = Xm_num.dropna()
    vif = pd.DataFrame({
        "variavel": Xn.columns,
        "VIF": [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])],
    })
    st.dataframe(vif.style.format({"VIF": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Regra prática: VIF > 10 indica multicolinearidade severa.")

# Predito vs Observado
st.subheader("Predito × Observado (conjunto de teste)")
plot_df = pd.DataFrame({"observado": y_test.values, "predito": pred_test})
lim_min = min(plot_df.min()) - 0.5
lim_max = max(plot_df.max()) + 0.5
fig_po = px.scatter(plot_df, x="observado", y="predito",
                    title="Predito vs Observado")
fig_po.add_trace(go.Scatter(x=[lim_min, lim_max], y=[lim_min, lim_max],
                            mode="lines", name="y=x",
                            line=dict(dash="dash", color="red")))
st.plotly_chart(fig_po, use_container_width=True)

# Resíduos
residuos = y_test - pred_test
colA, colB = st.columns(2)
colA.plotly_chart(
    px.scatter(x=pred_test, y=residuos,
               labels={"x": "Predito", "y": "Resíduo"},
               title="Resíduos vs Predito").add_hline(y=0, line_dash="dash"),
    use_container_width=True,
)

# QQ-plot manual
resid_sorted = np.sort(residuos)
theor = stats.norm.ppf((np.arange(1, len(resid_sorted) + 1) - 0.5) / len(resid_sorted),
                      loc=np.mean(resid_sorted), scale=np.std(resid_sorted, ddof=1))
fig_qq = px.scatter(x=theor, y=resid_sorted,
                    labels={"x": "Quantis teóricos (Normal)", "y": "Resíduos ordenados"},
                    title="QQ-Plot dos resíduos")
fig_qq.add_trace(go.Scatter(x=theor, y=theor, mode="lines",
                            line=dict(dash="dash", color="red"), name="y=x"))
colB.plotly_chart(fig_qq, use_container_width=True)

# Shapiro-Wilk múltipla
sw_stat, sw_p = stats.shapiro(residuos)
st.caption(f"**Shapiro-Wilk:** W={sw_stat:.3f}, p={sw_p:.4f} - "
           + ("normalidade aceitável ✅" if sw_p > 0.05 else "não-normalidade ⚠️"))

# Summary statsmodels
with st.expander("📑 Summary estatístico (statsmodels OLS - múltipla)"):
    Xsm = sm.add_constant(X_train).astype(float)
    ols = sm.OLS(y_train.astype(float), Xsm).fit()
    st.text(ols.summary().as_text())

st.divider()

# =========================================================================
# 3. Conclusões
# =========================================================================
st.header("3. Conclusões")
st.markdown(
    f"""
#### Simples
- Foi ajustada uma regressão OLS com **{x_var}** como preditor e
  **`vl_agropecuaria`** como alvo (amostras treino/teste 70/30).
- Coeficiente β₁ = **{modelo_s.coef_[0]:.4f}**; R² treino = **{r2_score(y_train, pred_train):.3f}**.

#### Múltipla
- Preditores: {', '.join(list(Xm.columns)) or '—'}.
- R² treino = **{r2_score(y_train, pred_train):.3f}**, R² teste = **{r2_score(y_test, pred_test):.3f}**.
- **Transformação log1p** estabiliza a variância e reduz a influência de
  Brasília (que domina os valores absolutos da região).

#### Observações sobre qualidade dos modelos
- R² elevados **no treino** sem queda acentuada no teste sugerem boa
  generalização (dado o tamanho reduzido: n = {len(df)} municípios).
- O diagnóstico de resíduos (gráficos + Shapiro-Wilk) indica se as
  premissas de **normalidade** e **homoscedasticidade** estão razoavelmente
  atendidas. Violações severas justificam o uso da transformação log
  ou modelos alternativos (GLM com família Gamma, por exemplo).
- Considerar o **Valor Adicionado da Indústria e dos Serviços** como
  variáveis correlacionadas com a Agropecuária reflete a lógica de
  interdependência econômica regional (insumos, logística, demanda).
"""
)
