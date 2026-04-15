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
from statsmodels.stats.diagnostic import het_breuschpagan
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
    "(recomendado — distribuições muito assimétricas)", value=True,
)

def prep(s: pd.Series) -> pd.Series:
    return np.log1p(s) if usar_log else s

TARGET = "vl_agropecuaria"

st.info(f"Ano: **{ano}** | Observações: **{len(df)}** | "
        f"Transformação: **{'log1p' if usar_log else 'nenhuma'}**")

st.warning(
    "⚠️ **Nota metodológica — divisão treino/teste com n = 34:** "
    "a partição aleatória 70/30 é adotada por exigência pedagógica. "
    "Com apenas 34 municípios de perfis estruturalmente distintos "
    "(Brasília — economia de serviços; municípios industriais de GO; "
    "pequenos municípios rurais de MG), a amostragem aleatória pode "
    "concentrar pontos extremos em um único conjunto, distorcendo as métricas. "
    "Em estudos aplicados, seria preferível validação cruzada *leave-one-out* "
    "ou estratificação por UF."
)

# =========================================================================
# 0. DIAGNÓSTICO PRÉ-MODELAGEM
# =========================================================================
st.header("0. Diagnóstico Pré-Modelagem")
st.markdown(
    """
    Antes de ajustar qualquer modelo, verificamos se as premissas da regressão
    linear clássica são razoavelmente atendidas nos dados:

    1. **Normalidade** da variável resposta (Shapiro-Wilk)
    2. **Linearidade** da relação X → Y (scatter com reta de tendência)
    3. **Multicolinearidade** entre preditores (VIF — Fator de Inflação de Variância)

    Mesmo que alguma premissa seja violada, o modelo é ajustado por ser um
    estudo acadêmico — as violações são documentadas e discutidas nas conclusões.
    """
)

y_all = prep(df[TARGET])

# Distribuição da variável resposta
col1, col2 = st.columns([2, 1])
with col1:
    label_y = f"log1p({TARGET})" if usar_log else TARGET
    st.plotly_chart(
        px.histogram(y_all, nbins=15, title=f"Distribuição de {label_y}",
                     labels={"value": label_y, "count": "Frequência"}),
        use_container_width=True,
    )
with col2:
    sw_pre, sw_p_pre = stats.shapiro(y_all)
    st.metric("Shapiro-Wilk W", f"{sw_pre:.3f}")
    st.metric("p-valor (normalidade)", f"{sw_p_pre:.4f}")
    if sw_p_pre > 0.05:
        st.success("✅ Não rejeitamos normalidade da variável resposta (p > 0,05).")
    else:
        st.warning(
            "⚠️ Indício de não-normalidade (p ≤ 0,05). "
            "A transformação log1p pode atenuar este problema."
        )

# VIF pré-modelagem nos preditores numéricos padrão
st.subheader("Multicolinearidade (VIF) — preditores numéricos padrão")
feats_vif_default = ["vl_industria", "vl_servicos", "vl_administracao", "vl_pib_per_capta"]
Xvif = prep(df[feats_vif_default]).dropna()
vif_pre = pd.DataFrame({
    "Variável": Xvif.columns,
    "VIF": [variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1])],
})
vif_pre["VIF"] = vif_pre["VIF"].round(2)
vif_pre["Avaliação"] = vif_pre["VIF"].apply(
    lambda v: "✅ Aceitável" if v < 5 else ("⚠️ Moderado" if v < 10 else "❌ Severo")
)
st.dataframe(vif_pre, use_container_width=True, hide_index=True)
st.caption("Regra prática: VIF < 5 aceitável; VIF > 10 indica multicolinearidade severa.")

st.divider()

# =========================================================================
# 1. REGRESSÃO LINEAR SIMPLES
# =========================================================================
st.header("1. Regressão Linear Simples")

preditores_simples = ["vl_industria", "vl_servicos", "vl_administracao",
                      "vl_bruto_total", "vl_pib", "vl_pib_per_capta"]
x_var = st.selectbox("Variável preditora (X)", preditores_simples, index=0)

X = prep(df[[x_var]])
y = prep(df[TARGET])

# Verificação de linearidade ANTES do ajuste
st.subheader("Verificação de linearidade (X vs Y)")
label_x = f"log1p({x_var})" if usar_log else x_var
label_y = f"log1p({TARGET})" if usar_log else TARGET
fig_lin = px.scatter(
    x=X[x_var], y=y,
    labels={"x": label_x, "y": label_y},
    trendline="ols",
    title="Dispersão X × Y com reta de tendência",
)
st.plotly_chart(fig_lin, use_container_width=True)
r_lin, p_lin = stats.pearsonr(X[x_var], y)
st.caption(f"Correlação de Pearson: r = {r_lin:.3f}  (p = {p_lin:.4f})")

# Ajuste do modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
modelo_s = LinearRegression().fit(X_train, y_train)
pred_train_s = modelo_s.predict(X_train)
pred_test_s  = modelo_s.predict(X_test)

r2_tr_s = r2_score(y_train, pred_train_s)
r2_te_s = r2_score(y_test,  pred_test_s)

st.subheader("Resultados do modelo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("β₀ (intercepto)", f"{modelo_s.intercept_:.4f}")
c2.metric(f"β₁ ({x_var})",   f"{modelo_s.coef_[0]:.4f}")
c3.metric("R² treino",        f"{r2_tr_s:.3f}")
c4.metric("R² teste",         f"{r2_te_s:.3f}")

c1, c2 = st.columns(2)
c1.metric("RMSE teste", f"{np.sqrt(mean_squared_error(y_test, pred_test_s)):.3f}")
c2.metric("MAE teste",  f"{mean_absolute_error(y_test, pred_test_s):.3f}")

# Diagnóstico pós-ajuste (resíduos do treino — mais pontos)
st.subheader("Diagnóstico pós-ajuste (resíduos)")
Xsm_s   = sm.add_constant(X_train)
ols_s   = sm.OLS(y_train, Xsm_s).fit()
resid_s = ols_s.resid
bp_s    = het_breuschpagan(resid_s, Xsm_s)
sw_s, sw_p_s = stats.shapiro(resid_s)

colA, colB = st.columns(2)
colA.plotly_chart(
    px.scatter(x=pred_train_s, y=resid_s,
               labels={"x": "Predito", "y": "Resíduo"},
               title="Resíduos vs Predito (treino)").add_hline(y=0, line_dash="dash"),
    use_container_width=True,
)
colB.plotly_chart(
    px.histogram(resid_s, nbins=15, title="Distribuição dos resíduos (treino)"),
    use_container_width=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("Shapiro-Wilk W", f"{sw_s:.3f}")
col2.metric("p-valor normalidade", f"{sw_p_s:.4f}",
            help="p > 0,05: não rejeitamos normalidade dos resíduos")
col3.metric("Breusch-Pagan (p-valor)", f"{bp_s[1]:.4f}",
            help="p > 0,05: não rejeitamos homocedasticidade")

st.caption(
    f"**Normalidade (Shapiro-Wilk):** W={sw_s:.3f}, p={sw_p_s:.4f} — "
    + ("✅ Normal" if sw_p_s > 0.05 else "⚠️ Não-normal")
    + f"  |  **Homocedasticidade (Breusch-Pagan):** p={bp_s[1]:.4f} — "
    + ("✅ Homocedástico" if bp_s[1] > 0.05 else "⚠️ Heterocedástico")
)

with st.expander("📑 Summary estatístico (statsmodels OLS — simples)"):
    st.text(ols_s.summary().as_text())

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

# VIF pré-ajuste (apenas numéricos selecionados, requer ≥ 2)
if len(feats_num) >= 2:
    st.subheader("Multicolinearidade pré-ajuste (VIF dos preditores selecionados)")
    Xn = Xm_num.dropna()
    vif_m = pd.DataFrame({
        "Variável": Xn.columns,
        "VIF": [variance_inflation_factor(Xn.values, i) for i in range(Xn.shape[1])],
    })
    vif_m["VIF"] = vif_m["VIF"].round(2)
    vif_m["Avaliação"] = vif_m["VIF"].apply(
        lambda v: "✅ Aceitável" if v < 5 else ("⚠️ Moderado" if v < 10 else "❌ Severo")
    )
    st.dataframe(vif_m, use_container_width=True, hide_index=True)
    st.caption("VIF > 10 indica multicolinearidade severa — prejudica a interpretação dos coeficientes.")

# Ajuste
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    Xm, ym, test_size=0.3, random_state=42
)
modelo_m     = LinearRegression().fit(X_train_m, y_train_m)
pred_train_m = modelo_m.predict(X_train_m)
pred_test_m  = modelo_m.predict(X_test_m)

r2_tr_m = r2_score(y_train_m, pred_train_m)
r2_te_m = r2_score(y_test_m,  pred_test_m)

st.subheader("Resultados do modelo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² treino", f"{r2_tr_m:.3f}")
c2.metric("R² teste",  f"{r2_te_m:.3f}")
n_, p_ = X_train_m.shape
r2_adj = 1 - (1 - r2_tr_m) * (n_ - 1) / (n_ - p_ - 1)
c3.metric("R² ajustado (treino)", f"{r2_adj:.3f}")
c4.metric("RMSE teste", f"{np.sqrt(mean_squared_error(y_test_m, pred_test_m)):.3f}")

# Coeficientes
coefs = pd.DataFrame({
    "Variável":     ["(intercepto)"] + list(Xm.columns),
    "Coeficiente":  [modelo_m.intercept_] + list(modelo_m.coef_),
})
st.subheader("Coeficientes estimados")
st.dataframe(coefs.style.format({"Coeficiente": "{:.4f}"}),
             use_container_width=True, hide_index=True)

# Predito vs Observado
st.subheader("Predito × Observado (conjunto de teste)")
plot_df  = pd.DataFrame({"observado": y_test_m.values, "predito": pred_test_m})
lim_min  = min(plot_df.min()) - 0.5
lim_max  = max(plot_df.max()) + 0.5
fig_po   = px.scatter(plot_df, x="observado", y="predito",
                      title="Predito vs Observado (teste)")
fig_po.add_trace(go.Scatter(x=[lim_min, lim_max], y=[lim_min, lim_max],
                            mode="lines", name="y=x",
                            line=dict(dash="dash", color="red")))
st.plotly_chart(fig_po, use_container_width=True)

# Diagnóstico pós-ajuste
st.subheader("Diagnóstico pós-ajuste (resíduos)")
Xsm_m   = sm.add_constant(X_train_m).astype(float)
ols_m   = sm.OLS(y_train_m.astype(float), Xsm_m).fit()
resid_m = ols_m.resid
bp_m    = het_breuschpagan(resid_m, Xsm_m)
sw_m, sw_p_m = stats.shapiro(resid_m)

colA, colB = st.columns(2)
colA.plotly_chart(
    px.scatter(x=pred_train_m, y=resid_m,
               labels={"x": "Predito", "y": "Resíduo"},
               title="Resíduos vs Predito (treino)").add_hline(y=0, line_dash="dash"),
    use_container_width=True,
)

# QQ-plot
resid_sorted = np.sort(resid_m)
theor = stats.norm.ppf(
    (np.arange(1, len(resid_sorted) + 1) - 0.5) / len(resid_sorted),
    loc=np.mean(resid_sorted), scale=np.std(resid_sorted, ddof=1),
)
fig_qq = px.scatter(x=theor, y=resid_sorted,
                    labels={"x": "Quantis teóricos (Normal)", "y": "Resíduos ordenados"},
                    title="QQ-Plot dos resíduos (treino)")
fig_qq.add_trace(go.Scatter(x=theor, y=theor, mode="lines",
                            line=dict(dash="dash", color="red"), name="y=x"))
colB.plotly_chart(fig_qq, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Shapiro-Wilk W", f"{sw_m:.3f}")
col2.metric("p-valor normalidade", f"{sw_p_m:.4f}")
col3.metric("Breusch-Pagan (p-valor)", f"{bp_m[1]:.4f}")

st.caption(
    f"**Normalidade (Shapiro-Wilk):** W={sw_m:.3f}, p={sw_p_m:.4f} — "
    + ("✅ Normal" if sw_p_m > 0.05 else "⚠️ Não-normal")
    + f"  |  **Homocedasticidade (Breusch-Pagan):** p={bp_m[1]:.4f} — "
    + ("✅ Homocedástico" if bp_m[1] > 0.05 else "⚠️ Heterocedástico")
)

with st.expander("📑 Summary estatístico (statsmodels OLS — múltipla)"):
    st.text(ols_m.summary().as_text())

st.divider()

# =========================================================================
# 3. CONCLUSÕES
# =========================================================================
st.header("3. Conclusões")
st.markdown(
    f"""
#### Regressão Simples
- Preditor: **{x_var}** → Alvo: **vl_agropecuaria** (split treino/teste 70/30,
  n_treino = {len(X_train)}, n_teste = {len(X_test)}).
- β₁ = **{modelo_s.coef_[0]:.4f}** | R² treino = **{r2_tr_s:.3f}** | R² teste = **{r2_te_s:.3f}**.
- O R² de **{r2_tr_s:.3f}** indica que o modelo explica apenas
  {r2_tr_s * 100:.1f}% da variância — desempenho **fraco**, esperado dado o
  tamanho reduzido da amostra (n = {len(df)}) e a heterogeneidade estrutural
  dos municípios.

#### Regressão Múltipla
- Preditores: {', '.join(list(Xm.columns)) or '—'}.
- R² treino = **{r2_tr_m:.3f}** | R² teste = **{r2_te_m:.3f}**.
- A **transformação log1p** estabiliza a variância e reduz a influência de
  Brasília, que domina os valores absolutos da região.

#### Avaliação da qualidade e limitações
- Com **n = {len(df)} municípios**, qualquer divisão treino/teste é sensível à
  alocação aleatória de pontos extremos (ex.: Brasília, Cristalina).
  R² negativos no teste indicam que o modelo generaliza pior que a média
  simples — sinal de *overfitting* por amostra insuficiente.
- A partição aleatória é **metodologicamente questionável** aqui: municípios têm
  perfis estruturais distintos (DF = serviços/governo; GO = agronegócio e
  indústria; MG = municípios rurais pequenos), tornando qualquer split aleatório
  potencialmente enviesado. Estratificação por UF ou validação cruzada
  *leave-one-out* seria mais adequada.
- Os diagnósticos pré e pós-ajuste (Shapiro-Wilk, Breusch-Pagan, VIF) foram
  realizados para verificar a viabilidade do modelo antes e após o ajuste.
  Violações identificadas justificam transformações adicionais ou modelos
  alternativos (GLM com família Gamma e log-link, por exemplo).
"""
)
