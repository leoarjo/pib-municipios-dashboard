"""
Trabalho A2 - Modelos Lineares Generalizados (IESB 2026.1)
Análise do PIB dos Municípios da RIDE-DF (IBGE).

Estrutura:
    - Parte 1 : Captação e avaliação da qualidade dos dados
    - Parte 2 : Painéis, mapas, correlações/associações e modelos de regressão
"""
import streamlit as st

from db import get_available_years, get_pib_ride, get_ride_df_municipios
from utils import fmt_brl_mil

st.set_page_config(
    page_title="PIB RIDE-DF - Trabalho A2",
    page_icon="📊",
    layout="wide",
)

st.title("📊 PIB dos Municípios da RIDE-DF")
st.caption("Trabalho A2 - Modelos Lineares Generalizados - IESB 2026.1")

# ------------------------------------------------------------------
# Filtros globais (persistidos em session_state)
# ------------------------------------------------------------------
anos = get_available_years()
ano_default = max(anos)

with st.sidebar:
    st.header("Filtros globais")
    ano = st.selectbox(
        "Ano de referência",
        options=anos,
        index=anos.index(ano_default),
        help="Por padrão, o último ano disponível na base (2021).",
    )
    st.session_state["ano"] = int(ano)
    st.markdown(
        f"**Ano selecionado:** {ano}  \n"
        f"**Recorte territorial:** RIDE-DF (34 municípios)"
    )

# ------------------------------------------------------------------
# Introdução
# ------------------------------------------------------------------
st.markdown(
    """
### Sobre o trabalho

O objetivo é **desenvolver um modelo estatístico capaz de explicar os fatores
que influenciam o Valor Adicionado Bruto da Agropecuária** (`vl_agropecuaria`)
nos municípios da **Região Integrada de Desenvolvimento do Distrito Federal -
RIDE-DF**, considerando o **último ano disponível** na base do IBGE.

A base `pib_municipios` está hospedada no PostgreSQL do IESB
(`bigdata.dataiesb.com`) e é integrada ao Sistema de Contas Nacionais/Regionais
(SNA 2008 / CNAE 2.0).

### Navegação

Use o menu lateral para navegar pelas etapas:

1. **Parte 1 - Qualidade dos Dados**: estatísticas descritivas,
   *missing values*, *outliers* e anomalias.
2. **Parte 2 - Painéis de Informações**: visão geral do ano e da série
   histórica.
3. **Parte 2 - Mapas**: análises territoriais.
4. **Parte 2 - Correlações e Associações**: quantitativa × quantitativa,
   qualitativa × qualitativa, quantitativa × qualitativa.
5. **Parte 2 - Regressão Linear**: modelos simples e múltipla com
   treino/teste.
"""
)

# ------------------------------------------------------------------
# Visão geral resumida
# ------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

ride = get_ride_df_municipios()
df_ano = get_pib_ride(ano)

with col1:
    st.metric("Municípios na RIDE-DF", len(ride))
with col2:
    st.metric("Registros carregados (ano)", len(df_ano))
with col3:
    st.metric(
        "Soma do VAB Agropecuária",
        fmt_brl_mil(df_ano["vl_agropecuaria"].sum()),
    )

st.divider()

st.subheader("Amostra dos dados (ano selecionado)")
st.dataframe(
    df_ano[
        [
            "ano", "codigo_municipio_dv", "nome_municipio", "uf",
            "vl_agropecuaria", "vl_industria", "vl_servicos",
            "vl_administracao", "vl_bruto_total", "vl_pib", "vl_pib_per_capta",
        ]
    ].reset_index(drop=True),
    use_container_width=True,
    height=380,
)

with st.expander("📋 Lista completa dos municípios da RIDE-DF"):
    st.dataframe(
        ride[["codigo_municipio_dv", "nome_municipio", "uf"]]
        .sort_values(["uf", "nome_municipio"])
        .reset_index(drop=True),
        use_container_width=True,
    )
