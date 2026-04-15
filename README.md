# Trabalho A2 — Modelos Lineares Generalizados (IESB 2026.1)

Análise do **PIB dos municípios da RIDE-DF** (Região Integrada de
Desenvolvimento do Distrito Federal) no último ano disponível (2021),
com objetivo de modelar o **Valor Adicionado Bruto da Agropecuária**
(`vl_agropecuaria`).

Dados: tabela `pib_municipios` do PostgreSQL do IESB
(`bigdata.dataiesb.com`), cruzada com `municipio_ride_brasilia` (34
municípios) e `municipio` (nome/UF/lat-long).

## Estrutura

```
Trabalho Parte 1/
├── app.py                            # página inicial (filtros globais)
├── db.py                             # conexão PostgreSQL + helpers
├── pages/
│   ├── 1_Parte_1_Qualidade.py        # Parte 1 - qualidade dos dados
│   ├── 2_Parte_2_Paineis.py          # Parte 2 - painéis/dashboards
│   ├── 3_Parte_2_Mapas.py            # Parte 2 - mapas territoriais
│   ├── 4_Parte_2_Correlacoes.py      # Parte 2 - correlações/associações
│   └── 5_Parte_2_Regressao.py        # Parte 2 - regressão simples + múltipla
├── requirements.txt
├── run.bat
└── venv/                             # ambiente virtual (gerado)
```

## Como executar (Windows)

Duplo-clique em `run.bat` **ou** via terminal:

```bash
# 1) ativar o ambiente virtual
venv\Scripts\activate

# 2) subir o Streamlit
streamlit run app.py
```

A aplicação abre em `http://localhost:8501`.

## Recriar o ambiente (se necessário)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Conteúdo das páginas

| Página | Conteúdo |
| --- | --- |
| **Home** (`app.py`) | Filtros globais (ano, RIDE-DF), visão geral, amostra e lista dos 34 municípios |
| **Parte 1 — Qualidade** | Estatísticas descritivas, missing, duplicatas, outliers (IQR), histogramas, validações de integridade |
| **Parte 2 — Painéis** | KPIs, composição setorial, Top-10 municípios, distribuição por UF, série histórica 2010-2021 |
| **Parte 2 — Mapas** | Mapa Plotly (scatter mapbox), Folium com popups, animação temporal |
| **Parte 2 — Correlações** | Matriz Pearson/Spearman, scatter matrix, V de Cramér (qual×qual), ANOVA (quant×qual) |
| **Parte 2 — Regressão** | Regressão simples e múltipla, treino/teste 70/30, R², RMSE, VIF, QQ-plot, resíduos, Shapiro-Wilk, summary statsmodels |

## Credenciais do banco

A URI do Postgres é lida de `st.secrets["database"]["uri"]`, com fallback
para a variável de ambiente `DB_URI` e, em último caso, para o banco
institucional padrão (útil no desenvolvimento local).

Para rodar localmente com secrets, copie o template:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

## Deploy no Streamlit Cloud

1. Conecte este repositório em https://share.streamlit.io
2. **Main file path**: `app.py`
3. **Python version**: 3.11
4. Em *App settings → Secrets*, cole:

```toml
[database]
uri = "postgresql+psycopg2://data_iesb:iesb@bigdata.dataiesb.com:5432/iesb"
```

5. *Deploy*. A aplicação abre em `https://<app>.streamlit.app`.
