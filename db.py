"""
Conexão com o banco PostgreSQL do IESB e funções de carregamento dos dados.

Tabelas usadas:
    - pib_municipios        : fato (PIB municipal 2010-2021)
    - municipio_ride_brasilia : dimensão com os 34 municípios da RIDE-DF
    - municipio             : dimensão geral (nome, UF, lat/long)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


def _build_db_uri() -> str:
    """Monta a URI do Postgres lendo, nesta ordem: st.secrets -> env vars -> default.

    No Streamlit Cloud configurar em Settings -> Secrets:
        [database]
        uri = "postgresql+psycopg2://user:pass@host:5432/dbname"
    """
    try:
        if "database" in st.secrets and "uri" in st.secrets["database"]:
            return st.secrets["database"]["uri"]
    except Exception:
        pass
    env = os.environ.get("DB_URI")
    if env:
        return env
    # fallback para desenvolvimento local (banco institucional da disciplina)
    return "postgresql+psycopg2://data_iesb:iesb@bigdata.dataiesb.com:5432/iesb"


DB_URI = _build_db_uri()

# Mapa de códigos de UF do IBGE
UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}


@st.cache_resource
def get_engine():
    return create_engine(DB_URI, pool_pre_ping=True)


@st.cache_data(ttl=3600)
def get_available_years() -> list[int]:
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            "SELECT DISTINCT ano_pib FROM pib_municipios ORDER BY ano_pib;", conn
        )
    return sorted(df["ano_pib"].astype(int).tolist())


@st.cache_data(ttl=3600)
def get_ride_df_municipios() -> pd.DataFrame:
    """Retorna os 34 municípios da RIDE-DF com código, nome, UF, lat/long."""
    eng = get_engine()
    q = """
    SELECT r.codigo_municipio_dv,
           m.nome_municipio,
           m.cd_uf,
           m.latitude,
           m.longitude
    FROM municipio_ride_brasilia r
    LEFT JOIN municipio m USING (codigo_municipio_dv)
    ORDER BY m.cd_uf, m.nome_municipio;
    """
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
    df["uf"] = df["cd_uf"].map(UF_MAP)
    return df


@st.cache_data(ttl=3600)
def get_pib_ride(year: int | None = None) -> pd.DataFrame:
    """
    PIB dos municípios da RIDE-DF, já enriquecido com nome/UF/lat/long.
    Se `year` for None, retorna a série completa (2010-2021).
    """
    eng = get_engine()
    base = """
    SELECT p.ano_pib::int           AS ano,
           p.codigo_municipio_dv,
           m.nome_municipio,
           m.cd_uf,
           m.latitude,
           m.longitude,
           p.vl_agropecuaria,
           p.vl_industria,
           p.vl_servicos,
           p.vl_administracao,
           p.vl_bruto_total,
           p.vl_subsidios,
           p.vl_pib,
           p.vl_pib_per_capta
    FROM pib_municipios p
    JOIN municipio_ride_brasilia r USING (codigo_municipio_dv)
    LEFT JOIN municipio m          USING (codigo_municipio_dv)
    """
    params = {}
    if year is not None:
        base += " WHERE p.ano_pib = :ano"
        params["ano"] = str(year)
    base += " ORDER BY p.ano_pib, m.nome_municipio;"

    with eng.connect() as conn:
        df = pd.read_sql(text(base), conn, params=params)

    df["uf"] = df["cd_uf"].map(UF_MAP)
    num_cols = [
        "vl_agropecuaria", "vl_industria", "vl_servicos", "vl_administracao",
        "vl_bruto_total", "vl_subsidios", "vl_pib", "vl_pib_per_capta",
        "latitude", "longitude",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def setor_predominante(row) -> str:
    """Retorna o setor de maior valor adicionado bruto do município."""
    setores = {
        "Agropecuária": row["vl_agropecuaria"],
        "Indústria": row["vl_industria"],
        "Serviços": row["vl_servicos"],
        "Administração Pública": row["vl_administracao"],
    }
    return max(setores, key=setores.get)


def classe_porte(valor: float, q1: float, q3: float) -> str:
    """Classifica o porte econômico (Pequeno/Médio/Grande) por tercis do VAB total."""
    if valor <= q1:
        return "Pequeno"
    if valor <= q3:
        return "Médio"
    return "Grande"


def enriquecer_qualitativas(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona variáveis qualitativas úteis para a análise."""
    df = df.copy()
    df["setor_predominante"] = df.apply(setor_predominante, axis=1)
    # Porte por tercis do VAB total (por ano, se houver múltiplos anos)
    def _atribuir_porte(g: pd.DataFrame) -> pd.DataFrame:
        q1, q3 = g["vl_bruto_total"].quantile([1 / 3, 2 / 3])
        g["porte"] = g["vl_bruto_total"].apply(lambda v: classe_porte(v, q1, q3))
        return g

    if "ano" in df.columns:
        df["porte"] = (
            df.groupby("ano")["vl_bruto_total"]
            .transform(lambda s: pd.cut(
                s.rank(pct=True),
                bins=[0, 1 / 3, 2 / 3, 1.0001],
                labels=["Pequeno", "Médio", "Grande"],
                include_lowest=True,
            ))
            .astype(str)
        )
    else:
        q1, q3 = df["vl_bruto_total"].quantile([1 / 3, 2 / 3])
        df["porte"] = df["vl_bruto_total"].apply(lambda v: classe_porte(v, q1, q3))
    return df
