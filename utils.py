"""Funções utilitárias compartilhadas entre as páginas do dashboard."""


def fmt_brl_mil(valor_mil: float) -> str:
    """Converte R$ mil para string BRL compacta em pt-BR.

    Exemplos:
        11_792_497  → "R$ 11,79 bi"
           45_848   → "R$ 45,85 mi"
              500   → "R$ 500.000"
    """
    v = valor_mil * 1_000  # R$ mil → R$
    if v >= 1_000_000_000:
        return f"R$ {v / 1e9:.2f} bi".replace(".", ",")
    if v >= 1_000_000:
        return f"R$ {v / 1e6:.2f} mi".replace(".", ",")
    s = f"{v:,.0f}".replace(",", ".")
    return f"R$ {s}"


def fmt_brl_full(valor_mil: float) -> str:
    """Converte R$ mil para string BRL completa pt-BR com separadores corretos.

    Exemplos:
        1_064_604  → "R$ 1.064.604.000,00"
           45_848  → "R$ 45.848.000,00"
    """
    v = valor_mil * 1_000  # R$ mil → R$
    s = f"{v:,.2f}"                          # "45,848,000.00"  (US)
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # "45.848.000,00" (BR)
    return f"R$ {s}"


def fmt_brl_reais(valor: float) -> str:
    """Formata valor já em R$ (não em mil) para pt-BR.

    Usado para PIB per capita (que já vem em R$, não R$ mil).
    """
    s = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"
