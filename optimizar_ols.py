"""
Búsqueda automática de la mejor especificación OLS para predecir log(revenue)
en el dataset data__movies.csv.

Qué hace:
1. Carga y limpia los datos con mejoras respecto al notebook original.
2. Define un pool grande de variables candidatas (efectos principales + interacciones).
3. Corre selección stepwise forward-backward usando AIC.
4. Ajusta el modelo final y reporta coeficientes, p-values, R², AIC, BIC, VIF.
5. Imprime un diagnóstico comparativo con el modelo del notebook.

Uso:
    python optimizar_ols.py
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

CSV_PATH = "data__movies.csv"


# ============================================================
# 1) LIMPIEZA (mejorada respecto al notebook)
# ============================================================
def cargar_y_limpiar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.drop(columns=[
        "homepage", "status", "tagline", "title",
        "original_title", "overview", "id",
    ])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.lower()

    df["genres"] = df["genres"].fillna("desconocido")
    df["production_companies"] = df["production_companies"].fillna("otros")
    df = df.dropna(subset=["release_date", "runtime"])

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df["anio"] = df["release_date"].dt.year.astype(int)
    df["mes"] = df["release_date"].dt.month.astype(int)
    df = df.drop(columns=["release_date"])

    df = df[(df["budget"] > 0) & (df["revenue"] > 0)
            & (df["runtime"] > 0) & (df["vote_count"] > 0)].copy()

    generos_expandidos = df["genres"].str.split(",").apply(
        lambda lista: [g.strip() for g in lista] if isinstance(lista, list) else []
    )
    top_generos = pd.Series(
        [g for sub in generos_expandidos for g in sub]
    ).value_counts().head(10).index.tolist()

    def limpiar_nombre_gen(s: str) -> str:
        out = s
        for ch in [" ", ".", ",", "-", "(", ")", "/", "'", "&"]:
            out = out.replace(ch, "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    for g in top_generos:
        df[f"gen_{limpiar_nombre_gen(g)}"] = generos_expandidos.apply(
            lambda lst: int(g in lst)
        )
    df["n_generos"] = generos_expandidos.apply(len)

    prods_expandidas = df["production_companies"].str.split(",").apply(
        lambda lista: [p.strip() for p in lista] if isinstance(lista, list) else []
    )
    top_prods = pd.Series(
        [p for sub in prods_expandidas for p in sub]
    ).value_counts().head(10).index.tolist()
    def limpiar_nombre(s: str) -> str:
        out = s
        for ch in [" ", ".", ",", "-", "(", ")", "/", "'", "&"]:
            out = out.replace(ch, "_")
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_")

    for p in top_prods:
        col = "prod_" + limpiar_nombre(p)
        df[col] = prods_expandidas.apply(lambda lst: int(p in lst))

    top_lang = df["original_language"].value_counts().head(5).index
    df["original_language"] = df["original_language"].where(
        df["original_language"].isin(top_lang), "other"
    )

    df["log_budget"] = np.log(df["budget"])
    df["log_revenue"] = np.log(df["revenue"])
    df["log_popularity"] = np.log1p(df["popularity"])
    df["log_vote_count"] = np.log1p(df["vote_count"])

    df = df.drop(columns=["genres", "production_companies",
                          "budget", "revenue", "popularity", "vote_count"])

    q_low = df["log_revenue"].quantile(0.005)
    q_high = df["log_revenue"].quantile(0.995)
    df = df[(df["log_revenue"] >= q_low) & (df["log_revenue"] <= q_high)].copy()

    # Centrado de variables numéricas (reduce multicolinealidad en interacciones
    # y hace los coeficientes principales interpretables como "efecto en el promedio").
    for col in ["log_budget", "log_popularity", "log_vote_count",
                "vote_average", "runtime", "anio"]:
        df[col] = df[col] - df[col].mean()

    return df.reset_index(drop=True)


# ============================================================
# 2) POOL DE FEATURES CANDIDATAS
# ============================================================
def construir_candidatas(df: pd.DataFrame) -> list[str]:
    num_base = ["log_budget", "log_popularity", "log_vote_count",
                "vote_average", "runtime", "anio", "n_generos"]

    cat = ['C(original_language)', 'C(mes)']

    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    prod_cols = [c for c in df.columns if c.startswith("prod_")]

    inter_num = [
        "log_budget:log_popularity",
        "log_budget:log_vote_count",
        "log_budget:vote_average",
        "log_popularity:log_vote_count",
        "log_budget:runtime",
        "vote_average:log_popularity",
    ]

    inter_cat_num = [f"log_budget:{g}" for g in gen_cols] + \
                    [f"log_budget:{p}" for p in prod_cols]

    candidatas = num_base + cat + gen_cols + prod_cols + inter_num + inter_cat_num
    return candidatas


# ============================================================
# 3) FORWARD-BACKWARD STEPWISE POR AIC
# ============================================================
def ajustar(df, y, terms):
    formula = f"{y} ~ " + (" + ".join(terms) if terms else "1")
    return smf.ols(formula, data=df).fit()


def stepwise_aic(df, y, candidatas, verbose=True):
    seleccionadas = []
    mejor_aic = ajustar(df, y, ["1"]).aic
    cambio = True
    iteracion = 0

    while cambio:
        iteracion += 1
        cambio = False

        mejor_add = (None, mejor_aic)
        for c in candidatas:
            if c in seleccionadas:
                continue
            try:
                aic = ajustar(df, y, seleccionadas + [c]).aic
            except Exception:
                continue
            if aic < mejor_add[1] - 0.01:
                mejor_add = (c, aic)

        if mejor_add[0] is not None:
            seleccionadas.append(mejor_add[0])
            mejor_aic = mejor_add[1]
            cambio = True
            if verbose:
                print(f"  [it {iteracion}] + {mejor_add[0]:<45s}  AIC={mejor_aic:,.1f}")
            continue

        mejor_drop = (None, mejor_aic)
        for c in seleccionadas:
            candidato_set = [x for x in seleccionadas if x != c]
            try:
                aic = ajustar(df, y, candidato_set if candidato_set else ["1"]).aic
            except Exception:
                continue
            if aic < mejor_drop[1] - 0.01:
                mejor_drop = (c, aic)

        if mejor_drop[0] is not None:
            seleccionadas.remove(mejor_drop[0])
            mejor_aic = mejor_drop[1]
            cambio = True
            if verbose:
                print(f"  [it {iteracion}] - {mejor_drop[0]:<45s}  AIC={mejor_aic:,.1f}")

    return seleccionadas, mejor_aic


# ============================================================
# 4) CALCULAR VIF PARA DIAGNOSTICAR COLINEALIDAD
# ============================================================
def calcular_vif(modelo) -> pd.DataFrame:
    X = modelo.model.exog
    nombres = modelo.model.exog_names
    vifs = []
    for i, nom in enumerate(nombres):
        if nom == "Intercept":
            continue
        try:
            vif = variance_inflation_factor(X, i)
        except Exception:
            vif = np.nan
        vifs.append((nom, vif))
    out = pd.DataFrame(vifs, columns=["variable", "VIF"]).sort_values(
        "VIF", ascending=False
    )
    return out


# ============================================================
# 5) MAIN
# ============================================================
def main():
    print("=" * 72)
    print("CARGA Y LIMPIEZA")
    print("=" * 72)
    df = cargar_y_limpiar(CSV_PATH)
    print(f"N final tras limpieza: {len(df):,} filas")
    print(f"Columnas: {len(df.columns)}")
    print(f"Rango log_revenue: [{df['log_revenue'].min():.2f}, {df['log_revenue'].max():.2f}]")

    print("\n" + "=" * 72)
    print("MODELO BASELINE (replicando el notebook, solo efectos principales)")
    print("=" * 72)
    baseline_terms = [
        "log_budget", "log_popularity", "vote_average",
        "C(original_language)",
    ] + [c for c in df.columns if c.startswith("gen_")] \
      + [c for c in df.columns if c.startswith("prod_")]
    modelo_base = ajustar(df, "log_revenue", baseline_terms)
    print(f"R²         = {modelo_base.rsquared:.4f}")
    print(f"R² ajust   = {modelo_base.rsquared_adj:.4f}")
    print(f"AIC        = {modelo_base.aic:,.1f}")
    print(f"BIC        = {modelo_base.bic:,.1f}")
    print(f"cond_num   = {modelo_base.condition_number:,.0f}")
    print(f"n_params   = {int(modelo_base.df_model)}")

    print("\n" + "=" * 72)
    print("STEPWISE FORWARD-BACKWARD (AIC)")
    print("=" * 72)
    candidatas = construir_candidatas(df)
    print(f"Pool de {len(candidatas)} candidatas. Corriendo stepwise...\n")
    seleccionadas, aic_final = stepwise_aic(df, "log_revenue", candidatas, verbose=True)

    print("\n" + "=" * 72)
    print("MODELO FINAL SELECCIONADO")
    print("=" * 72)
    modelo_final = ajustar(df, "log_revenue", seleccionadas)

    print(f"R²         = {modelo_final.rsquared:.4f}")
    print(f"R² ajust   = {modelo_final.rsquared_adj:.4f}")
    print(f"AIC        = {modelo_final.aic:,.1f}")
    print(f"BIC        = {modelo_final.bic:,.1f}")
    print(f"cond_num   = {modelo_final.condition_number:,.0f}")
    print(f"n_params   = {int(modelo_final.df_model)}")

    print("\nTérminos seleccionados (orden de inclusión):")
    for i, t in enumerate(seleccionadas, 1):
        print(f"  {i:2d}. {t}")

    print("\nCoeficientes con p-value (ordenados por p):")
    tabla = pd.DataFrame({
        "coef": modelo_final.params,
        "std_err": modelo_final.bse,
        "p_value": modelo_final.pvalues,
    }).sort_values("p_value")
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)
    print(tabla.round(4))

    n_signif = (tabla["p_value"] < 0.05).sum()
    n_total = len(tabla)
    print(f"\nSignificativos (p < 0.05): {n_signif}/{n_total}  ({100*n_signif/n_total:.0f}%)")

    print("\n" + "=" * 72)
    print("COMPARACIÓN FINAL")
    print("=" * 72)
    comp = pd.DataFrame({
        "R2":       [modelo_base.rsquared,     modelo_final.rsquared],
        "R2_adj":   [modelo_base.rsquared_adj, modelo_final.rsquared_adj],
        "AIC":      [modelo_base.aic,          modelo_final.aic],
        "BIC":      [modelo_base.bic,          modelo_final.bic],
        "cond_num": [modelo_base.condition_number, modelo_final.condition_number],
        "n_params": [int(modelo_base.df_model), int(modelo_final.df_model)],
    }, index=["Baseline (notebook-like)", "Modelo stepwise"]).round(4)
    print(comp)

    print("\nTop 10 variables con mayor VIF:")
    try:
        vifs = calcular_vif(modelo_final)
        print(vifs.head(10).to_string(index=False))
    except Exception as e:
        print(f"  (no se pudo calcular VIF: {e})")

    print("\n" + "=" * 72)
    print("FÓRMULA SUGERIDA PARA EL NOTEBOOK")
    print("=" * 72)
    print(f"log_revenue ~ {' + '.join(seleccionadas)}")


if __name__ == "__main__":
    main()
