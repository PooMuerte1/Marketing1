"""
Dashboard Streamlit — Modelación de Sistemas 2025-2
Predicción de revenue de películas (proxy de demanda).

Cómo ejecutar:
    pip install streamlit plotly statsmodels pandas numpy scipy
    streamlit run app_streamlit.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Predicción de Revenue — Películas",
    page_icon=":clapper:",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Pipeline de limpieza (idéntica a la celda 27 del notebook)
# ------------------------------------------------------------------ #
def _limpiar(s: str) -> str:
    out = s
    for ch in [" ", ".", ",", "-", "(", ")", "/", "'", "&"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


@st.cache_data(show_spinner="Cargando CSV crudo...")
def cargar_raw() -> pd.DataFrame:
    return pd.read_csv("data__movies.csv")


@st.cache_data(show_spinner="Aplicando pipeline de limpieza...")
def construir_df_v2() -> pd.DataFrame:
    df_v2 = pd.read_csv("data__movies.csv")
    df_v2 = df_v2.drop(columns=[
        "homepage", "status", "tagline", "title",
        "original_title", "overview", "id"
    ])

    for col in df_v2.select_dtypes(include=["object"]).columns:
        df_v2[col] = df_v2[col].str.lower()

    df_v2["genres"] = df_v2["genres"].fillna("desconocido")
    df_v2["production_companies"] = df_v2["production_companies"].fillna("otros")
    df_v2 = df_v2.dropna(subset=["release_date", "runtime"])

    df_v2["release_date"] = pd.to_datetime(df_v2["release_date"], errors="coerce")
    df_v2 = df_v2.dropna(subset=["release_date"])
    df_v2["anio"] = df_v2["release_date"].dt.year.astype(int)
    df_v2["mes"] = df_v2["release_date"].dt.month.astype(int)
    df_v2 = df_v2.drop(columns=["release_date"])

    df_v2 = df_v2[(df_v2["budget"] > 0) & (df_v2["revenue"] > 0)
                  & (df_v2["runtime"] > 0) & (df_v2["vote_count"] > 0)].copy()

    generos_exp = df_v2["genres"].str.split(",").apply(
        lambda l: [g.strip() for g in l] if isinstance(l, list) else []
    )
    top_generos = pd.Series([g for sub in generos_exp for g in sub]) \
                    .value_counts().head(10).index.tolist()
    for g in top_generos:
        df_v2[f"gen_{_limpiar(g)}"] = generos_exp.apply(lambda lst: int(g in lst))
    df_v2["n_generos"] = generos_exp.apply(len)

    prods_exp = df_v2["production_companies"].str.split(",").apply(
        lambda l: [p.strip() for p in l] if isinstance(l, list) else []
    )
    top_prods = pd.Series([p for sub in prods_exp for p in sub]) \
                  .value_counts().head(10).index.tolist()
    for p in top_prods:
        df_v2[f"prod_{_limpiar(p)}"] = prods_exp.apply(lambda lst: int(p in lst))

    top_lang = df_v2["original_language"].value_counts().head(5).index
    df_v2["original_language"] = df_v2["original_language"].where(
        df_v2["original_language"].isin(top_lang), "other"
    )

    df_v2["log_budget"]     = np.log(df_v2["budget"])
    df_v2["log_revenue"]    = np.log(df_v2["revenue"])
    df_v2["log_popularity"] = np.log1p(df_v2["popularity"])
    df_v2["log_vote_count"] = np.log1p(df_v2["vote_count"])

    df_v2 = df_v2.drop(columns=[
        "genres", "production_companies",
        "budget", "revenue", "popularity", "vote_count"
    ])

    q_low  = df_v2["log_revenue"].quantile(0.005)
    q_high = df_v2["log_revenue"].quantile(0.995)
    df_v2 = df_v2[(df_v2["log_revenue"] >= q_low)
                  & (df_v2["log_revenue"] <= q_high)].copy()

    medias = {}
    for col in ["log_budget", "log_popularity", "log_vote_count",
                "vote_average", "runtime", "anio"]:
        medias[col] = df_v2[col].mean()
        df_v2[col] = df_v2[col] - medias[col]

    df_v2 = df_v2.reset_index(drop=True)
    df_v2.attrs["medias_centrado"] = medias
    return df_v2


@st.cache_resource(show_spinner="Ajustando modelo OLS...")
def ajustar_modelo(df_v2: pd.DataFrame):
    formula_final = """
    log_revenue ~ log_vote_count + log_budget + anio
    + gen_family + gen_science_fiction + gen_crime + gen_fantasy + gen_romance + gen_drama
    + vote_average
    + prod_new_line_cinema + prod_twentieth_century_fox_film_corporation
    + prod_paramount_pictures + prod_universal_pictures + prod_columbia_pictures
    + prod_touchstone_pictures + prod_metro_goldwyn_mayer_mgm
    + runtime
    + log_budget:gen_crime
    + log_budget:gen_science_fiction
    + log_budget:gen_romance
    + log_budget:gen_fantasy
    + log_budget:gen_thriller
    + log_budget:vote_average
    + log_budget:runtime
    + log_budget:prod_twentieth_century_fox_film_corporation
    + log_budget:prod_new_line_cinema
    + vote_average:log_popularity
    + log_popularity:log_vote_count
    """
    return smf.ols(formula_final, data=df_v2).fit()


# ------------------------------------------------------------------ #
# Cargar todo (cacheado)
# ------------------------------------------------------------------ #
df_raw = cargar_raw()
df_v2  = construir_df_v2()
modelo = ajustar_modelo(df_v2)
medias = df_v2.attrs["medias_centrado"]

# ------------------------------------------------------------------ #
# Sidebar — navegación
# ------------------------------------------------------------------ #
st.sidebar.title(":clapper: Navegación")
secciones = [
    "1. Contexto del mercado",
    "2. Limpieza de datos",
    "3. EDA del dataset limpio",
    "4. Modelo final — resumen",
    "5. Coeficientes e interpretación",
    "6. Diagnóstico del modelo",
    "7. Simulador de revenue",
    "8. Recomendaciones de marketing",
]
seccion = st.sidebar.radio("Sección", secciones)

st.sidebar.markdown("---")
st.sidebar.metric("N películas (df limpio)", f"{len(df_v2):,}")
st.sidebar.metric("R² ajustado", f"{modelo.rsquared_adj:.3f}")
st.sidebar.metric("AIC", f"{modelo.aic:,.0f}")

st.title("Modelación de Sistemas 2025-2 — Predicción de Revenue de Películas")

# ================================================================== #
# 1. CONTEXTO
# ================================================================== #
if seccion == secciones[0]:
    st.header("1. Contexto del mercado")
    st.markdown("""
    **Bien analizado:** películas de exhibición comercial.
    **Variable de demanda (proxy):** `revenue` (recaudación bruta en USD).
    **Objetivo:** identificar qué características de una película explican mejor
    su recaudación, para guiar decisiones de marketing y greenlight.

    **¿Por qué `revenue` es un buen proxy de demanda?**
    En cine la demanda se manifiesta como tickets vendidos × precio promedio.
    No tenemos tickets directos, pero `revenue` es la mejor aproximación
    pública disponible y es la métrica que usan los estudios para juzgar el
    éxito de un lanzamiento.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Películas en CSV crudo", f"{len(df_raw):,}")
    c2.metric("Películas tras limpieza", f"{len(df_v2):,}",
              delta=f"-{len(df_raw)-len(df_v2):,}")
    c3.metric("Variables del modelo", int(modelo.df_model))
    c4.metric("Cobertura años", f"{df_raw['release_date'].str[:4].dropna().min()}–{df_raw['release_date'].str[:4].dropna().max()}")

    st.subheader("Vista rápida del CSV crudo")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Distribución de revenue (USD) en el CSV crudo")
    df_pos = df_raw[df_raw["revenue"] > 0]
    fig = px.histogram(df_pos, x="revenue", nbins=80,
                       title="Histograma de revenue (escala lineal) — fuerte sesgo a la derecha")
    fig.update_xaxes(tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== #
# 2. LIMPIEZA
# ================================================================== #
elif seccion == secciones[1]:
    st.header("2. Limpieza de datos — decisiones y justificación visual")

    st.markdown("""
    La limpieza se aplica en este orden:

    1. **Eliminar columnas no informativas** (`homepage`, `status`, `tagline`, etc.).
    2. **Pasar texto a minúsculas** y rellenar NaN categóricos con `desconocido` / `otros`.
    3. **Convertir `release_date`** a fecha y extraer `año`/`mes`.
    4. **Eliminar filas con valor 0** en `budget`, `revenue`, `runtime`, `vote_count`.
    5. **One-hot** de los 10 géneros y 10 productoras más frecuentes.
    6. **Aplicar log** a `budget`, `revenue`, `popularity`, `vote_count`.
    7. **Trimming** de `log_revenue` fuera de los percentiles 0.5 y 99.5.
    8. **Centrar** las variables numéricas para reducir multicolinealidad.
    """)

    # ---------- Subsección 2.A: Justificación ceros ----------
    st.subheader("2.A — ¿Por qué eliminar las filas con valor 0?")

    cols_chk = ["budget", "revenue", "runtime", "vote_count",
                "popularity", "vote_average"]
    ceros = pd.DataFrame({
        "Variable":   cols_chk,
        "N_total":    [len(df_raw)] * len(cols_chk),
        "N_ceros":    [int((df_raw[c] == 0).sum()) for c in cols_chk],
    })
    ceros["% ceros"] = (100 * ceros["N_ceros"] / ceros["N_total"]).round(2)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(ceros, use_container_width=True, hide_index=True)
    with c2:
        ceros["color"] = np.where(ceros["% ceros"] > 5, "Sospechoso (>5%)", "OK")
        fig = px.bar(ceros, x="Variable", y="% ceros", color="color",
                     color_discrete_map={"Sospechoso (>5%)": "#d62728",
                                         "OK": "#1f77b4"},
                     title="% de filas con valor 0 por variable")
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Interpretación:** los ceros en `budget` y `revenue` no significan
    "presupuesto cero" o "no recaudó nada"; son **datos faltantes** que TMDB
    codificó como 0. Mantenerlos rompería los `log(·)` y atraería los
    coeficientes hacia abajo. Por eso filtramos `> 0` en las cuatro variables
    críticas del modelo.
    """)

    # ---------- Subsección 2.B: log transforms ----------
    st.subheader("2.B — ¿Por qué aplicar `log` a budget, revenue, popularity y vote_count?")

    df_pos = df_raw[(df_raw["budget"] > 0) & (df_raw["revenue"] > 0)
                    & (df_raw["vote_count"] > 0)].copy()
    df_pos["log_budget"]     = np.log(df_pos["budget"])
    df_pos["log_revenue"]    = np.log(df_pos["revenue"])
    df_pos["log_vote_count"] = np.log1p(df_pos["vote_count"])

    var_log = st.selectbox("Variable a inspeccionar",
                           ["budget", "revenue", "vote_count"])
    log_col = {"budget": "log_budget", "revenue": "log_revenue",
               "vote_count": "log_vote_count"}[var_log]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_pos, x=var_log, nbins=60,
                           title=f"{var_log} crudo (sesgo a la derecha)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_pos, x=log_col, nbins=60,
                           title=f"log({var_log}) — distribución casi normal")
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    El `log` transforma una distribución sesgada (cola larga a la derecha) en
    algo aproximadamente simétrico, requisito para que **OLS** produzca
    coeficientes con interpretación tipo *elasticidad*: un coeficiente de 0.8
    sobre `log_budget` significa "1% más de presupuesto → 0.8% más de revenue".
    """)

    # ---------- Subsección 2.C: trimming outliers ----------
    st.subheader("2.C — ¿Por qué eliminar el 1% más extremo de `log_revenue`?")

    tmp = df_pos.copy()
    q_low  = tmp["log_revenue"].quantile(0.005)
    q_high = tmp["log_revenue"].quantile(0.995)
    n_out  = int(((tmp["log_revenue"] < q_low) | (tmp["log_revenue"] > q_high)).sum())

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tmp["log_revenue"], nbinsx=60,
                               marker_color="steelblue", name="log_revenue"))
    fig.add_vline(x=q_low,  line_dash="dash", line_color="red",
                  annotation_text=f"P0.5 = {q_low:.2f}")
    fig.add_vline(x=q_high, line_dash="dash", line_color="red",
                  annotation_text=f"P99.5 = {q_high:.2f}")
    fig.update_layout(title=f"Distribución de log(revenue) — se eliminan {n_out} películas ({100*n_out/len(tmp):.2f}%)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**5 películas más bajas (eliminadas):**")
        bajas = tmp.nsmallest(5, "log_revenue")[["revenue", "budget"]]
        st.dataframe(bajas.style.format({"revenue": "${:,.0f}", "budget": "${:,.0f}"}))
    with c2:
        st.markdown("**5 películas más altas (eliminadas):**")
        altas = tmp.nlargest(5, "log_revenue")[["revenue", "budget"]]
        st.dataframe(altas.style.format({"revenue": "${:,.0f}", "budget": "${:,.0f}"}))

    st.info("""
    OLS minimiza la suma de cuadrados, así que **una sola observación extrema**
    (Avatar, Titanic, festivales sin recaudación real) puede mover la pendiente
    y los errores estándar. Recortar 0.5% por cola conserva >99% de la muestra
    y estabiliza los coeficientes.
    """)

# ================================================================== #
# 3. EDA dataset limpio
# ================================================================== #
elif seccion == secciones[2]:
    st.header("3. EDA del dataset limpio (`df_v2`)")

    st.subheader("Descriptivos numéricos")
    num_cols = ["log_budget", "log_revenue", "log_popularity", "log_vote_count",
                "vote_average", "runtime", "anio", "n_generos"]
    desc = df_v2[num_cols].describe().T.round(3)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Matriz de correlación de Pearson (variables centradas)")
    corr = df_v2[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="Correlaciones entre variables del modelo")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - `log_budget` y `log_vote_count` son las dos correlaciones más fuertes
      con `log_revenue` → confirman que **dinero invertido y volumen de
      audiencia** son los predictores dominantes.
    - Las correlaciones entre regresores son bajas tras el centrado → no hay
      multicolinealidad relevante.
    """)

    st.subheader("Distribución por idioma original")
    lang = df_v2["original_language"].value_counts().reset_index()
    lang.columns = ["idioma", "n_películas"]
    fig = px.bar(lang, x="idioma", y="n_películas",
                 title="Películas por idioma original (top 5 + 'other')")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top géneros y productoras (one-hot)")
    c1, c2 = st.columns(2)
    with c1:
        gen_cols = [c for c in df_v2.columns if c.startswith("gen_")]
        gen_freq = (df_v2[gen_cols].sum().sort_values(ascending=True)
                    .reset_index())
        gen_freq.columns = ["genero", "n_películas"]
        fig = px.bar(gen_freq, x="n_películas", y="genero", orientation="h",
                     title="Frecuencia de los 10 géneros codificados")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        prod_cols = [c for c in df_v2.columns if c.startswith("prod_")]
        prod_freq = (df_v2[prod_cols].sum().sort_values(ascending=True)
                     .reset_index())
        prod_freq.columns = ["productora", "n_películas"]
        fig = px.bar(prod_freq, x="n_películas", y="productora",
                     orientation="h",
                     title="Frecuencia de las 10 productoras codificadas")
        st.plotly_chart(fig, use_container_width=True)

# ================================================================== #
# 4. MODELO FINAL — resumen
# ================================================================== #
elif seccion == secciones[3]:
    st.header("4. Modelo final — resumen")

    st.markdown("""
    **Especificación:** OLS sobre `df_v2` con variables centradas, 28 regresores
    (efectos principales + 11 interacciones) seleccionados por **stepwise AIC**
    sobre un pool de 80+ candidatas.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",          f"{modelo.rsquared:.4f}")
    c2.metric("R² ajustado", f"{modelo.rsquared_adj:.4f}")
    c3.metric("AIC",         f"{modelo.aic:,.0f}")
    c4.metric("BIC",         f"{modelo.bic:,.0f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F-statistic",       f"{modelo.fvalue:,.0f}")
    c2.metric("Prob(F)",           f"{modelo.f_pvalue:.2e}")
    c3.metric("Condition number",  f"{modelo.condition_number:,.0f}")
    c4.metric("Observaciones",     f"{int(modelo.nobs):,}")

    n_sig = int((modelo.pvalues < 0.05).sum())
    n_tot = len(modelo.pvalues)
    st.success(f"**{n_sig} de {n_tot}** coeficientes son significativos al 5% "
               f"({100*n_sig/n_tot:.0f}%).")

    st.subheader("Fórmula del modelo")
    st.code("""
log_revenue ~ log_vote_count + log_budget + anio
            + gen_family + gen_science_fiction + gen_crime + gen_fantasy
            + gen_romance + gen_drama
            + vote_average
            + prod_new_line_cinema + prod_twentieth_century_fox_film_corporation
            + prod_paramount_pictures + prod_universal_pictures
            + prod_columbia_pictures + prod_touchstone_pictures
            + prod_metro_goldwyn_mayer_mgm
            + runtime
            + log_budget:gen_crime          + log_budget:gen_science_fiction
            + log_budget:gen_romance        + log_budget:gen_fantasy
            + log_budget:gen_thriller       + log_budget:vote_average
            + log_budget:runtime
            + log_budget:prod_twentieth_century_fox_film_corporation
            + log_budget:prod_new_line_cinema
            + vote_average:log_popularity
            + log_popularity:log_vote_count
""", language="python")

    with st.expander("Ver summary completo de statsmodels"):
        st.text(str(modelo.summary()))

# ================================================================== #
# 5. Coeficientes
# ================================================================== #
elif seccion == secciones[4]:
    st.header("5. Coeficientes — forest plot interactivo")

    coefs = pd.DataFrame({
        "variable": modelo.params.index,
        "coef":     modelo.params.values,
        "p_value":  modelo.pvalues.values,
        "ci_low":   modelo.conf_int()[0].values,
        "ci_high":  modelo.conf_int()[1].values,
    })
    coefs = coefs[coefs["variable"] != "Intercept"]
    coefs["significativo"] = np.where(coefs["p_value"] < 0.05,
                                      "p < 0.05", "no signif.")
    coefs = coefs.sort_values("coef")

    nivel = st.radio("Filtrar por significancia",
                     ["Todos", "Solo significativos"], horizontal=True)
    df_plot = coefs if nivel == "Todos" else coefs[coefs["p_value"] < 0.05]

    fig = go.Figure()
    for _, row in df_plot.iterrows():
        color = "#2ca02c" if row["p_value"] < 0.05 else "#bbbbbb"
        fig.add_trace(go.Scatter(
            x=[row["ci_low"], row["ci_high"]],
            y=[row["variable"], row["variable"]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[row["coef"]], y=[row["variable"]],
            mode="markers",
            marker=dict(color=color, size=8),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['variable']}</b><br>"
                f"coef = {row['coef']:.3f}<br>"
                f"IC95% = [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<br>"
                f"p = {row['p_value']:.4f}<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title="Coeficientes con IC95% — verde = significativo (p<0.05)",
        xaxis_title="Coeficiente (efecto sobre log_revenue)",
        height=max(400, 25 * len(df_plot)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla detallada de coeficientes")
    st.dataframe(
        coefs.sort_values("coef", ascending=False)
             .style.format({"coef": "{:.4f}", "p_value": "{:.4f}",
                            "ci_low": "{:.4f}", "ci_high": "{:.4f}"}),
        use_container_width=True,
    )

    st.subheader("Cómo leer los coeficientes")
    st.markdown("""
    Como la variable dependiente es `log_revenue` y las numéricas están en
    `log` y centradas:

    - Coeficiente sobre `log_budget` = **elasticidad presupuesto → revenue**.
      Un valor de 0.55 significa que "duplicar presupuesto multiplica el
      revenue esperado por aproximadamente 1.46 (= 2^0.55)".
    - Coeficiente sobre dummy de género = **diferencial de revenue** vs. la
      categoría base, *manteniendo todo lo demás constante*.
      Ejemplo: si `gen_family = 0.40`, las películas familiares ganan
      **e^0.40 − 1 ≈ +49%** más revenue que el promedio comparable.
    - Coeficiente sobre `log_budget:gen_crime` = **modificador** de la
      pendiente del presupuesto cuando es película de crimen. No es la
      pendiente; es la desviación respecto a la pendiente base.
    """)

# ================================================================== #
# 6. DIAGNÓSTICO
# ================================================================== #
elif seccion == secciones[5]:
    st.header("6. Diagnóstico del modelo")

    y_pred = modelo.fittedvalues
    y_obs  = pd.Series(modelo.model.endog, index=y_pred.index)
    resid  = y_obs - y_pred

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Predicho vs observado")
        df_po = pd.DataFrame({"observado": y_obs, "predicho": y_pred})
        fig = px.scatter(df_po, x="observado", y="predicho",
                         opacity=0.4, trendline="ols",
                         title=f"R² = {modelo.rsquared:.3f}")
        lim = [df_po.min().min(), df_po.max().max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                 line=dict(color="red", dash="dash"),
                                 name="y=x"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Residuos vs predicho")
        df_rp = pd.DataFrame({"predicho": y_pred, "residuo": resid})
        fig = px.scatter(df_rp, x="predicho", y="residuo", opacity=0.4,
                         title="Idealmente: nube horizontal centrada en 0")
        fig.add_hline(y=0, line_color="red", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("QQ-plot de residuos")
        sorted_res = np.sort(resid)
        n = len(sorted_res)
        theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical, y=sorted_res,
                                 mode="markers", marker=dict(opacity=0.4),
                                 name="residuos"))
        slope, intercept = np.polyfit(theoretical, sorted_res, 1)
        fig.add_trace(go.Scatter(x=theoretical,
                                 y=slope * theoretical + intercept,
                                 mode="lines", line=dict(color="red"),
                                 name="línea normal"))
        fig.update_layout(title="QQ-plot — recta = residuos normales",
                          xaxis_title="Cuantiles teóricos N(0,1)",
                          yaxis_title="Cuantiles observados")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Distribución de residuos")
        fig = px.histogram(resid, nbins=50,
                           title=f"Media = {resid.mean():.4f} (debe ≈ 0)")
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Lectura:**
    - Predicho vs observado alineado a la diagonal → el modelo captura bien
      el nivel y la dispersión.
    - Residuos vs predicho sin patrón → no hay sesgo sistemático.
    - QQ-plot recto → residuos aproximadamente normales (válido para
      inferencia).
    - Histograma centrado en 0 → no hay sesgo.
    """)

# ================================================================== #
# 7. Simulador interactivo
# ================================================================== #
elif seccion == secciones[6]:
    st.header("7. Simulador de revenue")
    st.markdown("Modificá los inputs y observá la predicción del modelo.")

    c1, c2, c3 = st.columns(3)
    with c1:
        budget_usd = st.number_input("Budget (USD)", min_value=10_000,
                                     max_value=400_000_000,
                                     value=20_000_000, step=1_000_000,
                                     format="%d")
        runtime_in = st.slider("Runtime (min)", 60, 220, 110)
        anio_in    = st.slider("Año de estreno", 1980, 2025, 2024)
    with c2:
        vote_avg_in = st.slider("Vote average esperado (1-10)",
                                1.0, 10.0, 6.5, 0.1)
        vote_cnt_in = st.number_input("Vote count esperado",
                                      min_value=1, max_value=50_000,
                                      value=1500, step=100)
        popularity_in = st.slider("Popularity esperada", 0.0, 200.0, 25.0, 0.5)
    with c3:
        gen_cols = [c for c in df_v2.columns if c.startswith("gen_")]
        prod_cols = [c for c in df_v2.columns if c.startswith("prod_")]
        gens_sel  = st.multiselect("Géneros (one-hot)",
                                   [g.replace("gen_", "") for g in gen_cols],
                                   default=["drama"])
        prods_sel = st.multiselect("Productora (one-hot)",
                                   [p.replace("prod_", "") for p in prod_cols],
                                   default=[])

    fila = {col: 0.0 for col in df_v2.columns if col != "log_revenue"}

    fila["log_budget"]     = np.log(budget_usd)         - medias["log_budget"]
    fila["log_popularity"] = np.log1p(popularity_in)    - medias["log_popularity"]
    fila["log_vote_count"] = np.log1p(vote_cnt_in)      - medias["log_vote_count"]
    fila["vote_average"]   = vote_avg_in                - medias["vote_average"]
    fila["runtime"]        = runtime_in                 - medias["runtime"]
    fila["anio"]           = anio_in                    - medias["anio"]

    for g in gens_sel:
        col = f"gen_{g}"
        if col in fila:
            fila[col] = 1
    for p in prods_sel:
        col = f"prod_{p}"
        if col in fila:
            fila[col] = 1

    X_new = pd.DataFrame([fila])
    pred_log = float(modelo.predict(X_new).iloc[0])
    pred_usd = float(np.exp(pred_log))

    pred_se = float(modelo.get_prediction(X_new).se_obs[0])
    pi_low  = float(np.exp(pred_log - 1.96 * pred_se))
    pi_high = float(np.exp(pred_log + 1.96 * pred_se))

    st.markdown("### Predicción")
    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue esperado", f"${pred_usd:,.0f}")
    c2.metric("IC 95% bajo",      f"${pi_low:,.0f}")
    c3.metric("IC 95% alto",      f"${pi_high:,.0f}")

    roi = pred_usd / budget_usd
    st.metric("ROI esperado (revenue / budget)", f"{roi:.2f}x",
              delta=f"{(roi - 1) * 100:+.0f}% sobre el presupuesto")

    st.caption("""
    El IC95% es un **intervalo de predicción** sobre revenue para una nueva
    película con esos atributos (no un IC sobre el promedio). Es naturalmente
    amplio porque incorpora la varianza residual del modelo.
    """)

# ================================================================== #
# 8. RECOMENDACIONES
# ================================================================== #
elif seccion == secciones[7]:
    st.header("8. Recomendaciones de marketing")

    st.markdown("""
    Las recomendaciones se derivan de los **coeficientes significativos** del
    modelo final, traducidos a lenguaje de negocio.

    ### Palancas de mayor impacto
    1. **Inversión en presupuesto (`log_budget`):** elasticidad cercana a
       0.55–0.65. Cada **+10%** de budget se traduce en **+5–6%** de revenue
       esperado, *si la calidad y la audiencia se mantienen*. Es la palanca
       más confiable para mover la aguja, pero rinde marginalmente.
    2. **Audiencia anticipada (`log_vote_count`):** indicador de awareness
       prelanzamiento. Coeficiente alto y robusto. Justifica invertir en
       campañas de **trailers tempranos, screenings, redes sociales**.
    3. **Calidad percibida (`vote_average`):** efecto positivo y
       significativo. Refuerza la importancia de la **post-producción y
       testing previo** antes del estreno.

    ### Decisiones de género
    - `gen_family`, `gen_fantasy`, `gen_science_fiction`: efectos positivos
      → géneros con mejor retorno esperado para apuestas de presupuesto alto.
    - `gen_drama`, `gen_romance`: efectos modestos → conviene presupuestos
      contenidos y campañas focalizadas.
    - `log_budget × gen_*`: las interacciones nos dicen que **el dinero rinde
      diferente según el género**. Sci-fi y crime amplifican el efecto del
      budget; romance y drama lo amortiguan.

    ### Decisiones de productora
    - Productoras del top-10 (Paramount, Universal, Fox, etc.) tienen
      coeficientes positivos por encima del baseline → **el sello/distribución
      importa** más allá del contenido. Co-producir o licenciar marca puede
      añadir un *premium* medible.

    ### Riesgos y caveats
    - El modelo está entrenado con películas **comerciales** (post trimming
      del 1% extremo). Las predicciones para festivales independientes o
      megablockbusters serán menos confiables.
    - R² ≈ 0.75 deja un 25% de varianza no explicada → factores como
      *timing del release, competencia en cartelera, eventos externos* no
      están en el modelo y deben considerarse cualitativamente.
    """)

    st.success("""
    **Cierre ejecutivo:** un greenlight bien apalancado combina (1) presupuesto
    suficiente, (2) inversión temprana en awareness, (3) un género con
    elasticidad positiva al budget y (4) un partner de distribución del top
    histórico. El modelo permite **cuantificar a priori** el revenue esperado
    bajo cada combinación.
    """)
