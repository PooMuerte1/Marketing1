"""
Dashboard Ejecutivo — ¿Cuánto va a recaudar mi película?
Versión orientada a empresario / decisor de marketing.

Cómo ejecutar:
    python -m streamlit run app_ejecutivo.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="¿Cuánto recauda mi película?",
    page_icon=":clapper:",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Pipeline (idéntica al notebook, oculta para el usuario)
# ------------------------------------------------------------------ #
def _limpiar(s: str) -> str:
    out = s
    for ch in [" ", ".", ",", "-", "(", ")", "/", "'", "&"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


@st.cache_data(show_spinner="Cargando histórico de películas...")
def construir_df_v2() -> pd.DataFrame:
    df_v2 = pd.read_csv("data__movies.csv")
    df_v2 = df_v2.drop(columns=["homepage", "status", "tagline",
                                "original_title", "overview", "id"])

    df_v2["production_companies"] = df_v2["production_companies"].str.split(",").str[0]
    df_v2 = df_v2.map(lambda x: x.lower() if isinstance(x, str) else x)
    df_v2["title"] = df_v2["title"].fillna("(sin título)")

    df_v2["genres"] = df_v2["genres"].fillna("desconocido")
    df_v2["production_companies"] = df_v2["production_companies"].fillna("otros")
    df_v2 = df_v2.dropna(subset=["release_date", "runtime"])
    df_v2["release_date"] = pd.to_datetime(df_v2["release_date"], errors="coerce")
    df_v2 = df_v2.dropna(subset=["release_date"])
    df_v2["anio"] = df_v2["release_date"].dt.year.astype(int)
    df_v2["mes"]  = df_v2["release_date"].dt.month.astype(int)
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
        df_v2["original_language"].isin(top_lang), "other")

    df_v2_full = df_v2.copy()  # captura ANTES de logs (valores originales en USD)

    df_v2["budget"]     = np.log(df_v2["budget"])
    df_v2["revenue"]    = np.log(df_v2["revenue"])
    df_v2["popularity"] = np.log1p(df_v2["popularity"])
    df_v2["vote_count"] = np.log1p(df_v2["vote_count"])

    df_v2 = df_v2.drop(columns=["genres", "production_companies", "title"])

    q_low  = df_v2["revenue"].quantile(0.005)
    q_high = df_v2["revenue"].quantile(0.995)
    keep_idx = (df_v2["revenue"] >= q_low) & (df_v2["revenue"] <= q_high)
    df_v2 = df_v2[keep_idx].copy()
    df_v2_full = df_v2_full.loc[df_v2.index].copy()

    medias = {}
    for col in ["budget", "popularity", "vote_count",
                "vote_average", "runtime", "anio"]:
        medias[col] = df_v2[col].mean()
        df_v2[col] = df_v2[col] - medias[col]

    df_v2 = df_v2.reset_index(drop=True)
    df_v2_full = df_v2_full.reset_index(drop=True)

    df_v2.attrs["medias_centrado"] = medias
    df_v2.attrs["full"] = df_v2_full
    return df_v2


FORMULA_FINAL = """
revenue ~ vote_count + budget + anio
+ gen_family + gen_science_fiction + gen_crime + gen_fantasy + gen_romance + gen_drama
+ vote_average
+ prod_new_line_cinema + prod_twentieth_century_fox_film_corporation
+ prod_paramount_pictures + prod_universal_pictures + prod_columbia_pictures
+ prod_miramax_films + prod_village_roadshow_pictures
+ runtime
+ budget:gen_crime
+ budget:gen_science_fiction
+ budget:gen_romance
+ budget:gen_fantasy
+ budget:gen_thriller
+ budget:vote_average
+ budget:runtime
+ budget:prod_twentieth_century_fox_film_corporation
+ budget:prod_new_line_cinema
+ vote_average:popularity
+ popularity:vote_count
""".strip()


@st.cache_resource(show_spinner="Calibrando modelo de predicción...")
def ajustar_modelo(df_v2: pd.DataFrame):
    modelo_inicial = smf.ols(FORMULA_FINAL, data=df_v2).fit(cov_type="HC3")
    residuos_estudentizados = modelo_inicial.get_influence().resid_studentized_internal
    mask_limpio = np.abs(residuos_estudentizados) <= 2.0
    df_limpio = df_v2[mask_limpio].copy()

    modelo_corregido = smf.ols(FORMULA_FINAL, data=df_limpio).fit()
    return modelo_corregido, modelo_inicial, df_limpio


# ------------------------------------------------------------------ #
# Cargar todo
# ------------------------------------------------------------------ #
df_v2  = construir_df_v2()
df_full = df_v2.attrs["full"]
modelo, modelo_inicial, df_limpio = ajustar_modelo(df_v2)
medias = df_v2.attrs["medias_centrado"]
df_full_limpio = df_full.loc[df_limpio.index].copy()

# Pre-cálculo de "premium por género" y "premium por productora"
# en términos de % cambio de revenue (interpretable para empresario)
def premium_pct(coef: float) -> float:
    return (np.exp(coef) - 1) * 100


coef_dict = modelo.params.to_dict()
pval_dict = modelo.pvalues.to_dict()

generos_legibles = {
    "gen_drama":           "Drama",
    "gen_comedy":          "Comedy",
    "gen_thriller":        "Thriller",
    "gen_action":          "Action",
    "gen_romance":         "Romance",
    "gen_adventure":       "Adventure",
    "gen_crime":           "Crime",
    "gen_science_fiction": "Sci-Fi",
    "gen_horror":          "Horror",
    "gen_family":          "Family",
    "gen_fantasy":         "Fantasy",
    "gen_animation":       "Animation",
    "gen_mystery":         "Mystery",
}
productoras_legibles = {
    "prod_paramount_pictures":                    "Paramount Pictures",
    "prod_universal_pictures":                    "Universal Pictures",
    "prod_twentieth_century_fox_film_corporation":"20th Century Fox",
    "prod_columbia_pictures":                     "Columbia Pictures",
    "prod_new_line_cinema":                       "New Line Cinema",
    "prod_miramax_films":                         "Miramax Films",
    "prod_village_roadshow_pictures":             "Village Roadshow Pictures",
    "prod_warner_bros":                           "Warner Bros.",
    "prod_walt_disney_pictures":                  "Walt Disney",
    "prod_dreamworks_skg":                        "DreamWorks",
}

# ------------------------------------------------------------------ #
# Sidebar = SIMULADOR (es el corazón del dashboard ejecutivo)
# ------------------------------------------------------------------ #
st.sidebar.title(":dart: Tu próxima película")
st.sidebar.caption("Configura los parámetros y observa la predicción en vivo.")

budget_usd = st.sidebar.number_input(
    "Presupuesto (USD)", min_value=100_000, max_value=400_000_000,
    value=20_000_000, step=1_000_000, format="%d")

st.sidebar.markdown("**Calidad esperada**")
vote_avg_in = st.sidebar.slider("Calificación esperada (1-10)",
                                1.0, 10.0, 6.5, 0.1)

st.sidebar.markdown("**Tamaño de audiencia esperado** (indicadores, no palancas directas)")
popularity_in = st.sidebar.slider(
    "Engagement esperado de la audiencia",
    0.0, 200.0, 25.0, 0.5,
    help=("Score TMDB que mide búsquedas, vistas e interacciones del público "
          "con la película. Es un INDICADOR del nivel de interés que se va a "
          "generar, no una palanca que se mueve sola: crece con inversión "
          "en distribución, plataformas y publicidad."))
vote_cnt_in = st.sidebar.slider(
    "Tamaño de audiencia que reseñará",
    100, 30_000, 1_500, 100,
    help=("Cantidad de personas que terminan calificando la película en TMDB. "
          "Es un INDICADOR del alcance que se va a lograr; no es una variable "
          "que se controle directamente. La palanca real para subirlo es la "
          "inversión en distribución y marketing."))

st.sidebar.markdown("**Producto**")
runtime_in = st.sidebar.slider("Duración (min)", 60, 220, 110)
anio_in    = st.sidebar.slider("Año de estreno", 1980, 2025, 2024)

gen_cols  = [c for c in df_v2.columns if c.startswith("gen_")]
prod_cols = [c for c in df_v2.columns
             if c.startswith("prod_") and c in coef_dict]

gens_disponibles  = [generos_legibles.get(c, c.replace("gen_", "").title())
                     for c in gen_cols]
prods_disponibles = ["Sin sello mayor / otra"] + \
                    [productoras_legibles.get(c, c.replace("prod_", "").title())
                     for c in prod_cols]

gens_sel  = st.sidebar.multiselect("Género(s)", gens_disponibles,
                                   default=["Drama"])
prod_sel  = st.sidebar.selectbox("Productora / sello", prods_disponibles)

# ------------------------------------------------------------------ #
# Predicción (acepta overrides opcionales sobre la config del sidebar)
# ------------------------------------------------------------------ #
def predecir(budget=None, vote_avg=None, popularity=None, vote_cnt=None,
             runtime=None, anio=None, gens_label=None, prod_label=None):
    budget     = budget     if budget     is not None else budget_usd
    vote_avg   = vote_avg   if vote_avg   is not None else vote_avg_in
    popularity = popularity if popularity is not None else popularity_in
    vote_cnt   = vote_cnt   if vote_cnt   is not None else vote_cnt_in
    runtime    = runtime    if runtime    is not None else runtime_in
    anio       = anio       if anio       is not None else anio_in
    gens_label = gens_label if gens_label is not None else gens_sel
    prod_label = prod_label if prod_label is not None else prod_sel

    fila = {col: 0.0 for col in df_v2.columns if col != "revenue"}
    fila["budget"]       = np.log(budget)         - medias["budget"]
    fila["popularity"]   = np.log1p(popularity)   - medias["popularity"]
    fila["vote_count"]   = np.log1p(vote_cnt)     - medias["vote_count"]
    fila["vote_average"] = vote_avg               - medias["vote_average"]
    fila["runtime"]      = runtime                - medias["runtime"]
    fila["anio"]         = anio                   - medias["anio"]

    label_to_col_g = {v: k for k, v in generos_legibles.items()}
    for g in gens_label:
        col = label_to_col_g.get(g)
        if col and col in fila:
            fila[col] = 1
    label_to_col_p = {v: k for k, v in productoras_legibles.items()}
    if prod_label and prod_label != "Sin sello mayor / otra":
        col = label_to_col_p.get(prod_label)
        if col and col in fila:
            fila[col] = 1

    X_new = pd.DataFrame([fila])
    pred_log = float(modelo.predict(X_new).iloc[0])
    se_obs   = float(modelo.get_prediction(X_new).se_obs[0])
    return {
        "revenue":  float(np.exp(pred_log)),
        "low":      float(np.exp(pred_log - 1.96 * se_obs)),
        "high":     float(np.exp(pred_log + 1.96 * se_obs)),
        "log_pred": pred_log,
    }


pred = predecir()  # sin overrides → usa la config actual del sidebar

# Coeficientes que se usan en varias tabs
elast_budget = coef_dict.get("budget",       0)
coef_va      = coef_dict.get("vote_average", 0)
coef_vc      = coef_dict.get("vote_count",   0)
coef_pop     = coef_dict.get("popularity",   0)

roi = pred["revenue"] / budget_usd
prob_break_even = 100 * float(
    1 - 0.5 * (1 + math.erf(
        (np.log(budget_usd) - pred["log_pred"]) /
        (np.sqrt(2) * float(modelo.scale ** 0.5))
    ))
)

# ------------------------------------------------------------------ #
# HEADER ejecutivo
# ------------------------------------------------------------------ #
st.title(":clapper: ¿Cuánto va a recaudar tu próxima película?")
st.markdown("### Predicción para tu película")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recaudación esperada", f"${pred['revenue']/1e6:,.1f} M USD")
c2.metric("Presupuesto",          f"${budget_usd/1e6:,.1f} M USD")
c3.metric("ROI esperado", f"{roi:.2f}x",
          delta=f"{(roi - 1) * 100:+.0f}% sobre el budget",
          delta_color="normal" if roi >= 1 else "inverse")
c4.metric("Probabilidad de recuperar la inversión",
          f"{prob_break_even:.0f}%")

# Semáforo ejecutivo
if roi >= 2.5 and prob_break_even >= 75:
    st.success("**Apuesta sólida.** El modelo predice un retorno claramente "
               "positivo y alta probabilidad de recuperar la inversión.")
elif roi >= 1.2 and prob_break_even >= 55:
    st.warning("**Apuesta razonable, pero ajustada.** El retorno esperado es "
               "positivo pero el margen es estrecho. Conviene reforzar las "
               "palancas de marketing antes de greenlight.")
else:
    st.error("**Riesgo elevado.** El modelo anticipa un ROI bajo o probabilidad "
             "insuficiente de recuperar la inversión. Revisar presupuesto, "
             "género o estrategia de awareness.")

st.caption(f"Rango plausible (intervalo 95%): "
           f"**\\${pred['low']/1e6:,.1f}M – \\${pred['high']/1e6:,.1f}M USD**. "
           f"El rango es amplio porque el cine es un negocio inherentemente "
           f"volátil; el modelo refleja esa incertidumbre.")

st.divider()

# ------------------------------------------------------------------ #
# Tabs ejecutivas
# ------------------------------------------------------------------ #
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    ":bulb: Palancas",
    ":movie_camera: Géneros",
    ":office: Sellos",
    ":memo: Recomendaciones",
    ":mag: Películas similares",
    ":world_map: Mapa de decisión",
    ":tornado: Sensibilidad",
    ":scales: Comparador A/B",
    ":bar_chart: Posicionamiento histórico",
    ":books: Anexo: Modelo",
])

# ------------------ TAB 1: Palancas ------------------ #
with tab1:
    st.subheader("Curva de retorno por presupuesto")
    st.markdown(
        "Mostramos cómo cambia la **recaudación esperada** y el **ROI** "
        "a medida que mueve el presupuesto, manteniendo el resto de la "
        "configuración igual (calidad, género, sello, duración). La banda "
        "azul es el rango plausible (95%) para una película individual y "
        "la línea negra punteada marca el break-even (revenue = budget)."
    )

    budgets_range = np.logspace(
        np.log10(500_000), np.log10(300_000_000), 60
    )
    preds_curva = [predecir(budget=float(b)) for b in budgets_range]
    df_curva = pd.DataFrame({
        "Presupuesto":      budgets_range,
        "Revenue esperado": [p["revenue"] for p in preds_curva],
        "Low IC95":         [p["low"]     for p in preds_curva],
        "High IC95":        [p["high"]    for p in preds_curva],
    })
    df_curva["ROI"] = df_curva["Revenue esperado"] / df_curva["Presupuesto"]

    col_l, col_r = st.columns(2)

    with col_l:
        fig_curva = go.Figure()
        fig_curva.add_trace(go.Scatter(
            x=df_curva["Presupuesto"], y=df_curva["High IC95"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_curva.add_trace(go.Scatter(
            x=df_curva["Presupuesto"], y=df_curva["Low IC95"],
            fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0), name="Rango plausible 95%",
            hoverinfo="skip"))
        fig_curva.add_trace(go.Scatter(
            x=df_curva["Presupuesto"], y=df_curva["Revenue esperado"],
            mode="lines", name="Revenue esperado",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Budget: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>"))
        fig_curva.add_trace(go.Scatter(
            x=df_curva["Presupuesto"], y=df_curva["Presupuesto"],
            mode="lines", name="Break-even",
            line=dict(color="black", dash="dash", width=1)))
        fig_curva.add_trace(go.Scatter(
            x=[budget_usd], y=[pred["revenue"]],
            mode="markers", name="Tu setup actual",
            marker=dict(size=14, color="red", symbol="star")))
        fig_curva.update_layout(
            xaxis_type="log", yaxis_type="log",
            xaxis_title="Presupuesto (USD)",
            yaxis_title="Revenue esperado (USD)",
            title="Revenue esperado vs presupuesto",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_curva, use_container_width=True)

    with col_r:
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=df_curva["Presupuesto"], y=df_curva["ROI"],
            mode="lines", name="ROI esperado",
            line=dict(color="#2ca02c", width=3),
            hovertemplate="Budget: $%{x:,.0f}<br>ROI: %{y:.2f}x<extra></extra>"))
        fig_roi.add_hline(y=1, line_dash="dash", line_color="black",
                          annotation_text="ROI = 1x (break-even)",
                          annotation_position="bottom right")
        fig_roi.add_trace(go.Scatter(
            x=[budget_usd], y=[roi],
            mode="markers", name="Tu setup actual",
            marker=dict(size=14, color="red", symbol="star")))
        fig_roi.update_layout(
            xaxis_type="log",
            xaxis_title="Presupuesto (USD)",
            yaxis_title="ROI esperado (revenue / budget)",
            title="Cómo cae el ROI al escalar el budget",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_roi, use_container_width=True)

    budget_optimo_idx = int(df_curva["ROI"].idxmax())
    budget_optimo = float(df_curva.loc[budget_optimo_idx, "Presupuesto"])
    roi_optimo    = float(df_curva.loc[budget_optimo_idx, "ROI"])
    st.success(
        f"**Punto de máximo ROI:** ${budget_optimo/1e6:,.1f}M de budget → "
        f"ROI esperado ~{roi_optimo:.2f}x. "
        f"Por encima de ese nivel, cada dólar adicional rinde menos."
    )

    st.divider()

    st.subheader("Palancas de decisión calculadas para TU película")
    st.markdown(f"""
    Cada palanca de abajo simula **qué pasaría con la recaudación de TU película
    actual** (${budget_usd/1e6:.1f}M de budget, calidad {vote_avg_in},
    {', '.join(gens_sel) if gens_sel else 'sin género'}) si moviéramos
    SOLO esa palanca, dejando todo lo demás igual.

    Los números **incluyen automáticamente las interacciones del modelo**:
    por ejemplo, "subir la calidad" rinde diferente según el presupuesto y el
    género elegido. Por eso los porcentajes cambian al mover el panel
    lateral.
    """)

    base_rev = pred["revenue"]
    base_log = pred["log_pred"]

    label_to_col_g = {v: k for k, v in generos_legibles.items()}
    label_to_col_p = {v: k for k, v in productoras_legibles.items()}

    # ----------------------------------------------------------------
    # Construcción dinámica de palancas
    # ----------------------------------------------------------------
    palancas = []

    def _agregar(nombre, override_kwargs, categoria, descripcion):
        nuevo = predecir(**override_kwargs)
        delta_pct = (nuevo["revenue"] / base_rev - 1) * 100
        delta_usd = nuevo["revenue"] - base_rev
        palancas.append({
            "Palanca":      nombre,
            "Δ revenue %":  delta_pct,
            "Δ revenue USD": delta_usd,
            "Tipo":         categoria,
            "Detalle":      descripcion,
            "Nuevo revenue": nuevo["revenue"],
        })

    # 1) Subir budget +20%
    _agregar("Subir presupuesto +20%",
             {"budget": budget_usd * 1.2}, "Producción",
             f"Pasar de ${budget_usd/1e6:.1f}M a ${budget_usd*1.2/1e6:.1f}M")

    # 2) Bajar budget -20%
    _agregar("Bajar presupuesto -20%",
             {"budget": budget_usd * 0.8}, "Producción",
             f"Pasar de ${budget_usd/1e6:.1f}M a ${budget_usd*0.8/1e6:.1f}M")

    # 3) Calidad +1 punto (cap 10)
    if vote_avg_in < 10:
        nueva_q = min(vote_avg_in + 1, 10)
        _agregar(f"Subir calidad: {vote_avg_in} → {nueva_q}",
                 {"vote_avg": nueva_q}, "Calidad",
                 "Inversión en post-producción, talento y test screenings")

    # 4) Calidad -1 punto (floor 1)
    if vote_avg_in > 1:
        nueva_q = max(vote_avg_in - 1, 1)
        _agregar(f"Recorte de calidad: {vote_avg_in} → {nueva_q}",
                 {"vote_avg": nueva_q}, "Calidad",
                 "Saltarse pasos de calidad para ahorrar")

    # 5) Acortar 15 min
    _agregar(f"Acortar la película (-15 min)",
             {"runtime": max(60, runtime_in - 15)}, "Producto",
             f"Editar a {max(60, runtime_in - 15)} min")

    # 6) Alargar 15 min
    _agregar(f"Alargar la película (+15 min)",
             {"runtime": min(220, runtime_in + 15)}, "Producto",
             f"Editar a {min(220, runtime_in + 15)} min")

    # 7) Asociarse con el MEJOR sello (si actualmente no tiene sello mayor)
    prods_modelo_labels = [productoras_legibles[c]
                           for c in productoras_legibles
                           if c in coef_dict]
    if prod_sel == "Sin sello mayor / otra":
        mejor_prod, mejor_delta = None, -np.inf
        for label_p in prods_modelo_labels:
            tmp = predecir(prod_label=label_p)
            d = (tmp["revenue"] / base_rev - 1) * 100
            if d > mejor_delta:
                mejor_delta, mejor_prod = d, label_p
        if mejor_prod:
            _agregar(f"Asociarte con {mejor_prod}",
                     {"prod_label": mejor_prod}, "Distribución",
                     "Co-producir o licenciar bajo el mejor sello con efecto estimado")
    else:
        mejor_alt, mejor_delta = None, -np.inf
        for label_p in prods_modelo_labels:
            if label_p == prod_sel:
                continue
            tmp = predecir(prod_label=label_p)
            d = (tmp["revenue"] / base_rev - 1) * 100
            if d > mejor_delta:
                mejor_delta, mejor_alt = d, label_p
        if mejor_alt and mejor_delta > 0.5:
            _agregar(f"Cambiar a {mejor_alt}",
                     {"prod_label": mejor_alt}, "Distribución",
                     f"En vez del sello actual ({prod_sel})")

    # 8) Sumar el género MÁS rentable que no tengas
    gens_actuales_cols = [label_to_col_g[g] for g in gens_sel
                          if g in label_to_col_g]
    mejor_gen, mejor_delta = None, -np.inf
    for label_g, col_g in label_to_col_g.items():
        if col_g in gens_actuales_cols:
            continue
        tmp = predecir(gens_label=gens_sel + [label_g])
        d = (tmp["revenue"] / base_rev - 1) * 100
        if d > mejor_delta:
            mejor_delta, mejor_gen = d, label_g
    if mejor_gen and mejor_delta > 0.5:
        _agregar(f"Sumar género: {mejor_gen}",
                 {"gens_label": gens_sel + [mejor_gen]}, "Posicionamiento",
                 "Reorientar marketing/guion para incluir este género")

    # 9) Quitar el género MENOS rentable que tengas
    if len(gens_sel) >= 2:
        peor_gen, peor_delta = None, -np.inf
        for label_g in gens_sel:
            tmp = predecir(gens_label=[g for g in gens_sel if g != label_g])
            d = (tmp["revenue"] / base_rev - 1) * 100
            if d > peor_delta:
                peor_delta, peor_gen = d, label_g
        if peor_gen and peor_delta > 0.5:
            _agregar(f"Quitar género: {peor_gen}",
                     {"gens_label": [g for g in gens_sel if g != peor_gen]},
                     "Posicionamiento",
                     "Reposicionar para no diluir la propuesta")

    # 10) Combo: budget +20% Y calidad +1
    if vote_avg_in < 10:
        nueva_q = min(vote_avg_in + 1, 10)
        _agregar(f"Combo: budget +20% + calidad +1",
                 {"budget": budget_usd * 1.2, "vote_avg": nueva_q}, "Combo",
                 "Estrategia integrada de inversión y calidad")

    df_pal = pd.DataFrame(palancas).sort_values("Δ revenue %", ascending=True)

    # ----------------------------------------------------------------
    # Visualización
    # ----------------------------------------------------------------
    color_map = {
        "Producción":      "#1f77b4",
        "Calidad":         "#ff7f0e",
        "Producto":        "#9467bd",
        "Distribución":    "#2ca02c",
        "Posicionamiento": "#e377c2",
        "Combo":           "#d62728",
    }

    df_pal["Etiqueta"] = df_pal.apply(
        lambda r: f"{r['Δ revenue %']:+.1f}% (${r['Δ revenue USD']/1e6:+,.1f}M)",
        axis=1,
    )

    fig = px.bar(df_pal, x="Δ revenue %", y="Palanca",
                 color="Tipo", orientation="h",
                 text="Etiqueta",
                 color_discrete_map=color_map,
                 hover_data={"Detalle": True, "Nuevo revenue": ":,.0f",
                             "Δ revenue USD": ":,.0f", "Tipo": False},
                 title="Cuánto cambia la recaudación de TU película según cada decisión")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="% de cambio en revenue vs. tu configuración actual",
                      height=max(420, 35 * len(df_pal)),
                      showlegend=True)
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------
    # Top recomendaciones automáticas
    # ----------------------------------------------------------------
    palancas_positivas = df_pal[df_pal["Δ revenue %"] > 0].sort_values(
        "Δ revenue %", ascending=False)

    if len(palancas_positivas) >= 1:
        top1 = palancas_positivas.iloc[0]
        st.success(
            f"**Tu mejor palanca ahora mismo: {top1['Palanca']}** → "
            f"+{top1['Δ revenue %']:.1f}% de recaudación "
            f"(+${top1['Δ revenue USD']/1e6:.1f}M USD). _{top1['Detalle']}_."
        )

    palancas_negativas = df_pal[df_pal["Δ revenue %"] < 0].sort_values(
        "Δ revenue %", ascending=True)
    if len(palancas_negativas) >= 1:
        peor = palancas_negativas.iloc[0]
        st.error(
            f"**Decisión que más te costaría: {peor['Palanca']}** → "
            f"{peor['Δ revenue %']:.1f}% de recaudación "
            f"({peor['Δ revenue USD']/1e6:.1f}M USD). _{peor['Detalle']}_."
        )

    st.divider()

    # ----------------------------------------------------------------
    # Indicadores (no palancas)
    # ----------------------------------------------------------------
    st.markdown("#### :information_source: Indicadores de éxito (no son palancas directas)")
    st.markdown("""
    Estos NO son decisiones que se toman antes de filmar. Son **señales que
    se miden después del estreno** (cantidad de gente que reseñó la película,
    popularidad acumulada en TMDB). Tienen una relación circular con el
    revenue: una película que recauda más es vista por más gente y por eso
    recibe más reseñas. La palanca real para moverlos es la **inversión en
    distribución y marketing**.
    """)

    indicadores = pd.DataFrame([
        {"Indicador": "Doblar la audiencia que reseña",
         "Asociación con revenue": premium_pct(coef_vc * np.log(2))},
        {"Indicador": "Doblar el engagement / popularidad",
         "Asociación con revenue": premium_pct(coef_pop * np.log(2))},
    ])

    fig = px.bar(indicadores.sort_values("Asociación con revenue"),
                 x="Asociación con revenue", y="Indicador",
                 orientation="h",
                 text=indicadores["Asociación con revenue"]
                                  .apply(lambda v: f"+{v:.1f}%"),
                 color_discrete_sequence=["#7f7f7f"],
                 title="Asociación promedio (no causal) entre indicadores y revenue")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="% asociado en revenue", height=250)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **Recordatorio:** estos números son **asociaciones promedio**, no
    causales. La recomendación accionable es invertir en lo que hace crecer
    estos indicadores **y** el revenue al mismo tiempo: salas, plataformas,
    países, publicidad pagada, redes sociales, prensa, eventos.
    """)

# ------------------ TAB 2: Géneros ------------------ #
with tab2:
    st.subheader("Géneros: cuáles dan más, cuáles dan menos")

    # Calculamos premium por género (efecto principal expresado como %)
    rows = []
    for col, label in generos_legibles.items():
        if col in coef_dict:
            rows.append({
                "Género":          label,
                "Premium revenue": premium_pct(coef_dict[col]),
                "Significativo":   pval_dict[col] < 0.05,
                "Películas":       int(df_v2[col].sum()) if col in df_v2.columns else 0,
            })
    df_gen = pd.DataFrame(rows).sort_values("Premium revenue")

    fig = px.bar(df_gen, x="Premium revenue", y="Género",
                 color="Significativo",
                 color_discrete_map={True: "#2ca02c", False: "#bbbbbb"},
                 orientation="h",
                 title="Premium / descuento de revenue vs. el promedio (manteniendo lo demás constante)",
                 hover_data={"Películas": True, "Significativo": False})
    fig.update_layout(xaxis_title="% sobre el revenue promedio",
                      height=420)
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)

    # ¿Cuánto rinde el dinero según el género?
    st.subheader("¿Dónde rinde más cada dólar invertido?")
    st.markdown("""
    Para cada género calculamos cuánto sube la recaudación si se aumenta el
    presupuesto un 10%. Algunos géneros **amplifican** el efecto del dinero,
    otros lo **amortiguan**.
    """)

    rows2 = []
    for col, label in generos_legibles.items():
        inter_key = f"budget:{col}"
        eff = elast_budget + coef_dict.get(inter_key, 0)
        rows2.append({
            "Género":             label,
            "Retorno por +10% budget": eff * 10,
        })
    df_eff = pd.DataFrame(rows2).sort_values("Retorno por +10% budget",
                                              ascending=True)

    fig = px.bar(df_eff, x="Retorno por +10% budget", y="Género",
                 orientation="h",
                 color="Retorno por +10% budget",
                 color_continuous_scale="RdYlGn",
                 title="Sensibilidad de la recaudación al presupuesto, por género")
    fig.update_layout(xaxis_title="% extra de revenue por cada +10% de presupuesto",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Cómo leerlo:** un género en la zona verde devuelve más revenue por
    cada peso adicional invertido. Es donde tiene sentido **escalar el
    presupuesto**. Los géneros en zonas más bajas conviene producirlos con
    presupuestos contenidos: agregar dinero rinde menos.
    """)

# ------------------ TAB 3: Productoras ------------------ #
with tab3:
    st.subheader("Sellos / productoras: el premium por distribución")
    st.markdown("""
    Manteniendo presupuesto, calidad y género constantes, **estar bajo el
    paraguas de un sello mayor** se traduce en recaudación extra. Esto
    captura el efecto de la red de distribución, marketing institucional y
    poder de negociación con cines.
    """)

    rows = []
    for col, label in productoras_legibles.items():
        if col in coef_dict:
            rows.append({
                "Sello":           label,
                "Premium revenue": premium_pct(coef_dict[col]),
                "Significativo":   pval_dict[col] < 0.05,
                "Películas":       int(df_v2[col].sum()) if col in df_v2.columns else 0,
            })

    prod_cols_modelo = [c for c in productoras_legibles if c in coef_dict]
    n_otros = int((df_v2[prod_cols_modelo].sum(axis=1) == 0).sum())
    rows.append({
        "Sello":           "Sin sello mayor (base de comparación)",
        "Premium revenue": 0.0,
        "Significativo":   False,
        "Películas":       n_otros,
    })

    df_prod = pd.DataFrame(rows).sort_values("Premium revenue")

    fig = px.bar(df_prod, x="Premium revenue", y="Sello",
                 color="Significativo",
                 color_discrete_map={True: "#2ca02c", False: "#bbbbbb"},
                 orientation="h",
                 title="Premium de revenue vs. películas independientes",
                 hover_data={"Películas": True, "Significativo": False})
    fig.update_layout(xaxis_title="% sobre una película equivalente sin sello mayor",
                      height=400)
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Implicancia:** si se va a producir una película con presupuesto medio
    o alto, **co-producir o licenciar la marca/distribución de un sello del
    top** puede sumar entre un dígito alto y dos dígitos de revenue
    adicional, sin necesidad de aumentar producción. Es una palanca
    contractual, no de presupuesto.

    **Cómo leer el 0%:** la barra "Sin sello mayor" representa la
    **categoría base** del modelo. Engloba películas independientes y a las
    productoras del top cuyo efecto no fue estadísticamente distinguible de
    cero (Walt Disney, United Artists, Columbia Pictures Corp.). Todas las
    demás barras se interpretan como **premium / descuento sobre esa base**.
    """)

# ------------------ TAB 4: Recomendaciones personalizadas ------------------ #
with tab4:
    st.subheader("Recomendaciones específicas para tu película")
    st.caption(f"Basadas en tu configuración actual: ${budget_usd/1e6:.1f}M de "
               f"presupuesto, calificación esperada {vote_avg_in}, "
               f"género(s) {', '.join(gens_sel) if gens_sel else 'sin definir'}.")

    recos = []

    if roi < 1.2:
        recos.append((
            ":red_circle: ROI bajo",
            f"Con tu configuración actual el modelo predice apenas "
            f"{roi:.2f}x sobre el budget. Sugerencias: bajar el presupuesto "
            f"a un nivel más rentable o reforzar agresivamente el awareness "
            f"para empujar la recaudación esperada."
        ))
    elif roi < 2:
        recos.append((
            ":large_orange_circle: ROI ajustado",
            f"Tu ROI esperado es {roi:.2f}x: positivo pero estrecho. "
            f"Una pequeña sorpresa negativa puede dejarte en pérdidas. "
            f"Buscar palancas de marketing antes de cerrar el budget."
        ))
    else:
        recos.append((
            ":large_green_circle: ROI sólido",
            f"Con ROI esperado de {roi:.2f}x, la apuesta luce robusta. "
            f"Conviene proteger ese margen evitando inflar el budget más "
            f"allá del óptimo y manteniendo la calidad esperada."
        ))

    # Recomendación sobre calidad
    if vote_avg_in < 6.5:
        recos.append((
            ":star: Subir la calidad esperada",
            f"Una calificación esperada de {vote_avg_in} está por debajo "
            f"del promedio de mercado. Cada punto extra de calificación "
            f"vale aproximadamente {premium_pct(coef_va):.0f}% más de "
            f"recaudación. Considerar test screenings y re-edición."
        ))

    # Recomendación sobre alcance de audiencia
    if vote_cnt_in < 1000:
        recos.append((
            ":mega: Tu estimación de alcance es baja",
            f"Configuraste una audiencia esperada de solo {vote_cnt_in} "
            f"reseñas, lo cual indica que el modelo ve poca exposición de "
            f"la película al público. **Esto NO es una palanca directa que "
            f"se mueve sola**: para subir esa cifra hay que invertir en las "
            f"verdaderas palancas comerciales: cantidad de salas, "
            f"plataformas de streaming, países de distribución, publicidad "
            f"pagada (digital + tradicional), prensa y redes sociales. "
            f"Un mayor alcance está asociado, en promedio, a "
            f"~{premium_pct(coef_vc * np.log(2)):.0f}% más de recaudación "
            f"cuando se duplica."
        ))

    # Recomendación sobre productora
    if prod_sel == "Sin sello mayor / otra" and budget_usd > 30_000_000:
        prods_signif = [(label, coef_dict[col])
                        for col, label in productoras_legibles.items()
                        if col in coef_dict and pval_dict[col] < 0.1
                        and coef_dict[col] > 0]
        prods_signif.sort(key=lambda x: -x[1])
        if prods_signif:
            top3 = ", ".join([p[0] for p in prods_signif[:3]])
            recos.append((
                ":office: Buscar partner de distribución",
                f"Para una película con tu presupuesto (>30M USD), no "
                f"contar con un sello mayor deja revenue sobre la mesa. "
                f"Los sellos con mayor premium son: **{top3}**."
            ))

    # Recomendación sobre género
    if gens_sel:
        label_to_col_g = {v: k for k, v in generos_legibles.items()}
        elasts = []
        for g in gens_sel:
            col = label_to_col_g.get(g)
            if col:
                inter_key = f"budget:{col}"
                e = elast_budget + coef_dict.get(inter_key, 0)
                elasts.append((g, e))
        if elasts:
            mejor = max(elasts, key=lambda x: x[1])
            peor  = min(elasts, key=lambda x: x[1])
            if mejor[1] > peor[1] + 0.05:
                recos.append((
                    ":dart: Foco en el género más rentable",
                    f"Entre los géneros elegidos, **{mejor[0]}** es donde "
                    f"el presupuesto rinde más (+{mejor[1]*10:.1f}% por cada "
                    f"+10% de presupuesto) y **{peor[0]}** donde rinde menos "
                    f"(+{peor[1]*10:.1f}%). El posicionamiento del marketing "
                    f"debería enfatizar el primero."
                ))

    if not recos:
        recos.append((":white_check_mark: Setup óptimo",
                      "Tu configuración actual ya combina buen presupuesto, "
                      "calidad esperada alta, awareness sólido y partner de "
                      "distribución fuerte. No hay recomendaciones críticas."))

    for titulo, texto in recos:
        st.markdown(f"#### {titulo}")
        st.markdown(texto)
        st.markdown("")

    st.divider()
    st.caption(":information_source: Las recomendaciones se actualizan "
               "automáticamente al cambiar los parámetros en el panel izquierdo.")

# ------------------ TAB 5: Películas similares ------------------ #
with tab5:
    st.subheader("Películas históricas más parecidas a tu setup")
    st.markdown(
        "Buscamos en el dataset las **10 películas reales más similares** "
        "a tu configuración (presupuesto, calidad, duración, géneros y "
        "productora) y mostramos su recaudación efectiva. Es evidencia "
        "histórica complementaria a la predicción del modelo."
    )

    log_budget_user = np.log(budget_usd)

    df_sim = df_full.copy()

    sigma_log_b = float(np.log(df_sim["budget"]).std())
    sigma_va    = float(df_sim["vote_average"].std())
    sigma_rt    = float(df_sim["runtime"].std())

    z_b  = (np.log(df_sim["budget"]) - log_budget_user) / max(sigma_log_b, 1e-6)
    z_va = (df_sim["vote_average"] - vote_avg_in)       / max(sigma_va, 1e-6)
    z_rt = (df_sim["runtime"] - runtime_in)             / max(sigma_rt, 1e-6)

    distancia = np.sqrt(z_b**2 + z_va**2 + z_rt**2)

    label_to_col_g = {v: k for k, v in generos_legibles.items()}
    user_gen_cols = [label_to_col_g[g] for g in gens_sel
                     if g in label_to_col_g and label_to_col_g[g] in df_sim.columns]
    if user_gen_cols:
        overlap_gen = df_sim[user_gen_cols].sum(axis=1)
        distancia = distancia - 0.4 * overlap_gen

    label_to_col_p = {v: k for k, v in productoras_legibles.items()}
    if prod_sel and prod_sel != "Sin sello mayor / otra":
        prod_col = label_to_col_p.get(prod_sel)
        if prod_col and prod_col in df_sim.columns:
            distancia = distancia - 0.6 * df_sim[prod_col]

    df_sim["_dist"] = distancia
    top_sim = df_sim.nsmallest(10, "_dist").copy()

    top_sim["ROI real"] = top_sim["revenue"] / top_sim["budget"]
    top_sim["Budget (M USD)"]   = top_sim["budget"]  / 1e6
    top_sim["Revenue (M USD)"]  = top_sim["revenue"] / 1e6

    cols_show = {
        "title":            "Película",
        "anio":             "Año",
        "Budget (M USD)":   "Budget (M USD)",
        "Revenue (M USD)":  "Revenue (M USD)",
        "ROI real":         "ROI real",
        "vote_average":     "Calidad",
        "runtime":          "Duración (min)",
        "genres":           "Géneros",
        "production_companies": "Productora",
    }
    tabla = top_sim[list(cols_show.keys())].rename(columns=cols_show)

    st.dataframe(
        tabla.style.format({
            "Budget (M USD)":  "${:,.1f}M",
            "Revenue (M USD)": "${:,.1f}M",
            "ROI real":        "{:.2f}x",
            "Calidad":         "{:.1f}",
            "Duración (min)":  "{:.0f}",
        }).background_gradient(subset=["ROI real"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

    roi_medio = top_sim["ROI real"].median()
    rev_medio = top_sim["revenue"].median() / 1e6
    st.info(
        f"**Realidad histórica:** entre las 10 películas más parecidas a tu "
        f"setup, la mediana de ROI fue **{roi_medio:.2f}x** y la mediana "
        f"de recaudación fue **${rev_medio:,.1f}M USD**. "
        f"Tu predicción del modelo es ${pred['revenue']/1e6:,.1f}M con "
        f"ROI {roi:.2f}x."
    )

# ------------------ TAB 6: Mapa de decisión calidad x budget ------------------ #
with tab6:
    st.subheader("Mapa de decisión: calidad × presupuesto")
    st.markdown(
        "Para cada combinación de **calidad esperada** y **presupuesto**, "
        "el modelo predice cuánto recaudaría tu película (manteniendo el "
        "resto del setup igual: género, sello, año, duración). Identifica "
        "las zonas verdes donde tu apuesta es más rentable."
    )

    n_b, n_q = 12, 13
    budgets_grid   = np.logspace(np.log10(1_000_000), np.log10(250_000_000), n_b)
    calidades_grid = np.linspace(3.0, 9.0, n_q)

    rev_matrix = np.zeros((n_q, n_b))
    roi_matrix = np.zeros((n_q, n_b))
    for i, q in enumerate(calidades_grid):
        for j, b in enumerate(budgets_grid):
            p = predecir(budget=float(b), vote_avg=float(q))
            rev_matrix[i, j] = p["revenue"] / 1e6
            roi_matrix[i, j] = p["revenue"] / float(b)

    metric_choice = st.radio(
        "Métrica a visualizar",
        ["Revenue esperado (M USD)", "ROI esperado"],
        horizontal=True,
    )

    if metric_choice.startswith("Revenue"):
        z_data = rev_matrix
        text_format = ".0f"
        cbar_title  = "Revenue (M USD)"
    else:
        z_data = roi_matrix
        text_format = ".2f"
        cbar_title  = "ROI (revenue / budget)"

    fig_heat = px.imshow(
        z_data,
        x=[f"${b/1e6:.0f}M" for b in budgets_grid],
        y=[f"{q:.1f}" for q in calidades_grid],
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels={"x": "Presupuesto", "y": "Calidad esperada (1-10)",
                "color": cbar_title},
        text_auto=text_format,
        origin="lower",
    )
    fig_heat.update_layout(height=520, title=metric_choice)
    st.plotly_chart(fig_heat, use_container_width=True)

    j_user = int(np.argmin(np.abs(np.log(budgets_grid) - np.log(budget_usd))))
    i_user = int(np.argmin(np.abs(calidades_grid - vote_avg_in)))
    st.caption(
        f"Tu setup actual cae cerca de la celda "
        f"**budget ≈ ${budgets_grid[j_user]/1e6:.0f}M** y "
        f"**calidad ≈ {calidades_grid[i_user]:.1f}**, donde el modelo "
        f"predice ${rev_matrix[i_user, j_user]:,.0f}M USD "
        f"y ROI {roi_matrix[i_user, j_user]:.2f}x."
    )

    st.info(
        "**Cómo leerlo:** las zonas más verdes son donde tu apuesta tiene "
        "mejor rendimiento esperado. Los gradientes te muestran si conviene "
        "subir calidad, subir budget, o ambas a la vez."
    )

# ------------------ TAB 7: Tornado chart de sensibilidad ------------------ #
with tab7:
    st.subheader("Sensibilidad del revenue a cada variable")
    st.markdown(
        "Para cada variable mostramos cuánto **sube** o **baja** el revenue "
        "esperado al moverla en un rango realista, dejando el resto fijo. "
        "Las barras más largas indican las palancas con mayor poder de "
        "mover el resultado."
    )

    base_rev_t = pred["revenue"]

    sensib = []

    def _add_sens(nombre, lo_kw, hi_kw, label_lo, label_hi):
        rev_lo = predecir(**lo_kw)["revenue"]
        rev_hi = predecir(**hi_kw)["revenue"]
        sensib.append({
            "Variable":    nombre,
            "Pct bajo":    (rev_lo / base_rev_t - 1) * 100,
            "Pct alto":    (rev_hi / base_rev_t - 1) * 100,
            "Etiqueta lo": label_lo,
            "Etiqueta hi": label_hi,
        })

    _add_sens("Presupuesto",
              {"budget": budget_usd * 0.8}, {"budget": budget_usd * 1.2},
              "−20%", "+20%")
    _add_sens("Calidad esperada",
              {"vote_avg": max(1, vote_avg_in - 1)},
              {"vote_avg": min(10, vote_avg_in + 1)},
              "−1 punto", "+1 punto")
    _add_sens("Duración",
              {"runtime": max(60, runtime_in - 15)},
              {"runtime": min(220, runtime_in + 15)},
              "−15 min", "+15 min")
    _add_sens("Año de estreno",
              {"anio": anio_in - 5}, {"anio": anio_in + 5},
              "−5 años", "+5 años")
    _add_sens("Engagement (popularity)",
              {"popularity": popularity_in * 0.5},
              {"popularity": popularity_in * 1.5},
              "−50%", "+50%")
    _add_sens("Reseñas (vote_count)",
              {"vote_cnt": vote_cnt_in * 0.5},
              {"vote_cnt": vote_cnt_in * 1.5},
              "−50%", "+50%")

    df_sens = pd.DataFrame(sensib)
    df_sens["Rango"] = df_sens["Pct alto"].abs() + df_sens["Pct bajo"].abs()
    df_sens = df_sens.sort_values("Rango", ascending=True)

    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(
        y=df_sens["Variable"], x=df_sens["Pct bajo"],
        orientation="h", name="Escenario bajo",
        marker_color="#d62728",
        text=[f"{v:+.1f}% ({l})" for v, l in
              zip(df_sens["Pct bajo"], df_sens["Etiqueta lo"])],
        textposition="auto",
    ))
    fig_t.add_trace(go.Bar(
        y=df_sens["Variable"], x=df_sens["Pct alto"],
        orientation="h", name="Escenario alto",
        marker_color="#2ca02c",
        text=[f"{v:+.1f}% ({l})" for v, l in
              zip(df_sens["Pct alto"], df_sens["Etiqueta hi"])],
        textposition="auto",
    ))
    fig_t.update_layout(
        barmode="overlay",
        title="¿Qué tanto cambia el revenue al mover cada variable?",
        xaxis_title="% de cambio en revenue vs. tu setup actual",
        height=max(420, 50 * len(df_sens)),
        legend=dict(orientation="h", y=-0.15),
    )
    fig_t.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_t, use_container_width=True)

    top_palanca = df_sens.iloc[-1]
    st.success(
        f"**Variable más influyente en tu setup actual:** "
        f"**{top_palanca['Variable']}** "
        f"(rango total ~{top_palanca['Rango']:.1f}% entre escenario bajo y alto)."
    )

    st.caption(
        ":information_source: Recuerda que **popularity** y **vote_count** "
        "no son palancas directas previas al estreno: son señales que se "
        "miden ex-post. Si aparecen como muy influyentes, lo que en "
        "realidad se está midiendo es **cuánto rinde invertir en marketing "
        "y distribución** para empujar esos indicadores."
    )

# ------------------ TAB 8: Comparador A/B ------------------ #
with tab8:
    st.subheader("Comparador A/B: dos planes lado a lado")
    st.markdown(
        "**Plan A** es el setup que tienes en el panel lateral. Configura "
        "un **Plan B** distinto aquí abajo y compara las dos apuestas: "
        "revenue esperado, ROI, IC95% y probabilidad de recuperar la "
        "inversión."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### :a: Plan A — del panel lateral")
        st.markdown(f"- **Budget:** ${budget_usd/1e6:,.1f}M")
        st.markdown(f"- **Calidad:** {vote_avg_in}")
        st.markdown(f"- **Duración:** {runtime_in} min")
        st.markdown(f"- **Año:** {anio_in}")
        st.markdown(f"- **Engagement (popularity):** {popularity_in}")
        st.markdown(f"- **Reseñas (vote_count):** {vote_cnt_in:,}")
        st.markdown(f"- **Géneros:** {', '.join(gens_sel) if gens_sel else '(sin género)'}")
        st.markdown(f"- **Sello:** {prod_sel}")

    with col_b:
        st.markdown("#### :b: Plan B — configura aquí")
        budget_b   = st.number_input(
            "Presupuesto B (USD)", min_value=100_000, max_value=400_000_000,
            value=int(budget_usd * 0.7), step=1_000_000, format="%d", key="budget_b")
        vote_avg_b = st.slider("Calidad B", 1.0, 10.0,
                               min(10.0, vote_avg_in + 0.5), 0.1, key="va_b")
        runtime_b  = st.slider("Duración B (min)", 60, 220, runtime_in, key="rt_b")
        anio_b     = st.slider("Año B", 1980, 2025, anio_in, key="anio_b")
        pop_b      = st.slider("Engagement B", 0.0, 200.0, popularity_in, 0.5, key="pop_b")
        vc_b       = st.slider("Reseñas B", 100, 30_000, vote_cnt_in, 100, key="vc_b")
        gens_b     = st.multiselect("Géneros B", gens_disponibles,
                                    default=gens_sel, key="gens_b")
        prod_b     = st.selectbox("Sello B", prods_disponibles,
                                  index=prods_disponibles.index(prod_sel)
                                  if prod_sel in prods_disponibles else 0,
                                  key="prod_b")

    pred_a = pred
    pred_b = predecir(
        budget=budget_b, vote_avg=vote_avg_b, popularity=pop_b, vote_cnt=vc_b,
        runtime=runtime_b, anio=anio_b, gens_label=gens_b, prod_label=prod_b,
    )

    roi_a = pred_a["revenue"] / budget_usd
    roi_b = pred_b["revenue"] / budget_b
    sigma = float(modelo.scale ** 0.5)
    prob_a = 100 * float(1 - 0.5 * (1 + math.erf(
        (np.log(budget_usd) - pred_a["log_pred"]) / (np.sqrt(2) * sigma))))
    prob_b = 100 * float(1 - 0.5 * (1 + math.erf(
        (np.log(budget_b) - pred_b["log_pred"]) / (np.sqrt(2) * sigma))))

    st.divider()
    st.markdown("#### Resultados comparados")

    m1, m2, m3 = st.columns(3)
    m1.metric("Revenue esperado",
              f"${pred_a['revenue']/1e6:,.1f}M",
              delta=f"B: ${pred_b['revenue']/1e6:,.1f}M  "
                    f"({(pred_b['revenue']/pred_a['revenue']-1)*100:+.1f}%)")
    m2.metric("ROI esperado",
              f"{roi_a:.2f}x",
              delta=f"B: {roi_b:.2f}x  ({(roi_b/roi_a-1)*100:+.1f}%)")
    m3.metric("Prob. break-even",
              f"{prob_a:.0f}%",
              delta=f"B: {prob_b:.0f}%  ({prob_b - prob_a:+.0f} pp)")

    df_comp = pd.DataFrame({
        "Métrica": ["Revenue esperado", "Revenue low (IC95)", "Revenue high (IC95)",
                    "Presupuesto", "ROI esperado", "Prob. break-even"],
        "Plan A": [pred_a["revenue"], pred_a["low"], pred_a["high"],
                   budget_usd, roi_a, prob_a],
        "Plan B": [pred_b["revenue"], pred_b["low"], pred_b["high"],
                   budget_b, roi_b, prob_b],
    })

    fig_ab = go.Figure()
    fig_ab.add_trace(go.Bar(
        x=["Revenue esperado", "Presupuesto"],
        y=[pred_a["revenue"]/1e6, budget_usd/1e6],
        name="Plan A", marker_color="#1f77b4",
        text=[f"${pred_a['revenue']/1e6:,.1f}M", f"${budget_usd/1e6:,.1f}M"],
        textposition="auto",
        error_y=dict(
            type="data", symmetric=False,
            array=[(pred_a["high"]-pred_a["revenue"])/1e6, 0],
            arrayminus=[(pred_a["revenue"]-pred_a["low"])/1e6, 0]),
    ))
    fig_ab.add_trace(go.Bar(
        x=["Revenue esperado", "Presupuesto"],
        y=[pred_b["revenue"]/1e6, budget_b/1e6],
        name="Plan B", marker_color="#ff7f0e",
        text=[f"${pred_b['revenue']/1e6:,.1f}M", f"${budget_b/1e6:,.1f}M"],
        textposition="auto",
        error_y=dict(
            type="data", symmetric=False,
            array=[(pred_b["high"]-pred_b["revenue"])/1e6, 0],
            arrayminus=[(pred_b["revenue"]-pred_b["low"])/1e6, 0]),
    ))
    fig_ab.update_layout(
        barmode="group", title="Plan A vs Plan B (USD M, con IC95% en revenue)",
        yaxis_title="USD millones", height=380)
    st.plotly_chart(fig_ab, use_container_width=True)

    if pred_b["revenue"] > pred_a["revenue"] and roi_b >= roi_a:
        st.success(
            f"**Plan B domina:** mayor revenue (+{(pred_b['revenue']/pred_a['revenue']-1)*100:.1f}%) "
            f"y mejor ROI ({roi_b:.2f}x vs {roi_a:.2f}x). Conviene B."
        )
    elif pred_a["revenue"] > pred_b["revenue"] and roi_a >= roi_b:
        st.success(
            f"**Plan A domina:** mayor revenue (+{(pred_a['revenue']/pred_b['revenue']-1)*100:.1f}%) "
            f"y mejor ROI ({roi_a:.2f}x vs {roi_b:.2f}x). Conviene A."
        )
    else:
        if roi_b > roi_a:
            st.warning(
                f"**Trade-off:** Plan A recauda más en absoluto pero Plan B "
                f"tiene mejor ROI ({roi_b:.2f}x vs {roi_a:.2f}x). "
                f"Si se optimiza retorno por dólar, conviene B; si se busca "
                f"maximizar el revenue total, A."
            )
        else:
            st.warning(
                f"**Trade-off:** Plan B recauda más en absoluto pero Plan A "
                f"tiene mejor ROI ({roi_a:.2f}x vs {roi_b:.2f}x). "
                f"Si se optimiza retorno por dólar, conviene A; si se busca "
                f"maximizar el revenue total, B."
            )

    with st.expander("Ver tabla detallada"):
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

# ------------------ TAB 9: Posicionamiento histórico ------------------ #
with tab9:
    st.subheader("¿Dónde se ubica tu predicción dentro del mercado?")
    st.markdown(
        "Comparamos tu predicción contra la **distribución real** del "
        "histórico de películas. Te dice si tu apuesta es conservadora, "
        "promedio o ambiciosa para el mercado."
    )

    df_full_pos = df_full.copy()
    df_full_pos["roi_real"] = df_full_pos["revenue"] / df_full_pos["budget"]

    pct_rev    = float((df_full_pos["revenue"] < pred["revenue"]).mean() * 100)
    pct_budget = float((df_full_pos["budget"]  < budget_usd       ).mean() * 100)
    pct_roi    = float((df_full_pos["roi_real"] < roi             ).mean() * 100)

    p1, p2, p3 = st.columns(3)
    p1.metric("Percentil de tu revenue", f"P{pct_rev:.0f}",
              delta=f"Mediana: ${df_full_pos['revenue'].median()/1e6:,.1f}M")
    p2.metric("Percentil de tu budget",  f"P{pct_budget:.0f}",
              delta=f"Mediana: ${df_full_pos['budget'].median()/1e6:,.1f}M")
    p3.metric("Percentil de tu ROI",     f"P{pct_roi:.0f}",
              delta=f"Mediana: {df_full_pos['roi_real'].median():.2f}x")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Bineamos sobre log10(revenue) para que los bins queden uniformes
        # en escala log (px.histogram con log_x=True bin-ea linealmente).
        rev_pos = df_full_pos.loc[df_full_pos["revenue"] > 0, "revenue"]
        log_rev = np.log10(rev_pos)
        fig_h_rev = px.histogram(
            x=log_rev, nbins=60,
            title="Distribución del revenue real (escala log)",
            color_discrete_sequence=["#1f77b4"],
        )
        fig_h_rev.add_vline(
            x=float(np.log10(pred["revenue"])),
            line_dash="dash", line_color="red", line_width=3,
            annotation_text=f"Tu predicción · P{pct_rev:.0f}",
            annotation_position="top",
        )
        tick_powers = list(range(int(np.floor(log_rev.min())),
                                 int(np.ceil(log_rev.max())) + 1))

        def _fmt_usd(v: float) -> str:
            if v >= 1e9:
                return f"${v/1e9:.0f}B"
            if v >= 1e6:
                return f"${v/1e6:.0f}M"
            if v >= 1e3:
                return f"${v/1e3:.0f}K"
            return f"${v:.0f}"

        fig_h_rev.update_xaxes(
            tickmode="array",
            tickvals=tick_powers,
            ticktext=[_fmt_usd(10 ** p) for p in tick_powers],
            title_text="Revenue real (USD, escala log)",
        )
        fig_h_rev.update_layout(height=380, showlegend=False,
                                yaxis_title="Cantidad de películas")
        st.plotly_chart(fig_h_rev, use_container_width=True)

    with col_g2:
        fig_h_roi = px.histogram(
            df_full_pos[df_full_pos["roi_real"] < df_full_pos["roi_real"].quantile(0.99)],
            x="roi_real", nbins=60,
            title="Distribución del ROI real (recortado al P99)",
            labels={"roi_real": "ROI real (revenue / budget)"},
            color_discrete_sequence=["#2ca02c"],
        )
        fig_h_roi.add_vline(
            x=roi, line_dash="dash", line_color="red", line_width=3,
            annotation_text=f"Tu ROI · P{pct_roi:.0f}",
            annotation_position="top",
        )
        fig_h_roi.add_vline(x=1, line_dash="dot", line_color="black",
                            annotation_text="Break-even (1x)",
                            annotation_position="bottom right")
        fig_h_roi.update_layout(height=380, showlegend=False,
                                yaxis_title="Cantidad de películas")
        st.plotly_chart(fig_h_roi, use_container_width=True)

    if gens_sel:
        st.markdown("#### Tu predicción vs. el género seleccionado")
        label_to_col_g = {v: k for k, v in generos_legibles.items()}
        gen_cols_user = [label_to_col_g[g] for g in gens_sel
                         if g in label_to_col_g and label_to_col_g[g] in df_full_pos.columns]
        if gen_cols_user:
            mask_gen = df_full_pos[gen_cols_user].sum(axis=1) > 0
            df_gen_subset = df_full_pos[mask_gen]
            if len(df_gen_subset) > 0:
                med_rev_gen = df_gen_subset["revenue"].median()
                med_roi_gen = df_gen_subset["roi_real"].median()
                pct_rev_gen = float((df_gen_subset["revenue"] < pred["revenue"]).mean() * 100)

                g1, g2, g3 = st.columns(3)
                g1.metric(f"Películas en {', '.join(gens_sel)}",
                          f"{len(df_gen_subset):,}")
                g2.metric("Mediana revenue del género",
                          f"${med_rev_gen/1e6:,.1f}M",
                          delta=f"Tu pred: {(pred['revenue']/med_rev_gen-1)*100:+.0f}%")
                g3.metric("Mediana ROI del género",
                          f"{med_roi_gen:.2f}x",
                          delta=f"Tu ROI: {(roi/med_roi_gen-1)*100:+.0f}%")

                veredicto_gen = (
                    "**Apuesta ambiciosa**: tu predicción está en el "
                    f"P{pct_rev_gen:.0f} del género."
                    if pct_rev_gen >= 75 else
                    "**Apuesta conservadora**: tu predicción está debajo "
                    f"de la mediana del género (P{pct_rev_gen:.0f})."
                    if pct_rev_gen < 50 else
                    f"**Apuesta promedio** para el género (P{pct_rev_gen:.0f})."
                )
                st.info(veredicto_gen)

    if pct_rev >= 90:
        st.warning(
            f"Tu predicción ({pct_rev:.0f}%) está en el **top 10% histórico**. "
            "Es una apuesta muy ambiciosa: revisa si los inputs son "
            "realistas (especialmente vote_count y popularity)."
        )
    elif pct_rev <= 25:
        st.info(
            f"Tu predicción está en el **bottom 25% histórico** "
            f"(P{pct_rev:.0f}). Es una apuesta conservadora; podría ser "
            "una película de bajo presupuesto rentable, pero verifica si "
            "se espera mayor alcance."
        )

# ------------------ TAB 10: Anexo — Modelo ------------------ #
with tab10:
    st.subheader(":books: Anexo técnico — Modelo de regresión")
    st.markdown("""
    Este anexo documenta el **modelo de regresión lineal múltiple (OLS)** que
    alimenta todas las predicciones del dashboard. Se entrenó replicando el
    pipeline del notebook *FINAL FINAL con gráficos.ipynb*: limpieza,
    transformación logarítmica de variables monetarias, centrado de
    predictoras y depuración de outliers vía residuos studentizados.
    """)

    st.markdown("### Pipeline en una línea")
    st.markdown(
        f"""
1. **Datos crudos:** `data__movies.csv` (películas con budget, revenue, runtime y vote_count > 0).
2. **Transformaciones:** `log(budget)`, `log(revenue)`, `log1p(popularity)`, `log1p(vote_count)`.
3. **Recorte de colas:** se eliminan los percentiles 0,5 % inferior y superior de revenue.
4. **Centrado:** se resta la media a las variables continuas para que los efectos principales sean interpretables al setup promedio.
5. **One-hot:** top 10 géneros y top 10 productoras → variables binarias `gen_*` y `prod_*`.
6. **Modelo inicial:** OLS con errores robustos (HC3) sobre {len(df_v2):,} observaciones.
7. **Filtro de outliers:** se excluyen filas con \\|residuo studentizado\\| > 2.
8. **Modelo corregido:** OLS estándar sobre {len(df_limpio):,} observaciones (el que usa el dashboard).
        """
    )

    # ----------------------------------------------------------------
    # 1. Métricas globales
    # ----------------------------------------------------------------
    st.markdown("### 1. Métricas globales del modelo corregido")

    rmse_log = float(np.sqrt(modelo.scale))
    f_pvalue = float(modelo.f_pvalue) if modelo.f_pvalue is not None else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observaciones (N)", f"{int(modelo.nobs):,}")
    m2.metric("R²", f"{modelo.rsquared:.3f}")
    m3.metric("R² ajustado", f"{modelo.rsquared_adj:.3f}")
    m4.metric("RMSE (log revenue)", f"{rmse_log:.3f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("F-statistic", f"{modelo.fvalue:,.1f}")
    m6.metric("p-value (F)", f"{f_pvalue:.2e}")
    m7.metric("AIC", f"{modelo.aic:,.0f}")
    m8.metric("BIC", f"{modelo.bic:,.0f}")

    st.caption(
        f":information_source: Outliers eliminados por |residuo studentizado| > 2: "
        f"**{len(df_v2) - len(df_limpio):,}** películas "
        f"({(len(df_v2)-len(df_limpio))/len(df_v2)*100:.1f}% del dataset)."
    )

    st.markdown("### 2. Fórmula del modelo")
    st.code(FORMULA_FINAL, language="text")
    st.caption(
        "La especificación incluye **interacciones** (operador `:`) entre el "
        "presupuesto y los géneros más sensibles, entre presupuesto y "
        "productoras de alto poder de distribución, y entre métricas de "
        "audiencia (`popularity`, `vote_count`, `vote_average`)."
    )

    # ----------------------------------------------------------------
    # 2. Tabla de coeficientes
    # ----------------------------------------------------------------
    st.markdown("### 3. Tabla de coeficientes")
    st.markdown(
        "Cada fila es una variable del modelo. **Coef.** es la elasticidad "
        "(o efecto marginal) sobre `log(revenue)`; **% revenue** lo traduce "
        "al cambio porcentual aproximado en recaudación si la variable sube "
        "una unidad. Los p-valores < 0,05 indican efectos estadísticamente "
        "significativos al 95%."
    )

    conf_int = modelo.conf_int().rename(columns={0: "IC95 inf", 1: "IC95 sup"})
    df_coefs = pd.DataFrame({
        "Coef.":     modelo.params,
        "Std. err.": modelo.bse,
        "t-stat":    modelo.tvalues,
        "p-value":   modelo.pvalues,
        "IC95 inf":  conf_int["IC95 inf"],
        "IC95 sup":  conf_int["IC95 sup"],
    })
    df_coefs["% revenue"] = (np.exp(df_coefs["Coef."]) - 1) * 100
    df_coefs["Significativo (5%)"] = df_coefs["p-value"] < 0.05
    df_coefs = df_coefs.reset_index().rename(columns={"index": "Variable"})

    st.dataframe(
        df_coefs.style.format({
            "Coef.":     "{:+.4f}",
            "Std. err.": "{:.4f}",
            "t-stat":    "{:+.2f}",
            "p-value":   "{:.4f}",
            "IC95 inf":  "{:+.4f}",
            "IC95 sup":  "{:+.4f}",
            "% revenue": "{:+.2f}%",
        }).background_gradient(subset=["Coef."], cmap="RdYlGn", vmin=-1, vmax=1),
        use_container_width=True, hide_index=True, height=520,
    )

    with st.expander(":scroll: Ver `summary()` completo de statsmodels"):
        st.code(str(modelo.summary()), language="text")

    with st.expander(":scroll: Ver `summary()` del modelo inicial (con HC3, antes de filtrar outliers)"):
        st.code(str(modelo_inicial.summary()), language="text")

    st.divider()

    # ----------------------------------------------------------------
    # 3.b Precisión de las proyecciones — Real vs. Predicho
    # ----------------------------------------------------------------
    st.markdown("### 4. Precisión de las proyecciones (Real vs. Predicho)")
    st.markdown(
        "Cada punto es una película del dataset limpio. En el eje X se "
        "ubican los **ingresos proyectados** por el modelo y en el eje Y "
        "los **ingresos reales**. La línea roja punteada es la diagonal "
        "ideal `y = x`: cuanto más alineados estén los puntos con esa "
        "línea, mejor predice el modelo."
    )

    predicciones_log = modelo.fittedvalues
    reales_log       = df_limpio["revenue"]

    pred_usd = np.exp(predicciones_log)
    real_usd = np.exp(reales_log)

    sample_n_rp = min(2500, len(pred_usd))
    idx_rp = np.random.RandomState(0).choice(len(pred_usd), sample_n_rp,
                                             replace=False)

    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatter(
        x=pred_usd.iloc[idx_rp], y=real_usd.iloc[idx_rp],
        mode="markers",
        marker=dict(size=5, color="#3498db", opacity=0.45),
        name="Películas",
        hovertemplate="Predicho: $%{x:,.0f}<br>Real: $%{y:,.0f}<extra></extra>",
    ))
    lim_lo = float(min(pred_usd.min(), real_usd.min()))
    lim_hi = float(max(pred_usd.max(), real_usd.max()))
    fig_rp.add_trace(go.Scatter(
        x=[lim_lo, lim_hi], y=[lim_lo, lim_hi],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Predicción perfecta (y = x)",
    ))
    fig_rp.update_layout(
        title="Precisión de las proyecciones — Real vs. Predicho",
        xaxis_title="Ingresos proyectados por el modelo (USD)",
        yaxis_title="Ingresos reales obtenidos (USD)",
        xaxis_type="log", yaxis_type="log",
        height=520,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_rp, use_container_width=True)

    # Métricas de ajuste sobre la propia muestra de entrenamiento
    corr_pred = float(np.corrcoef(predicciones_log, reales_log)[0, 1])
    rmse_log_train = float(np.sqrt(np.mean((reales_log - predicciones_log) ** 2)))
    mae_log_train  = float(np.mean(np.abs(reales_log - predicciones_log)))

    rp1, rp2, rp3 = st.columns(3)
    rp1.metric("Correlación Real vs. Predicho", f"{corr_pred:.3f}",
               help="Correlación de Pearson entre log(revenue) real y predicho.")
    rp2.metric("RMSE (log revenue)", f"{rmse_log_train:.3f}",
               help="Error cuadrático medio en escala log.")
    rp3.metric("MAE (log revenue)", f"{mae_log_train:.3f}",
               help="Error absoluto medio en escala log. "
                    "Aproximadamente, un MAE de 0,4 implica ~50% de error "
                    "promedio en USD.")

    st.info(
        f"**Lectura:** el modelo alcanza una correlación de "
        f"**{corr_pred:.3f}** entre ingresos reales y predichos en escala "
        f"log. La nube de puntos se concentra alrededor de la diagonal: la "
        f"calibración es buena en el rango medio, con mayor dispersión en "
        f"los extremos (películas blockbuster y de muy bajo presupuesto)."
    )

    st.divider()

    # ----------------------------------------------------------------
    # 3. Coeficientes destacados: géneros y productoras
    # ----------------------------------------------------------------
    st.markdown("### 5. Coeficientes destacados")
    st.markdown(
        "Visualizaciones equivalentes a las del notebook: **impacto base de "
        "los géneros** y **valor agregado por estudio productor**."
    )

    col_g, col_p = st.columns(2)

    with col_g:
        rows_g = []
        for col, label in generos_legibles.items():
            if col in coef_dict:
                rows_g.append({
                    "Género": label,
                    "Coeficiente": coef_dict[col],
                    "Significativo": pval_dict[col] < 0.05,
                })
        df_g = pd.DataFrame(rows_g).sort_values("Coeficiente")
        df_g["Color"] = np.where(df_g["Coeficiente"] > 0, "#2ecc71", "#e74c3c")

        fig_g = go.Figure(go.Bar(
            x=df_g["Coeficiente"], y=df_g["Género"],
            orientation="h",
            marker_color=df_g["Color"],
            text=[f"{v:+.4f}" for v in df_g["Coeficiente"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<extra></extra>",
        ))
        fig_g.add_vline(x=0, line_color="black", line_width=1.5)
        fig_g.update_layout(
            title="Impacto de los géneros en la recaudación base",
            xaxis_title="Coeficiente sobre log(revenue)",
            height=420, showlegend=False,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col_p:
        rows_p = []
        for col, label in productoras_legibles.items():
            if col in coef_dict:
                rows_p.append({
                    "Productora":   label,
                    "Coeficiente":  coef_dict[col],
                    "Significativo": pval_dict[col] < 0.05,
                })
        df_p = pd.DataFrame(rows_p).sort_values("Coeficiente")

        fig_p = go.Figure(go.Bar(
            x=df_p["Coeficiente"], y=df_p["Productora"],
            orientation="h",
            marker=dict(color=df_p["Coeficiente"], colorscale="Blues"),
            text=[f"{v:+.4f}" for v in df_p["Coeficiente"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<extra></extra>",
        ))
        fig_p.add_vline(x=0, line_color="black", line_width=1.5)
        fig_p.update_layout(
            title="Valor agregado por estudio productor",
            xaxis_title="Coeficiente sobre log(revenue)",
            height=420, showlegend=False,
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.info(
        "**Lectura:** un coeficiente de +0,46 en *Family* implica que las "
        "películas familiares recaudan, en promedio, "
        f"{(np.exp(0.46) - 1) * 100:.0f}% más que el grupo de comparación, "
        "manteniendo el resto del setup constante. Análogamente, *New Line "
        "Cinema* o *20th Century Fox* aportan ~30% extra de revenue versus "
        "una producción independiente equivalente."
    )

    st.divider()

    # ----------------------------------------------------------------
    # 4. Diagnóstico de residuos
    # ----------------------------------------------------------------
    st.markdown("### 6. Diagnóstico de residuos")
    st.markdown(
        "Validamos los supuestos clásicos de OLS: linealidad / homocedasticidad "
        "(residuos vs. ajustados), normalidad (QQ-plot e histograma) y "
        "presencia de observaciones influyentes (residuos studentizados)."
    )

    fitted = modelo.fittedvalues
    resid  = modelo.resid
    resid_std = modelo.get_influence().resid_studentized_internal

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        # Residuos vs ajustados
        sample_n = min(2000, len(fitted))
        idx_sample = np.random.RandomState(0).choice(len(fitted), sample_n, replace=False)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=fitted.iloc[idx_sample], y=resid.iloc[idx_sample],
            mode="markers",
            marker=dict(size=5, color="#1f77b4", opacity=0.45),
            name="Residuos",
            hovertemplate="ajustado: %{x:.2f}<br>residuo: %{y:.2f}<extra></extra>",
        ))
        fig_r.add_hline(y=0, line_dash="dash", line_color="black")
        fig_r.update_layout(
            title="Residuos vs. valores ajustados",
            xaxis_title="log(revenue) ajustado",
            yaxis_title="Residuo",
            height=380, showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_d2:
        # QQ-plot manual (cuantiles teóricos vs cuantiles muestrales)
        resid_sorted = np.sort(resid.values)
        teor = stats.norm.ppf(
            (np.arange(1, len(resid_sorted) + 1) - 0.5) / len(resid_sorted),
            loc=0, scale=resid_sorted.std(ddof=1),
        )
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=teor, y=resid_sorted,
            mode="markers",
            marker=dict(size=4, color="#9467bd", opacity=0.6),
            name="Residuos",
        ))
        lim = float(max(abs(teor.min()), abs(teor.max()),
                        abs(resid_sorted.min()), abs(resid_sorted.max())))
        fig_qq.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim], mode="lines",
            line=dict(color="black", dash="dash"),
            name="Referencia normal",
        ))
        fig_qq.update_layout(
            title="QQ-plot de residuos (vs. normal)",
            xaxis_title="Cuantiles teóricos",
            yaxis_title="Cuantiles muestrales",
            height=380, showlegend=False,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        fig_h = go.Figure(go.Histogram(
            x=resid, nbinsx=50, marker_color="#1f77b4",
            opacity=0.85,
        ))
        fig_h.update_layout(
            title="Distribución de residuos",
            xaxis_title="Residuo (log revenue)",
            yaxis_title="Frecuencia",
            height=360, bargap=0.02, showlegend=False,
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_d4:
        fig_s = go.Figure(go.Histogram(
            x=resid_std, nbinsx=50, marker_color="#ff7f0e",
            opacity=0.85,
        ))
        fig_s.add_vline(x=2,  line_dash="dash", line_color="red")
        fig_s.add_vline(x=-2, line_dash="dash", line_color="red")
        fig_s.update_layout(
            title="Residuos studentizados (umbral ±2)",
            xaxis_title="Residuo studentizado",
            yaxis_title="Frecuencia",
            height=360, bargap=0.02, showlegend=False,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # Tests estadísticos
    try:
        bp = sm.stats.diagnostic.het_breuschpagan(resid, modelo.model.exog)
        bp_lm_p = float(bp[1])
    except Exception:
        bp_lm_p = np.nan
    try:
        jb_stat, jb_p, _, _ = sm.stats.stattools.jarque_bera(resid)
        jb_p = float(jb_p)
    except Exception:
        jb_p = np.nan
    dw = float(sm.stats.stattools.durbin_watson(resid))

    t1, t2, t3 = st.columns(3)
    t1.metric("Breusch-Pagan p-value",
              f"{bp_lm_p:.3f}" if np.isfinite(bp_lm_p) else "n/a",
              help="H0: homocedasticidad. p < 0,05 sugiere heterocedasticidad.")
    t2.metric("Jarque-Bera p-value",
              f"{jb_p:.3f}" if np.isfinite(jb_p) else "n/a",
              help="H0: residuos normales. p < 0,05 rechaza normalidad.")
    t3.metric("Durbin-Watson", f"{dw:.2f}",
              help="≈ 2: sin autocorrelación. Por debajo de 1,5 / arriba de 2,5: posible problema.")

    st.divider()

    # ----------------------------------------------------------------
    # 5. Relaciones bivariadas (gráficos del notebook)
    # ----------------------------------------------------------------
    st.markdown("### 7. Relaciones bivariadas clave")
    st.markdown(
        "Replicamos las visualizaciones del notebook sobre el dataset "
        "**limpio** (sin outliers de residuo) en variables ya centradas y "
        "log-transformadas."
    )

    sample_plot = df_limpio.sample(min(2500, len(df_limpio)), random_state=0)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_br = go.Figure()
        fig_br.add_trace(go.Scatter(
            x=sample_plot["budget"], y=sample_plot["revenue"],
            mode="markers",
            marker=dict(size=5, color="#1f77b4", opacity=0.4),
            name="Películas",
            hovertemplate="budget: %{x:.2f}<br>revenue: %{y:.2f}<extra></extra>",
        ))
        lim_b = float(max(abs(sample_plot["budget"]).max(),
                          abs(sample_plot["revenue"]).max()))
        fig_br.add_trace(go.Scatter(
            x=[-lim_b, lim_b], y=[-lim_b, lim_b], mode="lines",
            line=dict(color="red", dash="dash"),
            name="revenue = budget",
        ))
        fig_br.update_layout(
            title="Presupuesto vs. recaudación (centrados, escala log)",
            xaxis_title="Budget (log, centrado)",
            yaxis_title="Revenue (log, centrado)",
            height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_br, use_container_width=True)

    with col_r2:
        # Vote count vs revenue con línea de regresión simple
        x_vc = sample_plot["vote_count"].values
        y_rv = sample_plot["revenue"].values
        slope, intercept, r_value, _, _ = stats.linregress(x_vc, y_rv)
        x_line = np.linspace(x_vc.min(), x_vc.max(), 100)
        y_line = intercept + slope * x_line

        fig_vc = go.Figure()
        fig_vc.add_trace(go.Scatter(
            x=x_vc, y=y_rv,
            mode="markers",
            marker=dict(size=5, color="#9467bd", opacity=0.35),
            name="Películas",
        ))
        fig_vc.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="black", width=3),
            name=f"OLS simple (r = {r_value:.2f})",
        ))
        fig_vc.update_layout(
            title="Volumen de audiencia vs. recaudación",
            xaxis_title="vote_count (log, centrado)",
            yaxis_title="Revenue (log, centrado)",
            height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_vc, use_container_width=True)

    # Matriz de correlación
    st.markdown("#### Matriz de correlación de variables financieras")
    vars_fin = ["revenue", "budget", "popularity", "vote_count", "vote_average", "runtime"]
    corr_matrix = df_limpio[vars_fin].corr().round(2)

    fig_corr = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.values.astype(str),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        showscale=True,
    )
    fig_corr.update_layout(
        title="Correlación entre variables financieras (df limpio)",
        height=460,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
        "**Lectura clave:** las correlaciones más fuertes con `revenue` son "
        "`vote_count` y `budget`. Esto justifica que sean las dos "
        "**variables centrales** del modelo (con interacciones). "
        "`vote_average` y `runtime` aportan información complementaria."
    )

    st.divider()

    # ----------------------------------------------------------------
    # 6. Análisis exploratorio (df limpio)
    # ----------------------------------------------------------------
    st.markdown("### 8. Análisis exploratorio (sobre df limpio)")

    columnas_desc = ["revenue", "budget", "vote_count", "popularity",
                     "vote_average", "runtime"]
    descripcion = df_limpio[columnas_desc].describe().T
    st.markdown("#### Descripción estadística")
    st.dataframe(
        descripcion.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("#### Distribución de las variables continuas")
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    for ax, col in zip(axes_hist.flat, columnas_desc):
        ax.hist(df_limpio[col], bins=30, color="skyblue", edgecolor="black")
        ax.set_title(col, fontweight="bold")
    fig_hist.suptitle("Distribución de variables numéricas (centradas / log)",
                      fontsize=14, fontweight="bold", y=1.02)
    fig_hist.tight_layout()
    st.pyplot(fig_hist, clear_figure=True)

    st.markdown("#### Estacionalidad: recaudación media por mes e idioma")
    if "mes" in df_full_limpio.columns and "original_language" in df_full_limpio.columns:
        df_estacion = df_full_limpio.copy()
        df_estacion["log_revenue"] = np.log(df_estacion["revenue"])
        agg_mes_lang = (df_estacion
                        .groupby(["mes", "original_language"])["log_revenue"]
                        .mean().reset_index())
        fig_mes = px.bar(
            agg_mes_lang, x="mes", y="log_revenue",
            color="original_language", barmode="group",
            labels={"log_revenue": "Recaudación media (log)",
                    "mes": "Mes de estreno",
                    "original_language": "Idioma"},
            title="Ingresos medios por mes de estreno e idioma",
        )
        fig_mes.update_layout(height=420)
        st.plotly_chart(fig_mes, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_box_mes = px.box(
                df_estacion, x="mes", y="log_revenue",
                color="mes",
                labels={"log_revenue": "log(revenue)",
                        "mes": "Mes de estreno"},
                title="Distribución de recaudación por mes",
            )
            fig_box_mes.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_box_mes, use_container_width=True)
        with col_b2:
            fig_box_lang = px.box(
                df_estacion, x="original_language", y="log_revenue",
                color="original_language",
                labels={"log_revenue": "log(revenue)",
                        "original_language": "Idioma"},
                title="Distribución de recaudación por idioma",
            )
            fig_box_lang.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_box_lang, use_container_width=True)

    st.caption(
        ":information_source: Este anexo es una réplica fiel del modelado "
        "presentado en *FINAL FINAL con gráficos.ipynb*: misma fórmula, "
        "mismo filtrado de outliers y mismas variables centradas / "
        "log-transformadas que alimentan al simulador del dashboard."
    )
