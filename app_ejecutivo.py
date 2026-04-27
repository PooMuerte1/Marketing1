"""
Dashboard Ejecutivo — ¿Cuánto va a recaudar mi película?
Versión orientada a empresario / decisor de marketing.

Cómo ejecutar:
    python -m streamlit run app_ejecutivo.py
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

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


@st.cache_resource(show_spinner="Calibrando modelo de predicción...")
def ajustar_modelo(df_v2: pd.DataFrame):
    formula_final = """
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
    """
    modelo_inicial = smf.ols(formula_final, data=df_v2).fit()
    residuos_estudentizados = modelo_inicial.get_influence().resid_studentized_internal
    df_limpio = df_v2[abs(residuos_estudentizados) <= 2.0].copy()

    modelo_corregido = smf.ols(formula_final, data=df_limpio).fit()
    return modelo_corregido


# ------------------------------------------------------------------ #
# Cargar todo
# ------------------------------------------------------------------ #
df_v2  = construir_df_v2()
df_full = df_v2.attrs["full"]
modelo = ajustar_modelo(df_v2)
medias = df_v2.attrs["medias_centrado"]

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
st.sidebar.caption("Configurá los parámetros y observá la predicción en vivo.")

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
          "con la película. Es un INDICADOR del nivel de interés que vas a "
          "generar, no una palanca que se mueve sola: crece con inversión "
          "en distribución, plataformas y publicidad."))
vote_cnt_in = st.sidebar.slider(
    "Tamaño de audiencia que reseñará",
    100, 30_000, 1_500, 100,
    help=("Cantidad de personas que terminan calificando la película en TMDB. "
          "Es un INDICADOR del alcance que vas a lograr; no es una variable "
          "que controles directamente. La palanca real para subirlo es la "
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
st.caption("Modelo entrenado con +4.700 películas. Ajustá los parámetros en el "
           "panel izquierdo y observá los resultados.")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    ":bulb: Palancas",
    ":movie_camera: Géneros",
    ":office: Sellos",
    ":memo: Recomendaciones",
    ":mag: Películas similares",
    ":world_map: Mapa de decisión",
    ":tornado: Sensibilidad",
    ":scales: Comparador A/B",
    ":bar_chart: Posicionamiento histórico",
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
    por ejemplo, "subir la calidad" rinde diferente según el budget y el
    género que elegiste. Por eso los porcentajes cambian al mover el
    sidebar.
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
    Estos NO son decisiones que tomás antes de filmar. Son **señales que se
    miden después del estreno** (cantidad de gente que reseñó la película,
    popularidad acumulada en TMDB). Tienen una relación circular con el
    revenue: una peli que recauda más es vista por más gente y por eso
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
    Para cada género calculamos cuánto sube la recaudación si subís el
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
    **Implicancia:** si vas a producir una película con presupuesto medio o
    alto, **co-producir o licenciar la marca/distribución de un sello del
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
                    f"Entre los géneros que elegiste, **{mejor[0]}** es donde "
                    f"el presupuesto rinde más (+{mejor[1]*10:.1f}% por cada "
                    f"+10% de budget) y **{peor[0]}** donde rinde menos "
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
        "resto del setup igual: género, sello, año, duración). Identificá "
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
        ":information_source: Recordá que **popularity** y **vote_count** "
        "no son palancas directas previas al estreno: son señales que se "
        "miden ex-post. Si aparecen como muy influyentes, lo que en "
        "realidad estás midiendo es **cuánto rinde invertir en marketing y "
        "distribución** para empujar esos indicadores."
    )

# ------------------ TAB 8: Comparador A/B ------------------ #
with tab8:
    st.subheader("Comparador A/B: dos planes lado a lado")
    st.markdown(
        "**Plan A** es el setup que tenés en el sidebar. Configurá un "
        "**Plan B** distinto acá abajo y compará las dos apuestas: revenue "
        "esperado, ROI, IC95% y probabilidad de recuperar la inversión."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### :a: Plan A — del sidebar")
        st.markdown(f"- **Budget:** ${budget_usd/1e6:,.1f}M")
        st.markdown(f"- **Calidad:** {vote_avg_in}")
        st.markdown(f"- **Duración:** {runtime_in} min")
        st.markdown(f"- **Año:** {anio_in}")
        st.markdown(f"- **Engagement (popularity):** {popularity_in}")
        st.markdown(f"- **Reseñas (vote_count):** {vote_cnt_in:,}")
        st.markdown(f"- **Géneros:** {', '.join(gens_sel) if gens_sel else '(sin género)'}")
        st.markdown(f"- **Sello:** {prod_sel}")

    with col_b:
        st.markdown("#### :b: Plan B — configurá acá")
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
                f"Si optimizás retorno por dólar, elegí B; si querés "
                f"maximizar el revenue total, A."
            )
        else:
            st.warning(
                f"**Trade-off:** Plan B recauda más en absoluto pero Plan A "
                f"tiene mejor ROI ({roi_a:.2f}x vs {roi_b:.2f}x). "
                f"Si optimizás retorno por dólar, elegí A; si querés "
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
        fig_h_rev = px.histogram(
            df_full_pos, x="revenue", nbins=60, log_x=True,
            title="Distribución del revenue real (escala log)",
            labels={"revenue": "Revenue real (USD)"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_h_rev.add_vline(
            x=pred["revenue"], line_dash="dash", line_color="red", line_width=3,
            annotation_text=f"Tu predicción · P{pct_rev:.0f}",
            annotation_position="top",
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
            "Es una apuesta muy ambiciosa: revisá si los inputs son "
            "realistas (especialmente vote_count y popularity)."
        )
    elif pct_rev <= 25:
        st.info(
            f"Tu predicción está en el **bottom 25% histórico** "
            f"(P{pct_rev:.0f}). Es una apuesta conservadora; podría ser "
            "una peli de bajo budget rentable, pero verificá si esperás "
            "mayor alcance."
        )
