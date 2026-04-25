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
    df_v2 = df_v2.drop(columns=["homepage", "status", "tagline", "title",
                                "original_title", "overview", "id"])
    for col in df_v2.select_dtypes(include=["object"]).columns:
        df_v2[col] = df_v2[col].str.lower()
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

    df_v2["log_budget"]     = np.log(df_v2["budget"])
    df_v2["log_revenue"]    = np.log(df_v2["revenue"])
    df_v2["log_popularity"] = np.log1p(df_v2["popularity"])
    df_v2["log_vote_count"] = np.log1p(df_v2["vote_count"])

    df_v2_full = df_v2.copy()  # antes del drop de columnas crudas

    df_v2 = df_v2.drop(columns=["genres", "production_companies",
                                "budget", "revenue", "popularity", "vote_count"])

    q_low  = df_v2["log_revenue"].quantile(0.005)
    q_high = df_v2["log_revenue"].quantile(0.995)
    keep_idx = (df_v2["log_revenue"] >= q_low) & (df_v2["log_revenue"] <= q_high)
    df_v2 = df_v2[keep_idx].copy()
    df_v2_full = df_v2_full.loc[df_v2.index].copy()

    medias = {}
    for col in ["log_budget", "log_popularity", "log_vote_count",
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
    "prod_touchstone_pictures":                   "Touchstone Pictures",
    "prod_metro_goldwyn_mayer_mgm":               "MGM",
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
prod_cols = [c for c in df_v2.columns if c.startswith("prod_")]

gens_disponibles  = [generos_legibles.get(c, c.replace("gen_", "").title())
                     for c in gen_cols]
prods_disponibles = ["(ninguna del top 10)"] + \
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

    fila = {col: 0.0 for col in df_v2.columns if col != "log_revenue"}
    fila["log_budget"]     = np.log(budget)         - medias["log_budget"]
    fila["log_popularity"] = np.log1p(popularity)   - medias["log_popularity"]
    fila["log_vote_count"] = np.log1p(vote_cnt)     - medias["log_vote_count"]
    fila["vote_average"]   = vote_avg               - medias["vote_average"]
    fila["runtime"]        = runtime                - medias["runtime"]
    fila["anio"]           = anio                   - medias["anio"]

    label_to_col_g = {v: k for k, v in generos_legibles.items()}
    for g in gens_label:
        col = label_to_col_g.get(g)
        if col and col in fila:
            fila[col] = 1
    label_to_col_p = {v: k for k, v in productoras_legibles.items()}
    if prod_label and prod_label != "(ninguna del top 10)":
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
elast_budget = coef_dict.get("log_budget",     0)
coef_va      = coef_dict.get("vote_average",   0)
coef_vc      = coef_dict.get("log_vote_count", 0)
coef_pop     = coef_dict.get("log_popularity", 0)

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
tab1, tab2, tab3, tab4 = st.tabs([
    ":bulb: Las palancas que mueven el revenue",
    ":movie_camera: Géneros: dónde rinde más invertir",
    ":office: Sellos: el premium por distribución",
    ":memo: Recomendaciones para tu lanzamiento",
])

# ------------------ TAB 1: Palancas ------------------ #
with tab1:
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
    if prod_sel == "(ninguna del top 10)":
        # Probar cada productora del top 10 y elegir la que más sume
        mejor_prod, mejor_delta = None, -np.inf
        for label_p in productoras_legibles.values():
            tmp = predecir(prod_label=label_p)
            d = (tmp["revenue"] / base_rev - 1) * 100
            if d > mejor_delta:
                mejor_delta, mejor_prod = d, label_p
        if mejor_prod:
            _agregar(f"Asociarte con {mejor_prod}",
                     {"prod_label": mejor_prod}, "Distribución",
                     "Co-producir o licenciar bajo el mejor sello del top 10")
    else:
        # Probar cambiar al mejor alternativo
        mejor_alt, mejor_delta = None, -np.inf
        for label_p in productoras_legibles.values():
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
        inter_key = f"log_budget:{col}"
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
    if prod_sel == "(ninguna del top 10)" and budget_usd > 30_000_000:
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
                inter_key = f"log_budget:{col}"
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
