# Dashboard ejecutivo — Documentación de tablas y cálculos

Documento de referencia técnica del dashboard `app_ejecutivo.py`. Detalla
cada tabla, métrica y gráfico que se muestra al usuario, junto con la
fórmula y el código exacto utilizado para calcularlo.

---

## 0. Pipeline de datos y modelo

### 0.1 Construcción del dataset (`construir_df_v2`)

| Paso | Operación | Resultado |
|---|---|---|
| 1 | Cargar `data__movies.csv` | dataframe crudo |
| 2 | Eliminar columnas no informativas | drop de `homepage`, `status`, `tagline`, `original_title`, `overview`, `id` |
| 3 | Quedarnos con la primera productora | `production_companies.split(',').str[0]` |
| 4 | Lower-case de strings | normalización |
| 5 | Imputar nulos en `genres` y `production_companies` | `"desconocido"` y `"otros"` |
| 6 | Convertir `release_date` a datetime y extraer `anio` y `mes` | dos columnas nuevas |
| 7 | Filtrar registros válidos | `budget>0`, `revenue>0`, `runtime>0`, `vote_count>0` |
| 8 | One-hot top-10 géneros | columnas `gen_*` (0/1) |
| 9 | One-hot top-10 productoras | columnas `prod_*` (0/1) |
| 10 | Reagrupar idiomas raros como `"other"` | top-5 + `other` |
| 11 | Transformaciones logarítmicas | `log(budget)`, `log(revenue)`, `log1p(popularity)`, `log1p(vote_count)` |
| 12 | Recortar colas extremas de revenue | quitar fuera de `[Q0,5%, Q99,5%]` |
| 13 | Centrar variables continuas | restar media de `budget`, `popularity`, `vote_count`, `vote_average`, `runtime`, `anio` |
| 14 | Reset de índices | dataset final `df_v2` |

Las medias usadas para centrar se guardan en
`df_v2.attrs["medias_centrado"]` y se reutilizan luego en el simulador
para des/centrar nuevas observaciones del usuario.

### 0.2 Ajuste del modelo (`ajustar_modelo`)

Fórmula completa (`FORMULA_FINAL`):

```text
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
```

Procedimiento en dos etapas:

1. **Modelo inicial**: OLS con errores robustos `HC3` sobre todo `df_v2`.
2. **Filtro de outliers**: se eliminan filas con
   \(\left|\text{residuo studentizado}\right| > 2\).
3. **Modelo corregido**: OLS estándar sobre el dataset filtrado
   (`df_limpio`). **Este es el modelo que usa todo el dashboard.**

---

## 1. Header — Predicción para tu película

### 1.1 Función central de predicción `predecir(...)`

Toma los inputs del panel lateral (o overrides) y construye una fila
con las mismas transformaciones que el dataset:

```text
fila[budget]       = log(budget_usuario)        - media_budget
fila[popularity]   = log1p(popularity_usuario)  - media_popularity
fila[vote_count]   = log1p(vote_cnt_usuario)    - media_vote_count
fila[vote_average] = vote_avg_usuario           - media_vote_average
fila[runtime]      = runtime_usuario            - media_runtime
fila[anio]         = anio_usuario               - media_anio
fila[gen_*]        = 1 si el género está en gens_label, sino 0
fila[prod_*]       = 1 si la productora coincide, sino 0
```

Output:

| Variable | Cálculo |
|---|---|
| `pred_log` | `modelo.predict(fila)` (log de revenue esperado) |
| `se_obs` | `modelo.get_prediction(fila).se_obs` (error std de la predicción individual) |
| `revenue` | \(e^{\text{pred\_log}}\) |
| `low` | \(e^{\text{pred\_log} - 1{,}96 \cdot se_{obs}}\) |
| `high` | \(e^{\text{pred\_log} + 1{,}96 \cdot se_{obs}}\) |

### 1.2 Métricas del header

| Métrica | Fórmula |
|---|---|
| **Recaudación esperada** | `revenue` (de `predecir()`) |
| **Presupuesto** | `budget_usd` (input del panel lateral) |
| **ROI esperado** | `revenue / budget_usd` |
| **Probabilidad de break-even** | \(P(\text{revenue} \ge \text{budget}) = 1 - \Phi\!\left(\dfrac{\log(\text{budget}) - \text{pred\_log}}{\sigma}\right)\) con \(\sigma = \sqrt{\text{modelo.scale}}\) |
| **Rango plausible 95%** | \([\text{low}, \text{high}]\) |

### 1.3 Semáforo ejecutivo

| Color | Condición |
|---|---|
| Verde — *Apuesta sólida* | `roi >= 2.5` y `prob_break_even >= 75` |
| Amarillo — *Apuesta razonable, ajustada* | `roi >= 1.2` y `prob_break_even >= 55` |
| Rojo — *Riesgo elevado* | en cualquier otro caso |

---

## 2. TAB 1 — Palancas

### 2.1 Curva de retorno por presupuesto

Se generan **60 puntos** en `np.logspace(log10(500_000), log10(300M), 60)`
y para cada uno se llama `predecir(budget=b)` manteniendo el resto fijo.

Tabla `df_curva`:

| Columna | Cálculo |
|---|---|
| `Presupuesto` | grilla logarítmica |
| `Revenue esperado` | `predecir(budget=b)["revenue"]` |
| `Low IC95` / `High IC95` | `predecir(...)["low"]` / `["high"]` |
| `ROI` | `Revenue esperado / Presupuesto` |

**Punto óptimo** = fila con `ROI` máximo.

### 2.2 Tabla de palancas dinámicas (`df_pal`)

Cada palanca llama `predecir(**override)` y registra:

| Columna | Cálculo |
|---|---|
| `Δ revenue %` | \(\dfrac{\text{nuevo}_{rev} - \text{base}_{rev}}{\text{base}_{rev}} \times 100\) |
| `Δ revenue USD` | `nuevo_revenue - base_revenue` |
| `Nuevo revenue` | `predecir(**override)["revenue"]` |

Palancas evaluadas:

| # | Palanca | Override |
|---|---|---|
| 1 | Subir presupuesto +20% | `budget = budget_usd * 1.2` |
| 2 | Bajar presupuesto −20% | `budget = budget_usd * 0.8` |
| 3 | Subir calidad +1 (cap 10) | `vote_avg = min(vote_avg+1, 10)` |
| 4 | Recorte de calidad −1 (floor 1) | `vote_avg = max(vote_avg-1, 1)` |
| 5 | Acortar película −15 min | `runtime = max(60, runtime-15)` |
| 6 | Alargar película +15 min | `runtime = min(220, runtime+15)` |
| 7 | Mejor sello (si actualmente no tiene) | iteramos todas las prods, elegimos la de mayor `Δ revenue %` |
| 8 | Mejor sello alternativo (si ya tiene uno) | igual, pero excluyendo el actual y solo si `Δ > 0,5%` |
| 9 | Sumar el género más rentable que no tenga | iteramos géneros faltantes, mayor `Δ` |
| 10 | Quitar el género menos rentable (si tiene ≥ 2) | iteramos los actuales, eliminamos el peor |
| 11 | Combo: budget +20% + calidad +1 | ambos overrides juntos |

### 2.3 Indicadores de éxito (no palancas directas)

Para cada indicador se calcula una asociación promedio con revenue
basándonos en su coeficiente:

| Indicador | Cálculo |
|---|---|
| Doblar la audiencia que reseña | `premium_pct(coef_vote_count * log(2))` |
| Doblar el engagement / popularidad | `premium_pct(coef_popularity * log(2))` |

Donde \(\text{premium\_pct}(c) = (e^{c} - 1) \times 100\).

---

## 3. TAB 2 — Géneros

### 3.1 Premium por género (`df_gen`)

Para cada `gen_*` con coeficiente significativo (`p < 0,05`):

| Columna | Cálculo |
|---|---|
| `Género` | label legible |
| `Premium revenue` | `(exp(coef) - 1) * 100` |
| `Significativo` | `pval < 0.05` |
| `Películas` | `df_v2[col].sum()` (cantidad de pelis con ese género) |

### 3.2 Sensibilidad al presupuesto, por género (`df_eff`)

Para cada género calculamos el efecto marginal del presupuesto que
**incluye la interacción**:

\[
\text{retorno por +10\% budget} = (\beta_{budget} + \beta_{budget:gen}) \times 10
\]

Donde \(\beta_{budget:gen} = 0\) si el modelo no tiene esa interacción.

---

## 4. TAB 3 — Sellos / productoras

Tabla `df_prod`:

| Columna | Cálculo |
|---|---|
| `Sello` | label legible |
| `Premium revenue` | `(exp(coef_prod) - 1) * 100` |
| `Significativo` | `pval_prod < 0.05` |
| `Películas` | `df_v2[col].sum()` |
| Fila base "Sin sello mayor" | `Premium = 0%`, `Películas = (sum prod_* == 0).sum()` |

---

## 5. TAB 4 — Recomendaciones personalizadas

Reglas activadas según el setup actual del usuario:

| Trigger | Recomendación generada |
|---|---|
| `roi < 1.2` | "ROI bajo" — sugerir bajar presupuesto o reforzar awareness |
| `1.2 ≤ roi < 2` | "ROI ajustado" — buscar palancas de marketing |
| `roi ≥ 2` | "ROI sólido" — proteger margen |
| `vote_avg_in < 6.5` | "Subir calidad esperada" — cita `premium_pct(coef_vote_average)` |
| `vote_cnt_in < 1000` | "Estimación de alcance baja" — cita `premium_pct(coef_vc * log(2))` |
| `prod_sel == "Sin sello mayor"` y `budget > 30M` | sugiere top-3 sellos con `coef > 0` y `pval < 0.1` |
| Si los géneros elegidos tienen elasticidad-budget muy diferente (≥ 0,05) | recomienda enfatizar el más rentable |
| Si nada se cumple | "Setup óptimo" |

---

## 6. TAB 5 — Películas similares

Distancia ponderada en espacio z-score:

\[
d_i = \sqrt{z_{b,i}^2 + z_{va,i}^2 + z_{rt,i}^2}
\quad - \; 0{,}4 \cdot \text{overlap\_genero}_i
\quad - \; 0{,}6 \cdot \mathbb{1}[\text{misma productora}_i]
\]

Donde:

| z-score | Cálculo |
|---|---|
| `z_b` | `(log(budget_i) - log(budget_user)) / std(log(budget))` |
| `z_va` | `(vote_average_i - vote_avg_user) / std(vote_average)` |
| `z_rt` | `(runtime_i - runtime_user) / std(runtime)` |

Las **10 películas con menor distancia** se muestran en la tabla, con
columnas formateadas (Budget, Revenue, ROI real, Calidad, Duración,
Géneros, Productora). Mediana de ROI y revenue se muestran como
benchmark.

---

## 7. TAB 6 — Mapa de decisión calidad × presupuesto

Grilla de **12 presupuestos × 13 calidades**:

```python
budgets_grid   = np.logspace(log10(1_000_000), log10(250_000_000), 12)
calidades_grid = np.linspace(3.0, 9.0, 13)
```

Para cada celda \((q, b)\):

| Métrica | Cálculo |
|---|---|
| Revenue (M USD) | `predecir(budget=b, vote_avg=q)["revenue"] / 1e6` |
| ROI | `predecir(...)["revenue"] / b` |

El usuario elige cuál visualizar (Revenue o ROI) en un heatmap.

---

## 8. TAB 7 — Tornado de sensibilidad

Para cada variable, se define un escenario "bajo" y "alto" y se
mide la variación porcentual de revenue:

| Variable | Escenario bajo | Escenario alto |
|---|---|---|
| Presupuesto | `budget * 0.8` | `budget * 1.2` |
| Calidad esperada | `max(1, vote_avg-1)` | `min(10, vote_avg+1)` |
| Duración | `max(60, runtime-15)` | `min(220, runtime+15)` |
| Año de estreno | `anio - 5` | `anio + 5` |
| Engagement (popularity) | `popularity * 0.5` | `popularity * 1.5` |
| Reseñas (vote_count) | `vote_cnt * 0.5` | `vote_cnt * 1.5` |

Por cada variable se calcula:

\[
\text{Pct bajo} = \dfrac{rev_{bajo}}{rev_{base}} - 1, \quad
\text{Pct alto} = \dfrac{rev_{alto}}{rev_{base}} - 1
\]

\[
\text{Rango} = \left|\text{Pct alto}\right| + \left|\text{Pct bajo}\right|
\]

Se ordenan por `Rango` (la variable más sensible queda al tope).

---

## 9. TAB 8 — Comparador A/B

| Columna | Cálculo |
|---|---|
| **Plan A** | usa los inputs del panel lateral; `pred_a = predecir()` |
| **Plan B** | usa los inputs B locales; `pred_b = predecir(budget=budget_b, vote_avg=vote_avg_b, ...)` |
| `roi_a` / `roi_b` | `pred_x["revenue"] / budget_x` |
| `prob_a` / `prob_b` | misma fórmula que el header (\(1 - \Phi\!\left(\dfrac{\log(\text{budget}) - \text{pred\_log}}{\sigma}\right)\)) |

Tabla resumida `df_comp`:

| Métrica | Plan A | Plan B |
|---|---|---|
| Revenue esperado | `pred_a["revenue"]` | `pred_b["revenue"]` |
| Revenue low (IC95) | `pred_a["low"]` | `pred_b["low"]` |
| Revenue high (IC95) | `pred_a["high"]` | `pred_b["high"]` |
| Presupuesto | `budget_usd` | `budget_b` |
| ROI esperado | `roi_a` | `roi_b` |
| Prob. break-even | `prob_a` | `prob_b` |

**Veredicto automático**:

| Caso | Mensaje |
|---|---|
| `revenue_b > revenue_a` y `roi_b ≥ roi_a` | Plan B domina |
| `revenue_a > revenue_b` y `roi_a ≥ roi_b` | Plan A domina |
| caso mixto | trade-off explícito (más revenue absoluto vs mejor ROI) |

---

## 10. TAB 9 — Posicionamiento histórico

Comparamos la predicción del usuario contra `df_full` (dataset previo a
las transformaciones log).

| Métrica | Cálculo |
|---|---|
| `pct_rev` | `(df_full.revenue < pred.revenue).mean() * 100` |
| `pct_budget` | `(df_full.budget < budget_usd).mean() * 100` |
| `pct_roi` | `(df_full.roi_real < roi).mean() * 100` con `roi_real = revenue / budget` |

**Histograma de revenue** sobre `log10(revenue)` (60 bins) con línea
roja punteada en `log10(pred_revenue)`.

**Histograma de ROI** recortado al P99 (60 bins) con línea roja en `roi`
del usuario y línea negra punteada en break-even (`ROI = 1`).

**Tabla por género** (si el usuario seleccionó al menos un género):

| Métrica | Cálculo |
|---|---|
| `Películas en {género}` | `len(df_full[mask_gen])` |
| `Mediana revenue del género` | `df_gen_subset["revenue"].median()` |
| `Mediana ROI del género` | `df_gen_subset["roi_real"].median()` |
| Delta vs. predicción | `(pred_revenue / med_rev_gen - 1) * 100` |

Veredictos:

| Condición | Mensaje |
|---|---|
| `pct_rev_gen ≥ 75` | Apuesta ambiciosa |
| `pct_rev_gen < 50` | Apuesta conservadora |
| caso intermedio | Apuesta promedio |
| `pct_rev ≥ 90` | Top 10% histórico — revisar inputs |
| `pct_rev ≤ 25` | Bottom 25% histórico — apuesta conservadora |

---

## 11. TAB 10 — Anexo técnico del modelo

### 11.1 Métricas globales

| Métrica | Origen |
|---|---|
| `N` | `modelo.nobs` |
| `R²` | `modelo.rsquared` |
| `R² ajustado` | `modelo.rsquared_adj` |
| `RMSE (log revenue)` | \(\sqrt{\text{modelo.scale}}\) |
| `F-statistic` | `modelo.fvalue` |
| `p-value (F)` | `modelo.f_pvalue` |
| `AIC` / `BIC` | `modelo.aic`, `modelo.bic` |
| Outliers eliminados | `len(df_v2) - len(df_limpio)` |

### 11.2 Tabla de coeficientes (`df_coefs`)

| Columna | Cálculo |
|---|---|
| `Coef.` | `modelo.params` |
| `Std. err.` | `modelo.bse` |
| `t-stat` | `modelo.tvalues` |
| `p-value` | `modelo.pvalues` |
| `IC95 inf` / `IC95 sup` | `modelo.conf_int()` |
| `% revenue` | `(exp(Coef.) - 1) * 100` |
| `Significativo (5%)` | `p-value < 0.05` |

Adicionalmente se muestra el `summary()` completo de `statsmodels`
tanto del modelo corregido como del modelo inicial (HC3, antes del
filtrado).

### 11.3 Precisión del modelo — Real vs. Predicho

| Cálculo | Fórmula |
|---|---|
| `predicciones_log` | `modelo.fittedvalues` |
| `reales_log` | `df_limpio["revenue"]` (log de revenue real) |
| `pred_usd` | `exp(predicciones_log)` |
| `real_usd` | `exp(reales_log)` |
| Correlación Real vs. Predicho | `np.corrcoef(predicciones_log, reales_log)[0, 1]` |
| RMSE (log revenue) | \(\sqrt{\frac{1}{N}\sum (real_i - pred_i)^2}\) |
| MAE (log revenue) | \(\frac{1}{N}\sum |real_i - pred_i|\) |

Visualización: scatter en escala log-log de `pred_usd` vs `real_usd`
(muestra de hasta 2.500 puntos) con línea diagonal `y = x` en rojo
punteado (predicción perfecta).

### 11.4 Coeficientes destacados

Dos gráficos de barras horizontales construidos directamente desde
`coef_dict` y `pval_dict`:

| Gráfico | Datos |
|---|---|
| Impacto de géneros | `coef_dict[gen_*]` para cada género del modelo |
| Valor agregado por productora | `coef_dict[prod_*]` para cada productora del modelo |

### 11.5 Diagnóstico de residuos

| Visualización | Cálculo |
|---|---|
| Residuos vs. ajustados | `modelo.fittedvalues` vs. `modelo.resid` (muestra de 2.000 puntos) |
| QQ-plot | cuantiles teóricos `stats.norm.ppf((i-0.5)/N, scale=std(resid))` vs. cuantiles muestrales |
| Histograma de residuos | 50 bins de `modelo.resid` |
| Residuos studentizados | `modelo.get_influence().resid_studentized_internal` con líneas en ±2 |

Tests estadísticos:

| Test | Cálculo | H0 |
|---|---|---|
| Breusch-Pagan | `sm.stats.diagnostic.het_breuschpagan(resid, modelo.model.exog)[1]` | homocedasticidad |
| Jarque-Bera | `sm.stats.stattools.jarque_bera(resid)[1]` | residuos normales |
| Durbin-Watson | `sm.stats.stattools.durbin_watson(resid)` | sin autocorrelación (≈ 2) |

### 11.6 Relaciones bivariadas (sobre `df_limpio`, muestra de 2.500)

| Gráfico | Cálculo |
|---|---|
| Budget vs. Revenue | scatter de `budget` (log centrado) vs. `revenue` (log centrado) + línea `revenue = budget` |
| Vote count vs. Revenue | scatter + línea OLS simple `stats.linregress(vote_count, revenue)` con coeficiente `r` |
| Matriz de correlación | `df_limpio[[revenue, budget, popularity, vote_count, vote_average, runtime]].corr()` |

### 11.7 Análisis exploratorio

| Visualización | Cálculo |
|---|---|
| Descripción estadística | `df_limpio[columnas_desc].describe().T` |
| Histogramas (matplotlib) | un hist por variable continua (30 bins) |
| Estacionalidad mes × idioma | `groupby(["mes", "original_language"])["log_revenue"].mean()` con `log_revenue = log(revenue)` |
| Boxplot por mes | `px.box(df_full_limpio, x="mes", y=log_revenue)` |
| Boxplot por idioma | `px.box(df_full_limpio, x="original_language", y=log_revenue)` |

---

## Apéndice — Funciones auxiliares clave

### `premium_pct(coef)`

\[
\text{premium\_pct}(c) = (e^{c} - 1) \times 100
\]

Convierte un coeficiente del modelo (sobre `log(revenue)`) en un
porcentaje aproximado de cambio en revenue.

### `_limpiar(s)`

Sanitiza un string para usarlo como sufijo de columna: reemplaza
espacios y puntuación por `_`, colapsa `__` repetidos.

### Cacheo de Streamlit

| Recurso | Decorador | Motivo |
|---|---|---|
| `construir_df_v2()` | `@st.cache_data` | dataset transformado, hashable |
| `ajustar_modelo()` | `@st.cache_resource` | objeto modelo (no hashable) |

---

**Nota final:** todos los porcentajes y deltas que ve el usuario en el
dashboard derivan de las fórmulas anteriores aplicadas sobre el
**modelo corregido** (`modelo`), entrenado con `df_limpio`.
