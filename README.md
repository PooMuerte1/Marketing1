# Predicción de Revenue de Películas — Modelación de Sistemas 2025-2

Trabajo de marketing universitario: predicción de demanda potencial (proxy
`revenue`) para películas, usando regresión lineal sobre el dataset TMDB.

## Contenido del repositorio

| Archivo | Qué es |
|---|---|
| `regresión 2 géneros.ipynb` | Notebook principal con todo el análisis: limpieza, EDA, justificaciones visuales, modelo OLS final y diagnóstico. |
| `app_ejecutivo.py` | Dashboard Streamlit orientado a empresario / decisor de marketing. Predicción interactiva, palancas accionables y recomendaciones personalizadas. |
| `app_streamlit.py` | Dashboard Streamlit orientado a presentación técnica. Métricas estadísticas (R², AIC, coeficientes, diagnóstico). |
| `optimizar_ols.py` | Script standalone que automatiza la selección de variables por stepwise AIC. |
| `data__movies.csv` | Dataset crudo (películas con metadata + revenue + budget). |
| `requirements.txt` | Dependencias de Python. |

## Cómo correr los dashboards localmente

### Requisitos

- Python 3.10 o superior.
- Los archivos del repo en una misma carpeta.

### Instalación de dependencias (una sola vez)

```bash
pip install -r requirements.txt
```

### Lanzar el dashboard ejecutivo

```bash
python -m streamlit run app_ejecutivo.py
```

Se abre solo en el navegador en `http://localhost:8501`.

### Lanzar el dashboard técnico

```bash
python -m streamlit run app_streamlit.py
```

### Si Streamlit pide email al primer arranque

Dejarlo en blanco y apretar Enter, o usar:

```bash
python -m streamlit run app_ejecutivo.py --server.headless=true --browser.gatherUsageStats=false
```

## Cómo correr el notebook

Abrirlo con Jupyter / VS Code / Cursor y ejecutar todas las celdas en orden.
La pipeline es autocontenida: solo necesita `data__movies.csv` en la misma
carpeta.

## Modelo final — resumen

- **Variable dependiente:** `log(revenue)`.
- **Pipeline:** filtros de validez (>0), one-hot de top-10 géneros y top-10
  productoras, log-transform, trimming P0.5/P99.5, centrado.
- **Selección de variables:** stepwise forward-backward por AIC sobre un pool
  de 80+ candidatas.
- **R² ajustado:** ~0.75. Condition number controlado por centrado.

## Equipo

Trabajo grupal — Modelación de Sistemas 2025-2.
