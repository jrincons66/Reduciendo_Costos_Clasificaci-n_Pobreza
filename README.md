

# Reducción de Costos de Clasificación de Pobreza en Colombia
### Encuesta Mínima con Machine Learning

**Tópicos de Inteligencia Artificial · Universidad de los Andes · 2026-10**  
**Profesor:** Álvaro Riascos  
**Autores:** Jose Rincón · Lucas Rodríguez · María Paula Osuna

---

## Pregunta central

> ¿Cuántas variables observables y no manipulables bastan para clasificar
> pobreza en hogares colombianos sin encuesta de ingresos?

---

## Descripción

Este proyecto implementa un pipeline de Machine Learning para predecir
pobreza a nivel de hogar usando únicamente variables observables y
difíciles de manipular — sin preguntar por ingresos. El objetivo es
construir una **encuesta mínima** de bajo costo que pueda reemplazar
el primer filtro del SISBEN.

El pipeline selecciona las **top-8 variables** por importancia (XGBoost
Gain) y compara tres modelos (Logit, Random Forest, XGBoost) sobre
datos reales y datos sintéticos calibrados con las mismas distribuciones.

---

## Datos

- **Fuente:** DANE — Muestra Estadística de Seguridad Económica (MESE) 2018, Bogotá
- **Tamaño:** ~164,960 hogares | ~20% pobres (desbalance 4:1)
- **Variables originales:** ~200 (hogares + personas)
- **Variables de ingreso:** excluidas para evitar data leakage
- **Descarga:** [Kaggle Competition](https://www.kaggle.com/competitions/uniandes-bdml-2026-10-ps-2)

Los datos van en `00_data/00_raw/` y **no se suben al repositorio**.

---

## Estructura del proyecto

```
├── 00_rundirectory.R              # Script maestro — corre todo el pipeline
│
├── 00_data/
│   ├── 00_raw/                    # Datos originales DANE (no en repo)
│   └── 01_processed/              # Datos procesados (generados al correr)
│
├── 01_R/
│   ├── 00_prep/
│   │   └── 00_clean.R             # Limpieza y exclusión de leakage
│   ├── 01_feat/
│   │   ├── 00_features.R          # Feature engineering (18 variables proxy)
│   │   └── 00_features_min.R      # Selección top-8 por importancia XGBoost
│   └── 02_functions/
│       ├── 00_optimizar_threshold.R
│       ├── 01_guardar_modelo.R
│       └── 02_generar_submission.R
│
├── 02_models/
│   ├── log.csv                    # Registro de métricas (generado al correr)
│   ├── top8_importancia.csv       # Importancia de las top-8 variables
│   └── 00_classes/
│       ├── 01_Logit.R             # Logit — datos reales (top-8)
│       ├── 02_RandomForest.R      # Random Forest — datos reales (top-8)
│       ├── 03_XGBoost.R           # XGBoost — datos reales (top-8)
│       ├── 04_Sinteticos.R        # Datos sintéticos + 3 modelos
│       └── 05_Comparativa.R       # Tabla y gráficos finales
│
└── 04_outputs/
    ├── figures/                   # Gráficos generados
    └── tables/                    # Tablas generadas
```

---

## Cómo correr

### 1. Instalar dependencias

```r
install.packages("pacman")
pacman::p_load(here, tictoc, tidyverse, janitor, skimr,
               caret, glmnet, ranger, xgboost,
               yardstick, MLmetrics, ggplot2, patchwork)
```

### 2. Descargar datos

Descarga los datos desde Kaggle y ponlos en `00_data/00_raw/`:

```
00_data/00_raw/train_hogares.csv
00_data/00_raw/test_hogares.csv
00_data/00_raw/train_personas.csv
00_data/00_raw/test_personas.csv
```

### 3. Correr el pipeline completo

Abre `Reduciendo_Costos_Clasificación_Pobreza.Rproj` en RStudio y corre:

```r
source("00_rundirectory.R")
```

---

## Pipeline

| Paso | Script | Descripción |
|------|--------|-------------|
| 1 | `00_clean.R` | Limpieza y exclusión de variables de ingreso |
| 2 | `00_features.R` | Feature engineering — 18 variables proxy |
| 3 | `00_features_min.R` | Selección top-8 por XGBoost Gain |
| 4 | `01_Logit.R` | Regresión logística con top-8 |
| 5 | `02_RandomForest.R` | Random Forest con top-8 |
| 6 | `03_XGBoost.R` | XGBoost con top-8 |
| 7 | `04_Sinteticos.R` | Datos sintéticos + Logit + RF + XGBoost |
| 8 | `05_Comparativa.R` | Tabla y gráficos finales |

---

## Resultados

| Modelo | F1 Reales | F1 Sintéticos | Brecha |
|--------|-----------|---------------|--------|
| XGBoost | 0.666 | 0.647 | +0.019 |
| Random Forest | 0.662 | 0.638 | +0.024 |
| Logit | 0.625 | 0.653 | -0.028 |

*Métrica: F1-score | CV 5 folds | Threshold optimizado OOF*

---

## Features engineered (top-8 seleccionadas)

Las 8 variables seleccionadas por mayor Gain en XGBoost son
combinaciones observables de:

- **Hacinamiento** — personas por cuarto
- **Ratio de dependencia** — fracción del hogar sin ingreso laboral
- **Tasa de ocupación** — fracción de la PET empleada
- **Educación del jefe** — nivel educativo del jefe del hogar
- **Formalidad laboral** — cotización a pensión y salud
- **Tamaño del hogar** — número de personas
- **Jefe vulnerable** — jefe sin empleo y sin educación
- **Calidad del empleo** — horas trabajadas × formalidad

---

## Literatura de referencia

- Chen & Guestrin (2016) — XGBoost
- Corral et al. (2024) — Variables laborales y demográficas Colombia
- IDB Kaggle Costa Rica (2018) — Benchmark pobreza hogares
- Banerjee & Duflo (2011) — Poor Economics
- World Bank Poverty & Shared Prosperity (2019)

---

## Licencia

Uso académico — Universidad de los Andes 2026
```
