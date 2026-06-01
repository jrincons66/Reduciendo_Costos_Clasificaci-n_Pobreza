# ============================================================
# 00_rundirectory.R
# Pipeline maestro — Reducción de Costos de Clasificación
# de Pobreza en Colombia mediante Machine Learning
# ============================================================
#
# Tópicos de Inteligencia Artificial
# Universidad de los Andes | 2026-10
# Profesor: Álvaro Riascos
#
# Autores:
#   · Jose Rincón
#   · Lucas Rodríguez
#   · María Paula Osuna
#
# Pregunta central:
#   ¿Cuántas variables observables y no manipulables bastan
#   para clasificar pobreza en hogares colombianos sin
#   encuesta de ingresos?
#
# Pipeline:
#   [1] Limpieza y exclusión de leakage
#   [2] Feature engineering (18 variables proxy)
#   [3] Selección top variables por importancia XGBoost
#   [4] Modelos con top — datos reales
#   [5] Datos sintéticos + mismos modelos
#   [6] Comparativa final
#
# Datos: DANE MESE 2018, Bogotá
# Métrica: F1-score (CV 5 folds, threshold OOF)
# ============================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║   Tópicos de IA · Universidad de los Andes · 2026-10    ║\n")
cat("║   Profesor: Álvaro Riascos                               ║\n")
cat("║                                                          ║\n")
cat("║   Reducción de Costos de Clasificación de Pobreza        ║\n")
cat("║   en Colombia — Encuesta Mínima con ML                   ║\n")
cat("║                                                          ║\n")
cat("║   Jose Rincón · Lucas Rodríguez · María Paula Osuna      ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n")
cat(sprintf("  Inicio: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

# ============================================================
# PAQUETES
# ============================================================

if (!require("pacman", quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  # Entorno
  here, tictoc,
  
  # Manipulación
  tidyverse, janitor, skimr,
  
  # Modelado
  caret, glmnet, ranger, xgboost,
  
  # Métricas
  yardstick, MLmetrics,
  
  # Visualización
  ggplot2, patchwork
)

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================

SEED     <- 202601
CV_FOLDS <- 5
TOP_N    <- 8       # número de variables encuesta mínima

set.seed(SEED)

# ============================================================
# RUTAS
# ============================================================

paths <- list(
  root        = here::here(),
  raw         = here("00_data", "00_raw"),
  processed   = here("00_data", "01_processed"),
  prep        = here("01_R", "00_prep"),
  feat        = here("01_R", "01_feat"),
  functions   = here("01_R", "02_functions"),
  models      = here("02_models"),
  classes     = here("02_models", "00_classes"),
  submissions = here("02_models", "01_submissions"),
  figures     = here("04_outputs", "figures"),
  tables      = here("04_outputs", "tables")
)

invisible(lapply(paths, dir.create, recursive = TRUE, showWarnings = FALSE))

invisible(lapply(
  c("01_Logit", "02_RandomForest", "03_XGBoost",
    "04_Sinteticos", "05_Comparativa"),
  function(d) dir.create(file.path(paths$submissions, d),
                         recursive = TRUE, showWarnings = FALSE)
))

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

source(here(paths$functions, "00_optimizar_threshold.R"))
source(here(paths$functions, "01_guardar_modelo.R"))
source(here(paths$functions, "02_generar_submission.R"))

# ============================================================
# PIPELINE
# ============================================================

tic("Pipeline completo")

# --- [1] Limpieza -------------------------------------------
cat("─────────────────────────────────────────────────────────\n")
cat("  [1/6] Limpieza de datos y exclusión de ingresos\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Limpieza")
source(here(paths$prep, "00_clean.R"))
toc(log = TRUE)

# --- [2] Feature engineering --------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [2/6] Feature engineering — 18 variables proxy\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Features")
source(here(paths$feat, "00_features.R"))
toc(log = TRUE)

# --- [3] Selección top-8 ------------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [3/6] Selección top por importancia XGBoost\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Top-8")
source(here(paths$feat, "00_features_min.R"))
toc(log = TRUE)

# --- [4] Modelos datos reales -------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [4/6] Modelos con top variables — datos reales\n")
cat("        Logit · Random Forest · XGBoost\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Modelos reales")
source(here(paths$classes, "01_Logit.R"))
source(here(paths$classes, "02_Random_Forest.R"))
source(here(paths$classes, "03_XGBoost.R"))
toc(log = TRUE)

# --- [5] Datos sintéticos -----------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [5/6] Datos sintéticos + modelos\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Sintéticos")
source(here(paths$classes, "04_Sinteticos.R"))
toc(log = TRUE)

# --- [6] Comparativa final ----------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [6/6] Comparativa final\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Comparativa")
source(here(paths$classes, "05_Comparativa.R"))
toc(log = TRUE)

# ============================================================
# RESUMEN FINAL
# ============================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║   Pipeline completado                                    ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n")
cat("  Tiempos por etapa:\n")
tic.log(format = TRUE) |> unlist() |> cat(sep = "\n")
cat("\n")

if (file.exists(here(paths$tables, "comparativa_final.csv"))) {
  cat("\n>>> Comparativa final:\n")
  print(read.csv(here(paths$tables, "comparativa_final.csv")))
}

toc()
cat(sprintf("\n  Fin: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
