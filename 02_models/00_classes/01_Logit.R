# ============================================================
# 01_Logit.R
# Regresión Logística — encuesta mínima (top variables)
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
# ============================================================

TIPO       <- "01_Logit"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

cat("\n─────────────────────────────────────────────────────────\n")
cat("  Logit — datos reales (top)\n")
cat("─────────────────────────────────────────────────────────\n")

# --- Cargar datos -------------------------------------------
train_min <- readRDS(here(paths$processed, "train_min.rds"))
test_min  <- readRDS(here(paths$processed, "test_min.rds"))

train_min <- train_min |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Control CV ---------------------------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = CV_FOLDS,
  classProbs      = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final"
)

# ============================================================
# MODELO 1 — Logit baseline
# ============================================================
cat("\n>>> [logit - 1/2] Baseline...\n")
tic("Logit baseline")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train_min |> select(-id),
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "AUC"
)

opt1    <- optimizar_threshold(m1, train_min, train_min$pobre)
nombre1 <- "logit_baseline_top"
guardar_modelo(m1, nombre1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test_min, opt1$threshold, TIPO, nombre1)
cat(sprintf("    F1: %.4f | Threshold: %.3f\n", opt1$f1, opt1$threshold))
toc()

# ============================================================
# MODELO 2 — Logit con preprocesamiento
# ============================================================
cat("\n>>> [logit - 2/2] Preprocesado...\n")
tic("Logit preprocesado")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data       = train_min |> select(-id),
  method     = "glm",
  family     = binomial(link = "logit"),
  trControl  = ctrl,
  metric     = "AUC",
  preProcess = c("center", "scale")
)

opt2    <- optimizar_threshold(m2, train_min, train_min$pobre)
nombre2 <- "logit_scaled_top"
guardar_modelo(m2, nombre2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test_min, opt2$threshold, TIPO, nombre2)
cat(sprintf("    F1: %.4f | Threshold: %.3f\n", opt2$f1, opt2$threshold))
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n>>> Resumen Logit:\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# Guardar mejor resultado para comparativa
mejor_logit <- list(
  modelo    = "Logit",
  tipo      = TIPO,
  f1        = max(opt1$f1, opt2$f1),
  threshold = if (opt1$f1 >= opt2$f1) opt1$threshold else opt2$threshold,
  datos     = "reales"
)
saveRDS(mejor_logit, here(paths$processed, "mejor_logit.rds"))

# --- Limpiar entorno ----------------------------------------
rm(train_min, test_min, ctrl,
   m1, m2, opt1, opt2, nombre1, nombre2,
   dir_modelo, TIPO)
gc()
