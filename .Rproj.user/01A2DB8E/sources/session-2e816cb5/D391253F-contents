# ============================================================
# 08_Naive_Bayes.R
# Naive Bayes models
# ============================================================

TIPO       <- "08_Naive_Bayes"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

train <- train |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Control de entrenamiento -------------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = CV_FOLDS,
  classProbs      = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final"
)

# ============================================================
# MODELO 1 — Naive Bayes default
# ============================================================
cat("\n>>> [nb - 1/3] Naive Bayes default...\n")
tic("Naive Bayes default")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "naive_bayes",
  trControl = ctrl,
  metric    = "AUC"
)

opt1      <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- "NB_default"
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
toc()

# ============================================================
# MODELO 2 — Naive Bayes con kernel
# ============================================================
cat("\n>>> [nb - 2/3] Naive Bayes kernel...\n")
tic("Naive Bayes kernel")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "naive_bayes",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    laplace    = 0,
    usekernel  = TRUE,
    adjust     = 1
  )
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- "NB_kernel"
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# MODELO 3 — Naive Bayes grid
# ============================================================
cat("\n>>> [nb - 3/3] Naive Bayes grid...\n")
tic("Naive Bayes grid")
set.seed(SEED)

m3 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "naive_bayes",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    laplace   = c(0, 0.5, 1),
    usekernel = c(TRUE, FALSE),
    adjust    = c(0.5, 1, 2)
  )
)

opt3      <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- paste0("NB_laplace_", m3$bestTune$laplace,
                    "_kernel_",    m3$bestTune$usekernel,
                    "_adjust_",    m3$bestTune$adjust)
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, TIPO, nombre_m3)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen Naive Bayes\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO)
gc()