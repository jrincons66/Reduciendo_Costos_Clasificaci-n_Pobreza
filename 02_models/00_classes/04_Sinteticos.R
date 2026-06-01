# ============================================================
# 04_Sinteticos.R
# Datos sintéticos - generación y modelos
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
# ============================================================

TIPO       <- "04_Sinteticos"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

cat("\n─────────────────────────────────────────────────────────\n")
cat("  Datos sintéticos - generación y modelos\n")
cat("─────────────────────────────────────────────────────────\n")

# --- Cargar top ---------------------------------------------
top_vars   <- readRDS(here(paths$processed, "top_vars.rds"))
train_real <- readRDS(here(paths$processed, "train_min.rds"))

MIN_ENG_VARS <- c(
  "ratio_dependencia",
  "hacinamiento",
  "nper_sq",
  "edad_prom_sq",
  "calidad_empleo",
  "doble_proteccion"
)

cat("\n>>> Variables top a simular:\n")
cat("   ", paste(top_vars, collapse = ", "), "\n")

# ============================================================
# PASO 1 - Generar datos sintéticos
# ============================================================

cat("\n>>> [1/3] Generando dataset sintético...\n")

N_SINT <- 50000
set.seed(SEED)

# --- Variables base -----------------------------------------
nper            <- pmax(1, pmin(10, rpois(N_SINT, lambda = 3.2)))
p5000           <- pmax(1, round(rnorm(N_SINT, mean = 3.1, sd = 1.2)))
p5010           <- pmax(1, pmin(p5000, round(rnorm(N_SINT, mean = 1.8, sd = 0.8))))
tasa_ocupacion  <- pmin(1, pmax(0, rnorm(N_SINT, mean = 0.52, sd = 0.28)))
n_ocupados      <- pmin(nper, pmax(0, round(tasa_ocupacion * nper)))
n_pet           <- pmax(n_ocupados, round(nper * 0.75))
n_inactivos     <- pmax(0, n_pet - n_ocupados)
educ_jefe       <- sample(1:6, N_SINT, replace = TRUE,
                          prob = c(0.05, 0.30, 0.25, 0.20, 0.15, 0.05))
ocup_jefe       <- rbinom(N_SINT, 1, prob = 0.65)
nivel_educ_max  <- factor(
  pmin(6, educ_jefe + sample(0:2, N_SINT, replace = TRUE,
                             prob = c(0.6, 0.3, 0.1))),
  levels = 1:6
)
prop_mujeres        <- pmin(1, pmax(0, rnorm(N_SINT, 0.51, 0.18)))
jefe_mujer          <- rbinom(N_SINT, 1, prob = 0.38)
edad_jefe           <- pmax(18, pmin(90, round(rnorm(N_SINT, 45, 14))))
edad_promedio       <- pmax(5,  pmin(80, round(rnorm(N_SINT, 32, 12))))
edad_min            <- pmax(1, pmin(edad_promedio, round(rnorm(N_SINT, 12, 8))))
n_mayores_65        <- rbinom(N_SINT, 1, prob = 0.18)
prop_cotiza_pension <- pmin(1, pmax(0, rnorm(N_SINT, 0.38, 0.30)))
prop_afiliado_salud <- pmin(1, pmax(0, rnorm(N_SINT, 0.72, 0.25)))
horas_trabajo_prom  <- pmax(0, rnorm(N_SINT, 42, 15)) * ocup_jefe
clase               <- factor(sample(c("1", "2"), N_SINT,
                                     replace = TRUE,
                                     prob    = c(0.78, 0.22)))
prop_reg_subsidiado <- pmin(1, pmax(0, rnorm(N_SINT, 0.25, 0.22)))
p5090               <- factor(sample(c(1, 2, 3, 4), N_SINT,
                                     replace = TRUE,
                                     prob = c(0.60, 0.24, 0.12, 0.04)))
depto               <- factor(sample(levels(train_real$depto), N_SINT, replace = TRUE))

# --- Features engineered ------------------------------------
sint_df <- data.frame(
  prop_reg_subsidiado, nper, n_ocupados, p5000, p5090,
  horas_trabajo_prom, n_menores_18 = pmax(0, round(rnorm(N_SINT, 1.3, 1.4))),
  edad_promedio, prop_cotiza_pension, depto, edad_min,
  tasa_ocupacion, n_pet, n_inactivos, educ_jefe, ocup_jefe,
  nivel_educ_max, prop_mujeres, jefe_mujer,
  edad_jefe, n_mayores_65, prop_afiliado_salud,
  p5010, clase
) |>
  mutate(
    ratio_dependencia    = (nper - n_ocupados) / pmax(nper, 1),
    hacinamiento         = nper / pmax(p5000, 1),
    educ_x_ocup          = as.integer(nivel_educ_max) * tasa_ocupacion,
    rural_x_ocup         = as.integer(clase == "2") * tasa_ocupacion,
    formal_x_salud       = prop_cotiza_pension * prop_afiliado_salud,
    nper_sq              = nper^2,
    edad_prom_sq         = edad_promedio^2,
    mujeres_x_inact      = prop_mujeres * (1 - tasa_ocupacion),
    jefe_mujer_inact     = jefe_mujer * (1 - tasa_ocupacion),
    tasa_inactivos       = n_inactivos / pmax(n_pet, 1),
    sin_ocupados         = as.integer(n_ocupados == 0),
    educ_jefe_x_ocup     = as.integer(educ_jefe) * ocup_jefe,
    calidad_empleo       = horas_trabajo_prom * prop_cotiza_pension,
    presion_habitacional = p5010 / pmax(p5000, 1),
    jefe_vulnerable      = as.integer(ocup_jefe == 0 & educ_jefe <= 3),
    doble_proteccion     = prop_cotiza_pension^2,
    ratio_mayores_65     = n_mayores_65 / pmax(n_pet, 1),
    jefe_mayor_inactivo  = as.integer(edad_jefe > 60 & ocup_jefe == 0)
  )

# --- Outcome con DGP conocido -------------------------------
logit_score <- with(sint_df,
                    -2.5                       +
                      2.8  * ratio_dependencia   +
                      0.9  * hacinamiento        +
                      -1.2 * educ_x_ocup         +
                      0.6  * rural_x_ocup        +
                      -1.8 * formal_x_salud      +
                      1.1  * sin_ocupados        +
                      -0.8 * calidad_empleo / 50 +
                      1.5  * jefe_vulnerable     +
                      rnorm(N_SINT, 0, 0.8)
)

prob_pobre <- 1 / (1 + exp(-logit_score))
pobre_sint <- rbinom(N_SINT, 1, prob = prob_pobre)

cat(sprintf("    Hogares: %d | Tasa pobreza simulada: %.1f%%\n",
            N_SINT, mean(pobre_sint) * 100))

# Quedarse solo con las variables mínimas + id + pobre
sint_df <- sint_df |>
  select(any_of(c(top_vars, MIN_ENG_VARS))) |>
  mutate(
    id    = paste0("SINT_", seq_len(N_SINT)),
    pobre = pobre_sint
  ) |>
  select(id, pobre, everything())

# Split 80/20
set.seed(SEED)
idx_train  <- sample(seq_len(N_SINT), size = floor(0.8 * N_SINT))
sint_train <- sint_df[ idx_train, ]
sint_test  <- sint_df[-idx_train, ]

saveRDS(sint_train, here(paths$processed, "sint_train.rds"))
saveRDS(sint_test,  here(paths$processed, "sint_test.rds"))

cat("    sint_train:", nrow(sint_train), "x", ncol(sint_train), "\n")
cat("    sint_test: ", nrow(sint_test),  "x", ncol(sint_test),  "\n")

# ============================================================
# PASO 2 - Modelos sobre datos sintéticos
# ============================================================

cat("\n>>> [2/3] Entrenando modelos sobre datos sintéticos...\n")

sint_train <- sint_train |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

ctrl <- trainControl(
  method          = "cv",
  number          = CV_FOLDS,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# ---- 2a. Logit ---------------------------------------------
cat("\n    [2a] Logit sintético...\n")
tic("Logit sint")
set.seed(SEED)

m_logit <- train(
  pobre ~ .,
  data       = sint_train |> select(-id),
  method     = "glm",
  family     = binomial(link = "logit"),
  trControl  = ctrl,
  metric     = "ROC",
  preProcess = c("center", "scale")
)

opt_logit <- optimizar_threshold(m_logit, sint_train, sint_train$pobre)
nombre_l  <- "logit_sintetico_top"
guardar_modelo(m_logit, nombre_l, TIPO, dir_modelo,
               opt_logit$threshold, opt_logit$f1)
cat(sprintf("    Logit sint - F1: %.4f | Threshold: %.3f\n",
            opt_logit$f1, opt_logit$threshold))
toc()

# ---- 2b. Random Forest -------------------------------------
cat("\n    [2b] Random Forest sintético...\n")
tic("RF sint")
set.seed(SEED)

p_sint    <- length(setdiff(names(sint_train), c("id", "pobre")))
mtry_sint <- max(1, floor(sqrt(p_sint)))

m_rf <- ranger(
  pobre         ~ .,
  data          = sint_train |> select(-id),
  num.trees     = 1000,
  mtry          = mtry_sint,
  splitrule     = "gini",
  min.node.size = 1,
  probability   = TRUE,
  importance    = "permutation",
  num.threads   = parallel::detectCores() - 1,
  seed          = SEED
)

opt_rf   <- optimizar_threshold(m_rf, NULL, sint_train$pobre)
nombre_r <- "rf_sintetico_top"
guardar_modelo(m_rf, nombre_r, TIPO, dir_modelo,
               opt_rf$threshold, opt_rf$f1)
cat(sprintf("    RF sint - OOB F1: %.4f | Threshold: %.3f\n",
            opt_rf$f1, opt_rf$threshold))
toc()

# ---- 2c. XGBoost -------------------------------------------
cat("\n    [2c] XGBoost sintético...\n")
tic("XGB sint")

dummy_sint  <- dummyVars(~ ., data = sint_train |> select(-id, -pobre),
                         fullRank = TRUE)
X_sint      <- predict(dummy_sint, sint_train |> select(-id, -pobre))
y_sint      <- as.numeric(sint_train$pobre == "pobre")
dtrain_sint <- xgb.DMatrix(data = X_sint, label = y_sint)

params_sint <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  nthread          = parallel::detectCores() - 1
)

set.seed(SEED)
folds_sint <- createFolds(y_sint, k = CV_FOLDS,
                          list = TRUE, returnTrain = FALSE)
oof_sint   <- rep(NA_real_, length(y_sint))

for (k in seq_along(folds_sint)) {
  cat(sprintf("      Fold %d/%d\n", k, CV_FOLDS))
  val_idx <- folds_sint[[k]]
  tr_idx  <- setdiff(seq_along(y_sint), val_idx)
  
  if (length(unique(y_sint[tr_idx]))  < 2 ||
      length(unique(y_sint[val_idx])) < 2) next
  
  d_tr_k  <- xgb.DMatrix(data  = X_sint[tr_idx,  ],
                         label = y_sint[tr_idx])
  d_val_k <- xgb.DMatrix(data  = X_sint[val_idx, ])
  
  fold_m  <- xgb.train(
    params  = params_sint,
    data    = d_tr_k,
    nrounds = 300,
    verbose = 0
  )
  
  oof_sint[val_idx] <- predict(fold_m, d_val_k)
  rm(fold_m, d_tr_k, d_val_k)
}

# Threshold óptimo
idx_val_sint <- which(!is.na(oof_sint))
thresh_sint  <- seq(0.25, 0.55, by = 0.005)
f1_sint      <- map_dbl(thresh_sint, function(t) {
  preds <- as.integer(oof_sint[idx_val_sint] >= t)
  y_ref <- y_sint[idx_val_sint]
  tp    <- sum(preds == 1 & y_ref == 1)
  fp    <- sum(preds == 1 & y_ref == 0)
  fn    <- sum(preds == 0 & y_ref == 1)
  prec  <- if (tp + fp == 0) 0 else tp / (tp + fp)
  rec   <- if (tp + fn == 0) 0 else tp / (tp + fn)
  if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
})

opt_xgb  <- list(threshold = thresh_sint[which.max(f1_sint)],
                 f1        = max(f1_sint))

set.seed(SEED)
m_xgb <- xgb.train(
  params  = params_sint,
  data    = dtrain_sint,
  nrounds = 300,
  verbose = 0
)

nombre_x <- "xgb_sintetico_top"
guardar_modelo(m_xgb, nombre_x, TIPO, dir_modelo,
               opt_xgb$threshold, opt_xgb$f1)
cat(sprintf("    XGB sint - OOF F1: %.4f | Threshold: %.3f\n",
            opt_xgb$f1, opt_xgb$threshold))
toc()

# ============================================================
# PASO 3 - Guardar resultados
# ============================================================

cat("\n>>> [3/3] Guardando resultados sintéticos...\n")

resultados_sint <- data.frame(
  modelo    = c("Logit", "Random Forest", "XGBoost"),
  f1        = round(c(opt_logit$f1, opt_rf$f1, opt_xgb$f1), 4),
  threshold = round(c(opt_logit$threshold,
                      opt_rf$threshold,
                      opt_xgb$threshold), 3),
  datos     = "sinteticos",
  stringsAsFactors = FALSE
)

saveRDS(resultados_sint,
        here(paths$processed, "resultados_sint.rds"))

cat("\n>>> Resumen modelos sintéticos:\n")
print(resultados_sint)

# --- Limpiar entorno ----------------------------------------
rm(sint_df, sint_train, sint_test, top_vars, train_real,
   nper, p5000, p5010, tasa_ocupacion, n_ocupados,
   n_pet, n_inactivos, educ_jefe, ocup_jefe, nivel_educ_max,
   prop_mujeres, jefe_mujer, edad_jefe, edad_promedio,
   edad_min, n_mayores_65, prop_cotiza_pension, prop_afiliado_salud,
   horas_trabajo_prom, clase, prop_reg_subsidiado, p5090, depto,
   logit_score, prob_pobre, pobre_sint, idx_train, ctrl,
   dummy_sint, X_sint, y_sint, dtrain_sint, params_sint,
   folds_sint, oof_sint, idx_val_sint, thresh_sint, f1_sint,
   m_logit, m_rf, m_xgb,
   opt_logit, opt_rf, opt_xgb,
   nombre_l, nombre_r, nombre_x,
   p_sint, mtry_sint, resultados_sint,
   k, val_idx, tr_idx,
   dir_modelo, TIPO, N_SINT, MIN_ENG_VARS)
gc()

cat("\n>>> 04_Sinteticos.R completado\n")
