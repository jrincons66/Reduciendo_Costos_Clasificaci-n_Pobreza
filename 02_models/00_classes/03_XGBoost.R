# ============================================================
# 03_XGBoost.R
# XGBoost — encuesta mínima (top variables)
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
# ============================================================

TIPO       <- "03_XGBoost"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

cat("\n─────────────────────────────────────────────────────────\n")
cat("  XGBoost — datos reales (top)\n")
cat("─────────────────────────────────────────────────────────\n")

# --- Cargar datos -------------------------------------------
train_min <- readRDS(here(paths$processed, "train_min.rds"))
test_min  <- readRDS(here(paths$processed, "test_min.rds"))

train_min <- train_min |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Preparar matrices XGBoost ------------------------------
dummy_recipe <- dummyVars(~ ., data = train_min |> select(-id, -pobre),
                          fullRank = TRUE)
X_train <- predict(dummy_recipe, train_min |> select(-id, -pobre))
y_train <- as.numeric(train_min$pobre == "pobre")
X_test  <- predict(dummy_recipe, test_min |> select(-id))

dtrain  <- xgb.DMatrix(data = X_train, label = y_train)
dtest   <- xgb.DMatrix(data = X_test)

# ============================================================
# MODELO 1 — XGBoost nativo grid manual
# ============================================================
cat("\n>>> [xgb - 1/2] XGBoost grid manual...\n")
tic("XGBoost grid manual")

params_grid <- expand.grid(
  max_depth = c(3, 6),
  eta       = c(0.05, 0.1),
  stringsAsFactors = FALSE
)

resultados_grid_xgb <- map(seq_len(nrow(params_grid)), function(i) {
  p <- params_grid[i, ]
  cat(sprintf("    depth=%d | eta=%.2f\n", p$max_depth, p$eta))
  
  params_i <- list(
    booster          = "gbtree",
    objective        = "binary:logistic",
    eval_metric      = "auc",
    eta              = p$eta,
    max_depth        = p$max_depth,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    min_child_weight = 1,
    nthread          = parallel::detectCores() - 1
  )
  
  set.seed(SEED)
  folds_g <- createFolds(y_train, k = CV_FOLDS,
                         list = TRUE, returnTrain = FALSE)
  oof_g   <- rep(NA_real_, length(y_train))
  
  for (k in seq_along(folds_g)) {
    val_idx <- folds_g[[k]]
    tr_idx  <- setdiff(seq_along(y_train), val_idx)
    
    if (length(unique(y_train[tr_idx]))  < 2 ||
        length(unique(y_train[val_idx])) < 2) next
    
    d_tr  <- xgb.DMatrix(data  = X_train[tr_idx,  ],
                         label = y_train[tr_idx])
    d_val <- xgb.DMatrix(data  = X_train[val_idx, ])
    
    fold_m <- xgb.train(
      params  = params_i,
      data    = d_tr,
      nrounds = 200,
      verbose = 0
    )
    
    oof_g[val_idx] <- predict(fold_m, d_val)
    rm(fold_m, d_tr, d_val)
  }
  
  # F1 solo sobre predicciones válidas
  idx_validos <- which(!is.na(oof_g))
  
  thresh_g <- seq(0.25, 0.55, by = 0.005)
  f1_g     <- map_dbl(thresh_g, function(t) {
    preds <- as.integer(oof_g[idx_validos] >= t)
    y_ref <- y_train[idx_validos]
    tp    <- sum(preds == 1 & y_ref == 1)
    fp    <- sum(preds == 1 & y_ref == 0)
    fn    <- sum(preds == 0 & y_ref == 1)
    prec  <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rec   <- if (tp + fn == 0) 0 else tp / (tp + fn)
    if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
  })
  
  list(
    params    = params_i,
    f1        = max(f1_g),
    threshold = thresh_g[which.max(f1_g)]
  )
})

# Mejor combinación
f1s_grid  <- map_dbl(resultados_grid_xgb, ~ .x$f1)
best_grid <- resultados_grid_xgb[[which.max(f1s_grid)]]

cat(sprintf("    Mejor grid — depth=%d | eta=%.2f | F1=%.4f\n",
            best_grid$params$max_depth,
            best_grid$params$eta,
            best_grid$f1))

# Modelo final con mejores params
set.seed(SEED)
m1 <- xgb.train(
  params  = best_grid$params,
  data    = dtrain,
  nrounds = 200,
  verbose = 0
)

opt1    <- list(threshold = best_grid$threshold, f1 = best_grid$f1)
nombre1 <- paste0("XGB_grid_depth_", best_grid$params$max_depth,
                  "_eta_",           best_grid$params$eta, "_top")
guardar_modelo(m1, nombre1, TIPO, dir_modelo, opt1$threshold, opt1$f1)

probs1 <- predict(m1, dtest)
preds1 <- as.integer(probs1 >= opt1$threshold)
sub1   <- data.frame(id = test_min$id, pobre = preds1)
write.csv(sub1, file.path(dir_modelo, paste0(nombre1, ".csv")),
          row.names = FALSE)
cat(sprintf("    F1: %.4f | Threshold: %.3f\n", opt1$f1, opt1$threshold))
toc()

# ============================================================
# MODELO 2 — XGBoost nativo OOF + early stopping
# ============================================================
cat("\n>>> [xgb - 2/2] XGBoost nativo OOF + early stopping...\n")
tic("XGBoost nativo OOF")

params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 6,
  gamma            = 0,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  nthread          = parallel::detectCores() - 1
)

# --- OOF predictions ----------------------------------------
cat(sprintf("    Generando OOF (%d folds)...\n", CV_FOLDS))
set.seed(SEED)
folds     <- createFolds(y_train, k = CV_FOLDS,
                         list = TRUE, returnTrain = FALSE)
oof_preds <- rep(NA_real_, length(y_train))

for (k in seq_along(folds)) {
  cat(sprintf("      Fold %d/%d\n", k, CV_FOLDS))
  val_idx <- folds[[k]]
  tr_idx  <- setdiff(seq_along(y_train), val_idx)
  
  if (length(unique(y_train[tr_idx]))  < 2 ||
      length(unique(y_train[val_idx])) < 2) next
  
  d_tr_k  <- xgb.DMatrix(data  = X_train[tr_idx,  ],
                         label = y_train[tr_idx])
  d_val_k <- xgb.DMatrix(data  = X_train[val_idx, ])
  
  fold_m  <- xgb.train(
    params  = params,
    data    = d_tr_k,
    nrounds = 500,
    verbose = 0
  )
  
  oof_preds[val_idx] <- predict(fold_m, d_val_k)
  rm(fold_m, d_tr_k, d_val_k)
}

saveRDS(oof_preds, file.path(dir_modelo, "xgb_oof_preds_top.rds"))

# --- Modelo final con early stopping ------------------------
set.seed(SEED)
d_val_full <- xgb.DMatrix(
  data  = X_train[folds[[1]], ],
  label = y_train[folds[[1]]]
)

m2 <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 500,
  evals                 = list(val = d_val_full),
  early_stopping_rounds = 30,
  verbose               = 1,
  print_every_n         = 100
)

# --- Threshold óptimo sobre OOF ----------------------------
idx_validos <- which(!is.na(oof_preds))
thresh_grid <- seq(0.25, 0.55, by = 0.005)
f1_grid     <- map_dbl(thresh_grid, function(t) {
  preds <- as.integer(oof_preds[idx_validos] >= t)
  y_ref <- y_train[idx_validos]
  tp    <- sum(preds == 1 & y_ref == 1)
  fp    <- sum(preds == 1 & y_ref == 0)
  fn    <- sum(preds == 0 & y_ref == 1)
  prec  <- if (tp + fp == 0) 0 else tp / (tp + fp)
  rec   <- if (tp + fn == 0) 0 else tp / (tp + fn)
  if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
})

opt2 <- list(
  threshold = thresh_grid[which.max(f1_grid)],
  f1        = max(f1_grid)
)
cat(sprintf("    Threshold OOF: %.3f | F1 OOF: %.4f\n",
            opt2$threshold, opt2$f1))

nombre2 <- "XGB_nativo_early_stop_top"
guardar_modelo(m2, nombre2, TIPO, dir_modelo,
               opt2$threshold, opt2$f1)

# --- Submission ---------------------------------------------
probs_test <- predict(m2, dtest)
preds_test <- as.integer(probs_test >= opt2$threshold)
sub        <- data.frame(id = test_min$id, pobre = preds_test)
write.csv(sub, file.path(dir_modelo, paste0(nombre2, ".csv")),
          row.names = FALSE)
cat("    Submission guardada:", nombre2, ".csv\n")

# --- Importancia de variables -------------------------------
imp_xgb <- xgb.importance(feature_names = colnames(X_train),
                          model         = m2)
cat("\n    Importancia de variables (XGBoost nativo):\n")
print(imp_xgb)

p_imp_xgb <- imp_xgb |>
  mutate(Feature = reorder(Feature, Gain)) |>
  ggplot(aes(x = Feature, y = Gain)) +
  geom_col(fill = "#C8972B", width = 0.7) +
  coord_flip() +
  labs(
    title   = "Importancia XGBoost — encuesta mínima",
    x       = NULL,
    y       = "Gain",
    caption = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", color = "#1B3A6B")
  )

ggsave(here(paths$figures, "xgb_importancia_top.png"),
       plot = p_imp_xgb, width = 7, height = 4, dpi = 150)

toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n>>> Resumen XGBoost:\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# Guardar mejor resultado para comparativa
mejor_xgb <- list(
  modelo    = "XGBoost",
  tipo      = TIPO,
  f1        = max(opt1$f1, opt2$f1),
  threshold = if (opt1$f1 >= opt2$f1) opt1$threshold else opt2$threshold,
  datos     = "reales"
)
saveRDS(mejor_xgb, here(paths$processed, "mejor_xgb.rds"))

# --- Limpiar entorno ----------------------------------------
rm(train_min, test_min, dummy_recipe,
   X_train, y_train, X_test, dtrain, dtest,
   d_val_full, params, folds, oof_preds,
   thresh_grid, f1_grid,
   probs_test, preds_test, sub,
   imp_xgb, p_imp_xgb,
   params_grid, resultados_grid_xgb, f1s_grid, best_grid,
   probs1, preds1, sub1,
   m1, m2, opt1, opt2, nombre1, nombre2,
   k, val_idx, tr_idx, dir_modelo, TIPO)
gc()
