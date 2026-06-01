# ============================================================
# 02_RandomForest.R
# Random Forest — encuesta mínima (top variables)
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
# ============================================================

TIPO       <- "02_RandomForest"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

cat("\n─────────────────────────────────────────────────────────\n")
cat("  Random Forest — datos reales (top)\n")
cat("─────────────────────────────────────────────────────────\n")

# --- Cargar datos -------------------------------------------
train_min <- readRDS(here(paths$processed, "train_min.rds"))
test_min  <- readRDS(here(paths$processed, "test_min.rds"))

train_min <- train_min |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

p        <- ncol(train_min) - 2   # columnas sin id y pobre
mtry_def <- max(1, floor(sqrt(p)))

# ============================================================
# MODELO 1 — RF default
# ============================================================
cat("\n>>> [rf - 1/3] RF default...\n")
tic("RF default")
set.seed(SEED)

m1 <- ranger(
  pobre         ~ .,
  data          = train_min |> select(-id),
  num.trees     = 1000,
  mtry          = mtry_def,
  splitrule     = "gini",
  min.node.size = 1,
  probability   = TRUE,
  importance    = "permutation",
  num.threads   = parallel::detectCores() - 1,
  seed          = SEED
)

opt1    <- optimizar_threshold(m1, NULL, train_min$pobre)
nombre1 <- paste0("RF_default_mtry_", mtry_def, "_top")
guardar_modelo(m1, nombre1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test_min, opt1$threshold, TIPO, nombre1)
cat(sprintf("    OOB F1: %.4f | Threshold: %.3f | Brier: %.4f\n",
            opt1$f1, opt1$threshold, m1$prediction.error))
toc()

# ============================================================
# MODELO 2 — RF grid (mtry, node size, splitrule)
# ============================================================
cat("\n>>> [rf - 2/3] RF grid...\n")
tic("RF grid")
set.seed(SEED)

grid_m2 <- expand.grid(
  num.trees     = 1000,
  mtry          = c(mtry_def,
                    max(1, floor(mtry_def * 1.5)),
                    p),
  min.node.size = c(1, 5, 10),
  splitrule     = c("gini", "hellinger", "extratrees"),
  stringsAsFactors = FALSE
)

resultados_grid <- map(seq_len(nrow(grid_m2)), function(i) {
  g   <- grid_m2[i, ]
  fit <- ranger(
    pobre         ~ .,
    data          = train_min |> select(-id),
    num.trees     = g$num.trees,
    mtry          = g$mtry,
    splitrule     = g$splitrule,
    min.node.size = g$min.node.size,
    probability   = TRUE,
    num.threads   = parallel::detectCores() - 1,
    seed          = SEED
  )
  opt <- optimizar_threshold(fit, NULL, train_min$pobre)
  cat(sprintf("    mtry=%d | node=%d | rule=%-10s | F1=%.4f\n",
              g$mtry, g$min.node.size, g$splitrule, opt$f1))
  list(model = fit, opt = opt)
})

# Seleccionar mejor
f1s   <- map_dbl(resultados_grid, ~ .x$opt$f1)
best2 <- resultados_grid[[which.max(f1s)]]

nombre2 <- paste0("RF_grid_",
                  best2$model$splitrule, "_mtry_",
                  best2$model$mtry,     "_node_",
                  best2$model$min.node.size, "_top")
guardar_modelo(best2$model, nombre2, TIPO, dir_modelo,
               best2$opt$threshold, best2$opt$f1)
generar_submission(best2$model, test_min,
                   best2$opt$threshold, TIPO, nombre2)
cat(sprintf("    Mejor grid — F1: %.4f | Threshold: %.3f\n",
            best2$opt$f1, best2$opt$threshold))
toc()

# ============================================================
# MODELO 3 — RF Hellinger + bajo overfitting
# ============================================================
cat("\n>>> [rf - 3/3] RF Hellinger + bajo overfitting...\n")
tic("RF Hellinger")
set.seed(SEED)

m3 <- ranger(
  pobre         ~ .,
  data          = train_min |> select(-id),
  num.trees     = 1000,
  mtry          = p,
  splitrule     = "hellinger",
  min.node.size = 10,
  probability   = TRUE,
  importance    = "permutation",
  num.threads   = parallel::detectCores() - 1,
  seed          = SEED
)

opt3    <- optimizar_threshold(m3, NULL, train_min$pobre)
nombre3 <- "RF_hellinger_bagging_top"
guardar_modelo(m3, nombre3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test_min, opt3$threshold, TIPO, nombre3)
cat(sprintf("    OOB F1: %.4f | Threshold: %.3f | Brier: %.4f\n",
            opt3$f1, opt3$threshold, m3$prediction.error))

# Importancia de variables
imp_rf <- data.frame(
  variable   = names(m3$variable.importance),
  importance = m3$variable.importance
) |> arrange(desc(importance))
cat("\n    Importancia de variables (RF Hellinger):\n")
print(imp_rf)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n>>> Resumen Random Forest:\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# Guardar mejor resultado/ para comparativa
todos_f1  <- c(opt1$f1, best2$opt$f1, opt3$f1)
mejor_idx <- which.max(todos_f1)

mejor_rf <- list(
  modelo    = "Random Forest",
  tipo      = TIPO,
  f1        = todos_f1[mejor_idx],
  threshold = list(opt1$threshold,
                   best2$opt$threshold,
                   opt3$threshold)[[mejor_idx]],
  datos     = "reales"
)
saveRDS(mejor_rf, here(paths$processed, "mejor_rf.rds"))

# --- Limpiar entorno ----------------------------------------
rm(train_min, test_min, p, mtry_def, grid_m2,
   resultados_grid, f1s, best2, todos_f1, mejor_idx,
   m1, m3, opt1, opt3, nombre1, nombre2, nombre3,
   imp_rf, dir_modelo, TIPO)
gc()
