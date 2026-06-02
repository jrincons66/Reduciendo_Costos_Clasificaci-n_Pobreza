# ============================================================
# 05_comparativa.R
# ============================================================

TIPO <- "05_Comparativa"

cat("\n─────────────────────────────────────────────────────────\n")
cat("  Comparativa final — reales vs sintéticos\n")
cat("─────────────────────────────────────────────────────────\n")

# ---- Paleta institucional ----------------------------------
COL_REAL  <- "#1B3A6B"
COL_SINT  <- "#C8972B"
COL_GREY  <- "#888888"
COL_LIGHT <- "#D6E4F0"

# ============================================================
# HELPERS
# ============================================================
library(kableExtra)
library(knitr)

guardar_tabla_tex <- function(df, nombre_archivo, caption = NULL, label = NULL) {
  
  kbl(
    df,
    format = "latex",
    booktabs = TRUE,
    longtable = FALSE,
    caption = caption,
    label = label,
    digits = 4
  ) |>
    kable_styling(
      latex_options = c("hold_position")
    ) |>
    save_kable(
      file = here(paths$tables, nombre_archivo)
    )
}

calcular_metricas <- function(y_real, probs, threshold, modelo_nm, datos_nm) {
  preds <- as.integer(probs >= threshold)
  y_bin <- as.integer(y_real == "pobre" | y_real == 1)
  
  tp <- sum(preds == 1 & y_bin == 1)
  fp <- sum(preds == 1 & y_bin == 0)
  fn <- sum(preds == 0 & y_bin == 1)
  tn <- sum(preds == 0 & y_bin == 0)
  n  <- length(y_bin)
  
  prec    <- if (tp + fp == 0) NA else tp / (tp + fp)
  rec     <- if (tp + fn == 0) NA else tp / (tp + fn)
  f1      <- if (is.na(prec) || is.na(rec) || prec + rec == 0) NA else
    2 * prec * rec / (prec + rec)
  spec    <- tn / (tn + fp)
  acc     <- (tp + tn) / n
  fpr     <- fp / (fp + tn)          # tasa falsos positivos
  fnr     <- fn / (fn + tp)          # tasa falsos negativos (clave PP)
  brier   <- mean((probs - y_bin)^2)
  
  # AUC via trapecio simple
  ord     <- order(probs, decreasing = TRUE)
  y_s     <- y_bin[ord]
  tpr_v   <- cumsum(y_s) / sum(y_s)
  fpr_v   <- cumsum(1 - y_s) / sum(1 - y_s)
  auc_val <- sum(diff(c(0, fpr_v)) * (c(0, tpr_v[-length(tpr_v)]) + tpr_v) / 2)
  
  data.frame(
    modelo    = modelo_nm,
    datos     = datos_nm,
    threshold = round(threshold, 3),
    f1        = round(f1,    4),
    precision = round(prec,  4),
    recall    = round(rec,   4),
    auc       = round(auc_val, 4),
    specificity = round(spec, 4),
    accuracy  = round(acc,   4),
    fnr       = round(fnr,   4),  # falsos negativos (pobres no detectados)
    fpr       = round(fpr,   4),
    brier     = round(brier, 4),
    tp = tp, fp = fp, fn = fn, tn = tn,
    n  = n,
    stringsAsFactors = FALSE
  )
}

calcular_pr_curve <- function(y_real, probs, modelo_nm, datos_nm) {
  y_bin <- as.integer(y_real == "pobre" | y_real == 1)
  ths   <- seq(0.01, 0.99, by = 0.01)
  pr <- map_dfr(ths, function(t) {
    p  <- as.integer(probs >= t)
    tp <- sum(p == 1 & y_bin == 1)
    fp <- sum(p == 1 & y_bin == 0)
    fn <- sum(p == 0 & y_bin == 1)
    prec <- if (tp + fp == 0) NA else tp / (tp + fp)
    rec  <- if (tp + fn == 0) 0  else tp / (tp + fn)
    data.frame(threshold = t, precision = prec, recall = rec)
  })
  pr$modelo <- modelo_nm
  pr$datos  <- datos_nm
  pr
}

calcular_f1_threshold <- function(y_real, probs, modelo_nm, datos_nm) {
  y_bin <- as.integer(y_real == "pobre" | y_real == 1)
  ths   <- seq(0.10, 0.90, by = 0.005)
  map_dfr(ths, function(t) {
    p  <- as.integer(probs >= t)
    tp <- sum(p == 1 & y_bin == 1)
    fp <- sum(p == 1 & y_bin == 0)
    fn <- sum(p == 0 & y_bin == 1)
    prec <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rec  <- if (tp + fn == 0) 0 else tp / (tp + fn)
    f1   <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
    data.frame(threshold = t, f1 = f1, precision = prec, recall = rec,
               modelo = modelo_nm, datos = datos_nm)
  })
}

# ============================================================
# PASO 1 — Cargar resultados del log y modelos guardados
# ============================================================

cat("\n>>> [1/8] Cargando resultados y modelos...\n")

log_cv <- read.csv(here(paths$models, "log.csv")) |>
  group_by(tipo, modelo) |>
  slice_max(order_by = timestamp, n = 1) |>
  ungroup()

write.csv(log_cv, here(paths$models, "log.csv"), row.names = FALSE)

mejores_reales <- log_cv |>
  filter(tipo %in% c("01_Logit", "02_RandomForest", "03_XGBoost")) |>
  group_by(tipo) |>
  slice_max(order_by = cv_f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(
    modelo = case_when(
      tipo == "01_Logit"        ~ "Logit",
      tipo == "02_RandomForest" ~ "Random Forest",
      tipo == "03_XGBoost"      ~ "XGBoost"
    ),
    datos = "Reales (DANE)"
  ) |>
  select(modelo, f1 = cv_f1, threshold, datos)

resultados_sint <- readRDS(here(paths$processed, "resultados_sint.rds")) |>
  mutate(datos = "Sintéticos") |>
  select(modelo, f1, threshold, datos)

# Cargar datos de validación para cálculos extendidos
train_min  <- readRDS(here(paths$processed, "train_min.rds"))
sint_train <- readRDS(here(paths$processed, "sint_train.rds"))
sint_test  <- readRDS(here(paths$processed, "sint_test.rds"))

train_min <- train_min |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# Cargar OOF predictions de los modelos reales
# (generados en los scripts 01-03; necesarios para métricas honestas)
oof_xgb_real <- tryCatch(
  readRDS(here(paths$submissions, "03_XGBoost", "xgb_oof_preds_top8.rds")),
  error = function(e) NULL
)

# ============================================================
# PASO 2 — Tabla comparativa base
# ============================================================

cat("\n>>> [2/8] Tabla comparativa base...\n")

comparativa <- bind_rows(mejores_reales, resultados_sint) |>
  mutate(
    datos     = factor(datos, levels = c("Reales (DANE)", "Sintéticos")),
    f1        = round(f1, 4),
    threshold = round(threshold, 3)
  ) |>
  arrange(datos, desc(f1))

comparativa_wide <- comparativa |>
  group_by(modelo, datos) |>
  slice_max(order_by = f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(modelo, f1, datos) |>
  pivot_wider(names_from = datos, values_from = f1, values_fn = max) |>
  mutate(
    diferencia   = round(`Reales (DANE)` - `Sintéticos`, 4),
    gap_relativo = round(diferencia / `Reales (DANE)` * 100, 2)
  ) |>
  arrange(desc(`Reales (DANE)`))

write.csv(comparativa,      here(paths$tables, "comparativa_final.csv"),   row.names = FALSE)
write.csv(comparativa_wide, here(paths$tables, "comparativa_wide.csv"),    row.names = FALSE)

guardar_tabla_tex(
  comparativa,
  "comparativa_final.tex",
  caption = "Comparativa de desempeño entre modelos entrenados con datos reales y sintéticos.",
  label = "tab:comparativa_final"
)

guardar_tabla_tex(
  comparativa_wide,
  "comparativa_wide.tex",
  caption = "Brecha de desempeño entre modelos entrenados con datos reales y sintéticos.",
  label = "tab:comparativa_wide"
)

cat("    CSVs base guardados.\n")

# ============================================================
# [A] MÉTRICAS COMPLETAS
# ============================================================

cat("\n>>> [3/8] [A] Métricas completas (F1, Prec, Rec, AUC, Brier)...\n")

metricas_lista <- list()

sint_train_bin <- sint_train |>
  mutate(pobre = factor(pobre, levels = c(0, 1), labels = c("no_pobre", "pobre")))

modelos_sint_nombres <- c("logit_sintetico_top", "rf_sintetico_top", "xgb_sintetico_top")
labels_sint          <- c("Logit", "Random Forest", "XGBoost")

for (i in seq_along(modelos_sint_nombres)) {
  nm <- modelos_sint_nombres[i]
  lb <- labels_sint[i]
  
  # Leer threshold del log
  thr_i <- resultados_sint$threshold[resultados_sint$modelo == lb]
  if (length(thr_i) == 0) next
  
  # Intentar cargar modelo para obtener probabilidades
  rds_path <- here(paths$submissions, "04_Sinteticos", paste0(nm, ".rds"))
  if (file.exists(rds_path)) {
    m_i <- readRDS(rds_path)
    
    probs_i <- tryCatch({
      if (inherits(m_i, "ranger")) {
        m_i$predictions[, "pobre"]
      } else if (inherits(m_i, "train")) {
        predict(m_i, newdata = sint_train_bin |> select(-id, -pobre),
                type = "prob")[, "pobre"]
      } else {
        dummy_sint <- dummyVars(~ ., data = sint_train_bin |> select(-id, -pobre),
                                fullRank = TRUE)
        X_s <- predict(dummy_sint, sint_train_bin |> select(-id, -pobre))
        predict(m_i, xgb.DMatrix(data = X_s))
      }
    }, error = function(e) NULL)
    
    if (!is.null(probs_i)) {
      m_row <- calcular_metricas(sint_train_bin$pobre, probs_i, thr_i, lb, "Sintéticos")
      metricas_lista[[length(metricas_lista) + 1]] <- m_row
    }
  }
}

if (!is.null(oof_xgb_real)) {
  thr_xgb_real <- mejores_reales$threshold[mejores_reales$modelo == "XGBoost"]
  if (length(thr_xgb_real) > 0) {
    idx_val <- which(!is.na(oof_xgb_real))
    m_row <- calcular_metricas(
      train_min$pobre[idx_val],
      oof_xgb_real[idx_val],
      thr_xgb_real,
      "XGBoost", "Reales (DANE)"
    )
    metricas_lista[[length(metricas_lista) + 1]] <- m_row
  }
}

# Si hay métricas calculadas, guardar tabla extendida
if (length(metricas_lista) > 0) {
  metricas_completas <- bind_rows(metricas_lista)
  
  # Tabla presentable (selección de columnas para el informe)
  tabla_metricas <- metricas_completas |>
    select(modelo, datos, threshold, f1, precision, recall, auc, brier, fnr, fpr) |>
    arrange(datos, desc(f1))
  
  write.csv(tabla_metricas,      here(paths$tables, "metricas_completas.csv"),    row.names = FALSE)
  write.csv(metricas_completas,  here(paths$tables, "metricas_completas_full.csv"), row.names = FALSE)
  
  guardar_tabla_tex(
    tabla_metricas,
    "metricas_completas.tex",
    caption = "Métricas de desempeño de los modelos evaluados.",
    label = "tab:metricas_completas"
  )
  
  cat("    metricas_completas.csv guardado.\n")
  print(tabla_metricas)
} else {
  cat("    AVISO: no se pudieron calcular métricas extendidas (modelos no encontrados).\n")
  cat("    Se generarán con los valores del log.\n")
  metricas_completas <- NULL
}

# ============================================================
# [B] DESCOMPOSICIÓN DEL ERROR — política pública
# ============================================================

cat("\n>>> [4/8] [B] Descomposición del error (TP/FP/FN/TN + tasas PP)...\n")

if (!is.null(metricas_completas)) {
  error_pp <- metricas_completas |>
    mutate(
      # FNR = Tasa de pobres no detectados (el costo más alto en política social)
      costo_fn_pct    = round(fnr * 100, 2),
      # FPR = Tasa de no-pobres clasificados como pobres (costo fiscal)
      costo_fp_pct    = round(fpr * 100, 2),
      # Precisión de focalización (de los que clasificamos como pobres, qué % lo son)
      precision_pct   = round(precision * 100, 2),
      # Cobertura (de todos los pobres, qué % identificamos)
      cobertura_pct   = round(recall * 100, 2)
    ) |>
    select(modelo, datos, threshold,
           tp, fp, fn, tn, n,
           costo_fn_pct, costo_fp_pct,
           precision_pct, cobertura_pct,
           f1, brier)
  
  write.csv(error_pp, here(paths$tables, "descomposicion_error_pp.csv"), row.names = FALSE)
  
  guardar_tabla_tex(
    error_pp,
    "descomposicion_error_pp.tex",
    caption = "Descomposición de errores de clasificación por modelo.",
    label = "tab:descomposicion_error_pp"
  )
  
  # Tabla de costos de política pública
  tabla_costos_pp <- error_pp |>
    select(modelo, datos, threshold,
           `FNR (pobres no detectados)` = costo_fn_pct,
           `FPR (errores de focalización)` = costo_fp_pct,
           `Precisión (%)` = precision_pct,
           `Cobertura (%)` = cobertura_pct) |>
    arrange(datos, `FNR (pobres no detectados)`)
  
  write.csv(tabla_costos_pp, here(paths$tables, "costos_politica_publica.csv"), row.names = FALSE)
  guardar_tabla_tex(
    tabla_costos_pp,
    "costos_politica_publica.tex",
    caption = "Indicadores de focalización para política pública.",
    label = "tab:costos_politica_publica"
  )
  
  cat("    descomposicion_error_pp.csv guardado.\n")
  cat("    costos_politica_publica.csv guardado.\n")
  print(tabla_costos_pp)
}

# ============================================================
# [C] CROSS-EVALUATION MATRIX
# Mide generalización real: entrenar en A, evaluar en B
# ============================================================

cat("\n>>> [5/8] [C] Cross-evaluation matrix...\n")

vars_comunes <- intersect(
  setdiff(names(train_min),  c("id", "pobre")),
  setdiff(names(sint_train), c("id", "pobre"))
)

cat("    Variables comunes para cross-eval:", length(vars_comunes), "\n")

if (length(vars_comunes) >= 5) {
  
  X_real_cv <- train_min[, vars_comunes]
  y_real_cv <- as.numeric(train_min$pobre == "pobre")
  
  sint_all  <- bind_rows(sint_train, sint_test) |>
    mutate(pobre = as.integer(pobre == 1 | pobre == "pobre"))
  X_sint_cv <- sint_all[, vars_comunes]
  y_sint_cv <- sint_all$pobre
  
  all_data_cv <- bind_rows(
    X_real_cv |> mutate(.src = "real"),
    X_sint_cv |> mutate(.src = "sint")
  )
  
  dummy_cv <- dummyVars(~ ., data = all_data_cv |> select(-.src), fullRank = TRUE)
  X_real_m <- predict(dummy_cv, X_real_cv)
  X_sint_m <- predict(dummy_cv, X_sint_cv)
  
  dtrain_real <- xgb.DMatrix(data = X_real_m, label = y_real_cv)
  dtrain_sint <- xgb.DMatrix(data = X_sint_m, label = y_sint_cv)
  
  params_cv <- list(
    booster = "gbtree", objective = "binary:logistic",
    eval_metric = "auc", eta = 0.05, max_depth = 6,
    subsample = 0.8, colsample_bytree = 0.8,
    min_child_weight = 5,
    nthread = parallel::detectCores() - 1
  )
  
  set.seed(SEED)
  xgb_real_cv <- xgb.train(params = params_cv, data = dtrain_real,
                           nrounds = 300, verbose = 0)
  set.seed(SEED)
  xgb_sint_cv <- xgb.train(params = params_cv, data = dtrain_sint,
                           nrounds = 300, verbose = 0)
  
  # Threshold óptimo para cross-eval (sobre datos de entrenamiento)
  opt_real_thr <- mejores_reales$threshold[mejores_reales$modelo == "XGBoost"]
  opt_sint_thr <- resultados_sint$threshold[resultados_sint$modelo == "XGBoost"]
  if (length(opt_real_thr) == 0) opt_real_thr <- 0.40
  if (length(opt_sint_thr) == 0) opt_sint_thr <- 0.40
  
  # Evaluar las 4 celdas de la matriz
  cross_eval <- bind_rows(
    # Modelo real → datos reales (in-sample, referencia)
    calcular_metricas(
      factor(y_real_cv, levels = c(0, 1), labels = c("no_pobre", "pobre")),
      predict(xgb_real_cv, dtrain_real),
      opt_real_thr, "XGBoost entrenado en Reales", "Evaluado en Reales"
    ),
    # Modelo real → datos sintéticos (generalización fuera de dominio)
    calcular_metricas(
      factor(y_sint_cv, levels = c(0, 1), labels = c("no_pobre", "pobre")),
      predict(xgb_real_cv, dtrain_sint),
      opt_real_thr, "XGBoost entrenado en Reales", "Evaluado en Sintéticos"
    ),
    # Modelo sintético → datos sintéticos (in-sample)
    calcular_metricas(
      factor(y_sint_cv, levels = c(0, 1), labels = c("no_pobre", "pobre")),
      predict(xgb_sint_cv, dtrain_sint),
      opt_sint_thr, "XGBoost entrenado en Sintéticos", "Evaluado en Sintéticos"
    ),
    # Modelo sintético → datos reales (test de fidelidad del generador)
    calcular_metricas(
      factor(y_real_cv, levels = c(0, 1), labels = c("no_pobre", "pobre")),
      predict(xgb_sint_cv, dtrain_real),
      opt_sint_thr, "XGBoost entrenado en Sintéticos", "Evaluado en Reales"
    )
  )
  
  # Matriz presentable
  cross_matrix <- cross_eval |>
    select(modelo, datos, f1, precision, recall, auc, fnr, brier) |>
    rename(
      `Entrenado en`   = modelo,
      `Evaluado en`    = datos
    )
  
  write.csv(cross_eval,   here(paths$tables, "cross_evaluation_full.csv"),   row.names = FALSE)
  write.csv(cross_matrix, here(paths$tables, "cross_evaluation_matrix.csv"), row.names = FALSE)
  
  guardar_tabla_tex(
    cross_matrix,
    "cross_evaluation_matrix.tex",
    caption = "Matriz de cross-evaluación para XGBoost.",
    label = "tab:cross_evaluation_matrix"
  )
  
  guardar_tabla_tex(
    cross_eval |>
      select(modelo, datos, f1, precision, recall, auc, fnr, brier),
    "cross_evaluation_full.tex",
    caption = "Resultados completos de cross-evaluación.",
    label = "tab:cross_evaluation_full"
  )
  
  cat("\n    Cross-evaluation matrix (XGBoost):\n")
  print(cross_matrix)
  
} else {
  cat("    AVISO: pocas variables comunes para cross-eval. Omitiendo.\n")
  cross_eval <- NULL
}

# ============================================================
# [D] SENSIBILIDAD AL THRESHOLD
# ============================================================

cat("\n>>> [6/8] [D] Análisis de sensibilidad al threshold...\n")

f1_threshold_lista <- list()

# Para modelos sintéticos
for (i in seq_along(modelos_sint_nombres)) {
  nm <- modelos_sint_nombres[i]
  lb <- labels_sint[i]
  rds_path <- here(paths$submissions, "04_Sinteticos", paste0(nm, ".rds"))
  
  if (file.exists(rds_path)) {
    m_i <- readRDS(rds_path)
    probs_i <- tryCatch({
      if (inherits(m_i, "ranger")) {
        m_i$predictions[, "pobre"]
      } else if (inherits(m_i, "train")) {
        predict(m_i, newdata = sint_train_bin |> select(-id, -pobre),
                type = "prob")[, "pobre"]
      } else {
        dummy_sint <- dummyVars(~ ., data = sint_train_bin |> select(-id, -pobre),
                                fullRank = TRUE)
        X_s <- predict(dummy_sint, sint_train_bin |> select(-id, -pobre))
        predict(m_i, xgb.DMatrix(data = X_s))
      }
    }, error = function(e) NULL)
    
    if (!is.null(probs_i)) {
      f1_thr <- calcular_f1_threshold(sint_train_bin$pobre, probs_i, lb, "Sintéticos")
      f1_threshold_lista[[length(f1_threshold_lista) + 1]] <- f1_thr
    }
  }
}

# Para XGBoost real (si hay OOF)
if (!is.null(oof_xgb_real)) {
  idx_val <- which(!is.na(oof_xgb_real))
  f1_thr <- calcular_f1_threshold(
    train_min$pobre[idx_val], oof_xgb_real[idx_val],
    "XGBoost", "Reales (DANE)"
  )
  f1_threshold_lista[[length(f1_threshold_lista) + 1]] <- f1_thr
}

if (length(f1_threshold_lista) > 0) {
  f1_threshold_df <- bind_rows(f1_threshold_lista)
  write.csv(f1_threshold_df, here(paths$tables, "f1_vs_threshold.csv"), row.names = FALSE)
  cat("    f1_vs_threshold.csv guardado.\n")
}

# ============================================================
# [E] CURVAS PRECISION-RECALL
# ============================================================

cat("\n>>> [7/8] Generando gráficos extendidos...\n")

# ============================================================
# GRÁFICOS
# ============================================================

colores_datos <- c("Reales (DANE)" = COL_REAL, "Sintéticos" = COL_SINT)

# --- G1: F1 por modelo (existente, mejorado) ----------------
p_f1 <- comparativa |>
  group_by(modelo, datos) |>
  slice_max(f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  ggplot(aes(x = reorder(modelo, f1), y = f1, fill = datos)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", f1)),
            position = position_dodge(width = 0.7),
            hjust = -0.1, size = 3.2, color = "#333333") +
  coord_flip() +
  scale_fill_manual(values = colores_datos) +
  scale_y_continuous(limits = c(0, 0.85), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "F1-score por modelo",
       subtitle = "Datos reales (DANE MESE 2018) vs datos sintéticos",
       x = NULL, y = "F1-score", fill = NULL,
       caption = "Encuesta mínima: 11 vars raw + 6 features | Tópicos IA - Uniandes 2026") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", color = COL_REAL, size = 13),
        plot.subtitle = element_text(color = "#555555", size = 10),
        plot.caption  = element_text(color = COL_GREY, size = 8),
        legend.position = "bottom",
        panel.grid.major.y = element_blank())

# --- G2: Diferencia F1 -------------------------------------
p_diff <- comparativa_wide |>
  ggplot(aes(x = reorder(modelo, diferencia), y = diferencia,
             fill = diferencia >= 0)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = COL_GREY, linewidth = 0.5) +
  geom_text(aes(label = sprintf("%+.3f", diferencia)),
            hjust = ifelse(comparativa_wide$diferencia >= 0, -0.1, 1.1),
            size = 3.5, color = "#333333") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = COL_REAL, "FALSE" = COL_SINT), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.2, 0.2))) +
  labs(title = "Brecha de generalización por modelo",
       subtitle = "F1 reales − F1 sintéticos",
       x = NULL, y = "Diferencia F1",
       caption = "Tópicos IA - Uniandes 2026") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", color = COL_REAL, size = 13),
        plot.subtitle = element_text(color = "#555555", size = 10),
        panel.grid.major.y = element_blank())

# --- G3: Curvas F1 vs Threshold ----------------------------
if (length(f1_threshold_lista) > 0) {
  # Óptimos por modelo (líneas verticales)
  optimos <- bind_rows(
    mejores_reales |> mutate(datos = "Reales (DANE)"),
    resultados_sint
  ) |> distinct(modelo, datos, threshold)
  
  p_threshold <- f1_threshold_df |>
    ggplot(aes(x = threshold, y = f1, color = datos, linetype = modelo)) +
    geom_line(linewidth = 0.9) +
    geom_vline(data = optimos |> filter(datos == "Reales (DANE)"),
               aes(xintercept = threshold, color = datos),
               linetype = "dotted", linewidth = 0.5, alpha = 0.7) +
    scale_color_manual(values = colores_datos) +
    scale_x_continuous(breaks = seq(0.1, 0.9, 0.1)) +
    labs(title = "F1-score en función del threshold",
         subtitle = "Las líneas punteadas indican el threshold óptimo usado",
         x = "Threshold de clasificación", y = "F1-score",
         color = NULL, linetype = "Modelo",
         caption = "Tópicos IA - Uniandes 2026") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", color = COL_REAL),
          legend.position = "bottom")
  
  ggsave(here(paths$figures, "f1_vs_threshold.png"),
         plot = p_threshold, width = 8, height = 5, dpi = 150)
  cat("    f1_vs_threshold.png\n")
}

# --- G4: Cross-evaluation heatmap --------------------------
if (!is.null(cross_eval)) {
  cross_heat <- cross_eval |>
    mutate(
      entrenado = gsub("XGBoost entrenado en ", "", modelo),
      evaluado  = gsub("Evaluado en ", "", datos)
    )
  
  p_cross <- cross_heat |>
    ggplot(aes(x = evaluado, y = entrenado, fill = f1)) +
    geom_tile(color = "white", linewidth = 1.5) +
    geom_text(aes(label = sprintf("F1 = %.3f\nAUC = %.3f\nFNR = %.3f",
                                  f1, auc, fnr)),
              color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient2(low = COL_SINT, mid = COL_LIGHT, high = COL_REAL,
                         midpoint = 0.55, limits = c(0.3, 0.8),
                         name = "F1") +
    labs(title = "Matriz de cross-evaluación (XGBoost)",
         subtitle = "Diagonal = in-sample | Fuera diagonal = generalización",
         x = "Dataset de evaluación", y = "Dataset de entrenamiento",
         caption = "FNR = tasa de pobres no detectados | Tópicos IA - Uniandes 2026") +
    theme_minimal(base_size = 13) +
    theme(plot.title    = element_text(face = "bold", color = COL_REAL, size = 13),
          plot.subtitle = element_text(color = "#555555"),
          axis.text     = element_text(face = "bold"),
          panel.grid    = element_blank())
  
  ggsave(here(paths$figures, "cross_evaluation_heatmap.png"),
         plot = p_cross, width = 8, height = 5.5, dpi = 150)
  cat("    cross_evaluation_heatmap.png\n")
}

# --- G5: Descomposición de error (FNR vs FPR) ---------------
if (!is.null(metricas_completas)) {
  p_error <- metricas_completas |>
    ggplot(aes(x = fpr, y = fnr, color = datos, shape = modelo, label = modelo)) +
    geom_point(size = 5, alpha = 0.85) +
    geom_text(nudge_y = 0.015, size = 3.2, color = "#333333") +
    scale_color_manual(values = colores_datos) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0.15,
             fill = "#E8F5E9", alpha = 0.3) +
    annotate("text", x = 0.01, y = 0.01,
             label = "Zona objetivo (FNR bajo)", hjust = 0, size = 3,
             color = "#2E7D32", fontface = "italic") +
    labs(title = "Trade-off de errores por modelo",
         subtitle = "FNR = pobres no detectados (costo social) | FPR = errores de focalización (costo fiscal)",
         x = "Tasa de Falsos Positivos (FPR)", y = "Tasa de Falsos Negativos (FNR)",
         color = NULL, shape = "Modelo",
         caption = "Tópicos IA - Uniandes 2026") +
    theme_minimal(base_size = 12) +
    theme(plot.title    = element_text(face = "bold", color = COL_REAL),
          plot.subtitle = element_text(color = "#555555", size = 9),
          legend.position = "bottom")
  
  ggsave(here(paths$figures, "error_fnr_fpr.png"),
         plot = p_error, width = 8, height = 6, dpi = 150)
  cat("    error_fnr_fpr.png\n")
  
  # --- G6: Brier score (calibración) ----------------------
  p_brier <- metricas_completas |>
    ggplot(aes(x = reorder(modelo, brier), y = brier, fill = datos)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_hline(yintercept = 0.25, linetype = "dashed", color = COL_GREY,
               linewidth = 0.5) +
    annotate("text", x = 0.5, y = 0.255, label = "Modelo sin información (0.25)",
             hjust = 0, size = 3, color = COL_GREY) +
    geom_text(aes(label = sprintf("%.3f", brier)),
              position = position_dodge(width = 0.7),
              vjust = -0.3, size = 3.2, color = "#333333") +
    scale_fill_manual(values = colores_datos) +
    scale_y_continuous(limits = c(0, 0.3), expand = expansion(mult = c(0, 0.1))) +
    labs(title = "Brier Score por modelo",
         subtitle = "Menor es mejor | 0.25 = modelo sin información (aleatorio)",
         x = NULL, y = "Brier Score", fill = NULL,
         caption = "Tópicos IA - Uniandes 2026") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", color = COL_REAL),
          legend.position = "bottom",
          panel.grid.major.x = element_blank())
  
  ggsave(here(paths$figures, "brier_score.png"),
         plot = p_brier, width = 7, height = 4.5, dpi = 150)
  cat("    brier_score.png\n")
}

# --- G7: Spider/radar de métricas (alternativa compacta) ---
if (!is.null(metricas_completas)) {
  metricas_radar <- metricas_completas |>
    mutate(specificity_val = specificity) |>
    select(modelo, datos, f1, precision, recall, auc, specificity_val) |>
    pivot_longer(cols = c(f1, precision, recall, auc, specificity_val),
                 names_to = "metrica", values_to = "valor") |>
    mutate(metrica = recode(metrica,
                            "f1"              = "F1",
                            "precision"       = "Precisión",
                            "recall"          = "Cobertura",
                            "auc"             = "AUC",
                            "specificity_val" = "Especificidad"))
  
  p_radar <- metricas_radar |>
    ggplot(aes(x = metrica, y = valor, fill = datos,
               group = interaction(modelo, datos))) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.2f", valor)),
              position = position_dodge(width = 0.75),
              vjust = -0.3, size = 2.8, color = "#333333") +
    facet_wrap(~ modelo) +
    scale_fill_manual(values = colores_datos) +
    scale_y_continuous(limits = c(0, 1.05), expand = expansion(mult = c(0, 0.08))) +
    labs(title = "Perfil de métricas por modelo",
         subtitle = "Comparación reales vs sintéticos en 5 dimensiones",
         x = NULL, y = "Valor", fill = NULL,
         caption = "Tópicos IA - Uniandes 2026") +
    theme_minimal(base_size = 11) +
    theme(plot.title     = element_text(face = "bold", color = COL_REAL),
          legend.position = "bottom",
          strip.text      = element_text(face = "bold"),
          axis.text.x     = element_text(angle = 25, hjust = 1))
  
  ggsave(here(paths$figures, "perfil_metricas.png"),
         plot = p_radar, width = 10, height = 6, dpi = 150)
  cat("    perfil_metricas.png\n")
}

# ============================================================
# [G] ANÁLISIS POR SUBGRUPO: urbano/rural + dominio
# ============================================================

cat("\n>>> [8/8] [G] Desempeño por subgrupo...\n")

if (!is.null(oof_xgb_real)) {
  idx_val <- which(!is.na(oof_xgb_real))
  
  subgrupo_df <- train_min[idx_val, ] |>
    mutate(
      prob_xgb   = oof_xgb_real[idx_val],
      zona       = ifelse(clase == "1", "Cabecera (Urbano)", "Resto (Rural)"),
      y_bin      = as.integer(pobre == "pobre")
    )
  
  # F1 por zona
  thr_xgb <- mejores_reales$threshold[mejores_reales$modelo == "XGBoost"]
  if (length(thr_xgb) == 0) thr_xgb <- 0.40
  
  metricas_zona <- subgrupo_df |>
    group_by(zona) |>
    summarise(
      n_hogares  = n(),
      tasa_pob   = round(mean(y_bin) * 100, 1),
      f1         = {
        p <- prob_xgb; y <- y_bin; t <- thr_xgb
        pred <- as.integer(p >= t)
        tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0)
        fn <- sum(pred == 0 & y == 1)
        pr <- if (tp+fp==0) NA else tp/(tp+fp)
        rc <- if (tp+fn==0) NA else tp/(tp+fn)
        if (is.na(pr)||is.na(rc)||pr+rc==0) NA else round(2*pr*rc/(pr+rc), 4)
      },
      fnr = {
        p <- prob_xgb; y <- y_bin; t <- thr_xgb
        pred <- as.integer(p >= t)
        tp <- sum(pred==1 & y==1); fn <- sum(pred==0 & y==1)
        if (tp+fn==0) NA else round(fn/(tp+fn), 4)
      },
      .groups = "drop"
    )
  
  write.csv(metricas_zona, here(paths$tables, "desempeno_zona.csv"), row.names = FALSE)
  cat("    desempeno_zona.csv guardado.\n")
  print(metricas_zona)
  
  # Gráfico de subgrupos
  p_zona <- metricas_zona |>
    pivot_longer(cols = c(f1, fnr), names_to = "metrica", values_to = "valor") |>
    mutate(metrica = recode(metrica, "f1" = "F1-score", "fnr" = "FNR (pobres no detectados)")) |>
    ggplot(aes(x = zona, y = valor, fill = zona)) +
    geom_col(width = 0.55) +
    geom_text(aes(label = sprintf("%.3f", valor)), vjust = -0.4, size = 3.5) +
    facet_wrap(~ metrica, scales = "free_y") +
    scale_fill_manual(values = c("Cabecera (Urbano)" = COL_REAL,
                                 "Resto (Rural)"     = COL_SINT)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Desempeño del XGBoost por zona",
         subtitle = "XGBoost OOF sobre datos DANE MESE 2018",
         x = NULL, y = NULL, fill = NULL,
         caption = "Tópicos IA - Uniandes 2026") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", color = COL_REAL),
          legend.position = "none",
          panel.grid.major.x = element_blank())
  
  ggsave(here(paths$figures, "desempeno_zona.png"),
         plot = p_zona, width = 8, height = 4.5, dpi = 150)
  cat("    desempeno_zona.png\n")
}

# ============================================================
# GRÁFICOS importancia
# ============================================================

imp_full <- tryCatch(
  read.csv(here(paths$models, "imp_agregada.csv")),
  error = function(e) NULL
)

MIN_RAW_VARS <- c("prop_reg_subsidiado", "nper", "n_ocupados", "p5000",
                  "p5090", "horas_trabajo_prom", "n_menores_18",
                  "edad_promedio", "prop_cotiza_pension", "depto", "edad_min")
MIN_ENG_VARS <- c("ratio_dependencia", "hacinamiento", "nper_sq",
                  "edad_prom_sq", "calidad_empleo", "doble_proteccion")

if (!is.null(imp_full)) {
  if ("Gain_total" %in% names(imp_full)) {
    full_gain_total <- sum(imp_full$Gain_total, na.rm = TRUE)
    
    preparar_importancia <- function(df, full_total) {
      df |>
        arrange(desc(Gain_total)) |>
        mutate(gain_acum_pct_full = 100 * cumsum(Gain_total) / full_total,
               gain_pct_full = 100 * Gain_total / full_total)
    }
    
    imp_raw11 <- imp_full |>
      filter(var_base %in% MIN_RAW_VARS) |>
      preparar_importancia(full_gain_total)
    
    imp_17 <- imp_full |>
      filter(var_base %in% c(MIN_RAW_VARS, MIN_ENG_VARS)) |>
      preparar_importancia(full_gain_total)
    
    write.csv(imp_raw11, here(paths$tables, "resumen_importancia_raw11.csv"), row.names = FALSE)
    write.csv(imp_17,    here(paths$tables, "resumen_importancia_17.csv"),    row.names = FALSE)
    
    p_imp_raw <- imp_raw11 |>
      mutate(var_base = reorder(var_base, gain_pct_full)) |>
      ggplot(aes(x = var_base, y = gain_pct_full)) +
      geom_col(fill = COL_REAL, width = 0.7) +
      geom_text(aes(label = sprintf("%.1f%%", gain_pct_full)),
                hjust = -0.1, size = 3.5, color = "#333333") +
      coord_flip() +
      scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
      labs(title = "Importancia relativa — 11 variables raw",
           subtitle = "% del Gain total del XGBoost completo",
           x = NULL, y = "% del Gain total",
           caption = "DANE MESE 2018 | Tópicos IA - Uniandes 2026") +
      theme_minimal(base_size = 12) +
      theme(plot.title = element_text(face = "bold", color = COL_REAL),
            panel.grid.major.y = element_blank())
    
    p_imp_17 <- imp_17 |>
      mutate(var_base = reorder(var_base, gain_pct_full)) |>
      ggplot(aes(x = var_base, y = gain_pct_full)) +
      geom_col(fill = COL_SINT, width = 0.7) +
      geom_text(aes(label = sprintf("%.1f%%", gain_pct_full)),
                hjust = -0.1, size = 3.5, color = "#333333") +
      coord_flip() +
      scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
      labs(title = "Importancia relativa — 17 variables del modelo mínimo",
           subtitle = "11 vars raw + 6 features derivadas",
           x = NULL, y = "% del Gain total",
           caption = "DANE MESE 2018 | Tópicos IA - Uniandes 2026") +
      theme_minimal(base_size = 12) +
      theme(plot.title = element_text(face = "bold", color = COL_REAL),
            panel.grid.major.y = element_blank())
    
    ggsave(here(paths$figures, "importancia_raw11.png"), plot = p_imp_raw,
           width = 8, height = 5, dpi = 150)
    ggsave(here(paths$figures, "importancia_17.png"), plot = p_imp_17,
           width = 8, height = 5, dpi = 150)
    
    cobertura_gain <- data.frame(
      conjunto = c("11 raw", "17 completas"),
      n_vars   = c(nrow(imp_raw11), nrow(imp_17)),
      cobertura_pct = c(100 * sum(imp_raw11$Gain_total) / full_gain_total,
                        100 * sum(imp_17$Gain_total)    / full_gain_total)
    )
    write.csv(cobertura_gain, here(paths$tables, "cobertura_gain.csv"), row.names = FALSE)
  }
}

# ============================================================
# PANEL AMPLIADO PARA EL ARTÍCULO
# ============================================================

cat("\n>>> Generando panel ampliado...\n")

panel_base <- (p_f1 | p_diff) +
  plot_annotation(
    title    = "Clasificación de pobreza con encuesta mínima — Comparativa de modelos",
    subtitle = "11 variables raw + 6 features | DANE MESE 2018, Bogotá",
    caption  = "Jose Rincón · Lucas Rodríguez · María Paula Osuna | Tópicos IA · Uniandes 2026",
    theme = theme(
      plot.title    = element_text(face = "bold", color = COL_REAL, size = 14, hjust = 0.5),
      plot.subtitle = element_text(color = "#555555", size = 10, hjust = 0.5),
      plot.caption  = element_text(color = COL_GREY, size = 8, hjust = 0.5)
    )
  )

ggsave(here(paths$figures, "comparativa_panel.png"),
       plot = panel_base, width = 14, height = 6, dpi = 150)
cat("    comparativa_panel.png\n")

# ============================================================
# RESUMEN FINAL EN CONSOLA
# ============================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║   RESULTADOS FINALES                                        ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n")

cat("\n  [1] Comparativa F1 reales vs sintéticos:\n")
print(as.data.frame(comparativa))

cat("\n  [2] Brecha de generalización:\n")
print(as.data.frame(comparativa_wide))

if (!is.null(metricas_completas)) {
  cat("\n  [3] Métricas completas (F1 | Prec | Rec | AUC | Brier | FNR):\n")
  print(as.data.frame(tabla_metricas))
}

if (!is.null(cross_eval)) {
  cat("\n  [4] Cross-evaluation matrix (XGBoost):\n")
  print(as.data.frame(cross_matrix))
}

cat("\n  Outputs generados:\n")
archivos_out <- c(
  "comparativa_final.csv", "comparativa_wide.csv",
  "metricas_completas.csv", "metricas_completas_full.csv",
  "descomposicion_error_pp.csv", "costos_politica_publica.csv",
  "cross_evaluation_matrix.csv", "cross_evaluation_full.csv",
  "f1_vs_threshold.csv", "desempeno_zona.csv",
  "resumen_importancia_raw11.csv", "resumen_importancia_17.csv",
  "cobertura_gain.csv"
)
for (f in archivos_out) cat("    ·", f, "\n")

cat("\n  Figuras generadas:\n")
figuras_out <- c(
  "comparativa_f1.png", "comparativa_diferencia.png",
  "f1_vs_threshold.png", "cross_evaluation_heatmap.png",
  "error_fnr_fpr.png", "brier_score.png",
  "perfil_metricas.png", "desempeno_zona.png",
  "importancia_raw11.png", "importancia_17.png",
  "comparativa_panel.png"
)
for (f in figuras_out) cat("    ·", f, "\n")

# ============================================================
# Limpiar entorno
# ============================================================

rm(list = ls(pattern = "^(log_cv|mejores_reales|resultados_sint|
              comparativa|metricas|cross|f1_threshold|subgrupo|
              imp_raw|imp_17|cobertura|error_pp|tabla|
              p_|panel|params|dtrain|xgb|X_|y_|
              sint_|train_min|oof_|idx|optimos|all_data|
              dummy|vars_comunes|archivos|figuras|
              MIN_|COL_|TIPO).*"))
gc()

cat("\n>>> 05_comparativa.R v2 completado\n")
