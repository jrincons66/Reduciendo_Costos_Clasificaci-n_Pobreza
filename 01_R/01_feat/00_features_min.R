# ============================================================
# 00_features_min.R
# Encuesta mínima — selección top por importancia XGBoost
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
#
# Descripción:
#   Entrena un XGBoost sobre el dataset completo (18 features),
#   extrae la importancia por Gain, selecciona las 11 variables
#   raw más importantes y construye train_min.rds / test_min.rds
#   con solo esas variables, más las features que sí se pueden
#   reconstruir a partir de ellas. Estos datasets alimentan los
#   modelos de la encuesta mínima.
#
# Outputs:
#   - 00_data/01_processed/train_min.rds
#   - 00_data/01_processed/test_min.rds
#   - 00_data/01_processed/min_raw_vars.rds
#   - 00_data/01_processed/top_vars.rds
#   - 02_models/imp_agregada.csv
#   - 02_models/min_importancia.csv
#   - 04_outputs/tables/top15_importancia.csv
#   - 04_outputs/tables/top17_importancia.csv
#   - 04_outputs/tables/cortes_gain.csv
#   - 04_outputs/tables/cobertura_gain.csv
#   - 04_outputs/figures/top15_importancia.png
#   - 04_outputs/figures/gain_acumulado.png
#   - 04_outputs/figures/gap_top15_importancia.png
#   - 04_outputs/figures/min_importancia.png
#   - 04_outputs/figures/min_importancia_completa.png
#   - 04_outputs/figures/cobertura_gain.png
# ============================================================

train_full <- readRDS(here(paths$processed, "train_features.rds"))
test_full  <- readRDS(here(paths$processed, "test_features.rds"))

train_full <- train_full |>
  mutate(
    pobre = factor(pobre, levels = c(0, 1),
                   labels = c("no_pobre", "pobre"))
  )

# ============================================================
# PASO 1 — Entrenar XGBoost completo para importancia
# ============================================================

cat(">>> [1/4] Entrenando XGBoost para importancia...\n")

dummy_recipe <- dummyVars(~ ., data = train_full |> select(-id, -pobre),
                          fullRank = TRUE)
X_full  <- predict(dummy_recipe, train_full |> select(-id, -pobre))
y_full  <- as.numeric(train_full$pobre == "pobre")
dtrain  <- xgb.DMatrix(data = X_full, label = y_full)

params <- list(
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
xgb_full <- xgb.train(
  params        = params,
  data          = dtrain,
  nrounds       = 500,
  verbose       = 0
)

# --- Importancia por Gain -----------------------------------
imp_raw <- xgb.importance(
  feature_names = colnames(X_full),
  model         = xgb_full
)

# ============================================================
# PASO 2 — Mapear dummies → variables base y seleccionar top
# ============================================================

cat(">>> [2/4] Seleccionando variables...\n")

todas_vars <- setdiff(names(train_full), c("id", "pobre"))

mapear_dummy_a_base <- function(dummy_name, vars_dataset) {
  if (dummy_name %in% vars_dataset) return(dummy_name)
  candidatos <- vars_dataset[startsWith(dummy_name, vars_dataset)]
  if (length(candidatos) == 0) return(NA_character_)
  candidatos[which.max(nchar(candidatos))]
}

imp_agregada <- imp_raw |>
  mutate(var_base = sapply(Feature, mapear_dummy_a_base,
                           vars_dataset = todas_vars)) |>
  filter(!is.na(var_base)) |>
  group_by(var_base) |>
  summarise(Gain_total = sum(Gain), .groups = "drop") |>
  arrange(desc(Gain_total)) |>
  mutate(
    rank = row_number(),
    gain_gap = lag(Gain_total) - Gain_total,
    gain_acum = cumsum(Gain_total),
    gain_acum_pct = 100 * gain_acum / sum(Gain_total),
    gain_pct = 100 * Gain_total / sum(Gain_total)
  )

# Guardar ranking completo para reutilización posterior
write.csv(
  imp_agregada,
  here(paths$models, "imp_agregada.csv"),
  row.names = FALSE
)

write.csv(
  imp_agregada,
  here(paths$tables, "imp_agregada.csv"),
  row.names = FALSE
)

# Cortes de cobertura del gain
cortes_gain <- c(50, 70, 80, 90)

tabla_cortes <- do.call(rbind, lapply(cortes_gain, function(corte) {
  idx <- which(imp_agregada$gain_acum_pct >= corte)[1]
  data.frame(
    corte_pct      = corte,
    n_variables    = idx,
    variable_corte = if (is.na(idx)) NA_character_ else imp_agregada$var_base[idx],
    gain_acum_pct  = if (is.na(idx)) NA_real_ else round(imp_agregada$gain_acum_pct[idx], 2),
    stringsAsFactors = FALSE
  )
}))

write.csv(
  tabla_cortes,
  here(paths$tables, "cortes_gain.csv"),
  row.names = FALSE
)

# ============================================================
# PASO 3 — Construir dataset mínimo
# ============================================================

cat("\n>>> [3/4] Construyendo dataset mínimo...\n")

# Variables originales mínimas seleccionadas
MIN_RAW_VARS <- c(
  "prop_reg_subsidiado",
  "nper",
  "n_ocupados",
  "p5000",
  "p5090",
  "horas_trabajo_prom",
  "n_menores_18",
  "edad_promedio",
  "prop_cotiza_pension",
  "depto",
  "edad_min"
)

# Features que sí se pueden construir a partir de esas 11 variables
MIN_ENG_VARS <- c(
  "ratio_dependencia",
  "hacinamiento",
  "nper_sq",
  "edad_prom_sq",
  "calidad_empleo",
  "doble_proteccion"
)

# Dependencias de las features mínimas
DEPS_MIN <- list(
  ratio_dependencia = c("nper", "n_ocupados"),
  hacinamiento      = c("nper", "p5000"),
  nper_sq           = c("nper"),
  edad_prom_sq      = c("edad_promedio"),
  calidad_empleo    = c("horas_trabajo_prom", "prop_cotiza_pension"),
  doble_proteccion  = c("prop_cotiza_pension")
)

# Recalcular features mínimas desde train_clean / test_clean
train_clean <- readRDS(here(paths$processed, "train_clean.rds"))
test_clean  <- readRDS(here(paths$processed, "test_clean.rds"))

aplicar_eng_min <- function(df) {
  df |>
    mutate(
      ratio_dependencia = (nper - n_ocupados) / pmax(nper, 1),
      hacinamiento      = nper / pmax(p5000, 1),
      nper_sq           = nper^2,
      edad_prom_sq      = edad_promedio^2,
      calidad_empleo    = horas_trabajo_prom * prop_cotiza_pension,
      doble_proteccion  = prop_cotiza_pension^2
    )
}

train_min <- aplicar_eng_min(train_clean)
test_min  <- aplicar_eng_min(test_clean)

# Dejar solo las 11 variables raw y las 6 features derivadas
train_min <- train_min |>
  select(any_of(c("id", "pobre", MIN_RAW_VARS, MIN_ENG_VARS)))

test_min <- test_min |>
  select(any_of(c("id", MIN_RAW_VARS, MIN_ENG_VARS)))

saveRDS(train_min, here(paths$processed, "train_min.rds"))
saveRDS(test_min,  here(paths$processed, "test_min.rds"))

cat("    train_min:", nrow(train_min), "x", ncol(train_min), "\n")
cat("    test_min: ", nrow(test_min),  "x", ncol(test_min),  "\n")

# Guardar listas de variables mínimas
saveRDS(MIN_RAW_VARS, here(paths$processed, "min_raw_vars.rds"))
saveRDS(c(MIN_RAW_VARS, MIN_ENG_VARS), here(paths$processed, "top_vars.rds"))

# ============================================================
# TABLAS DE IMPORTANCIA PARA METODOLOGÍA
# ============================================================

imp_top15 <- imp_agregada |>
  slice_head(n = 15) |>
  mutate(
    tipo = ifelse(var_base %in% MIN_ENG_VARS, "Feature derivada", "Variable raw")
  )

imp_17 <- imp_agregada |>
  filter(var_base %in% c(MIN_RAW_VARS, MIN_ENG_VARS)) |>
  mutate(
    tipo = ifelse(var_base %in% MIN_ENG_VARS, "Feature derivada", "Variable raw")
  ) |>
  arrange(desc(Gain_total)) |>
  mutate(
    gain_gap = lag(Gain_total) - Gain_total,
    gain_acum = cumsum(Gain_total),
    gain_acum_pct_subset = 100 * gain_acum / sum(Gain_total),
    gain_pct_full = 100 * Gain_total / sum(imp_agregada$Gain_total)
  )

imp_min <- imp_agregada |>
  filter(var_base %in% MIN_RAW_VARS) |>
  arrange(desc(Gain_total)) |>
  mutate(
    tipo = "Variable raw",
    gain_gap = lag(Gain_total) - Gain_total,
    gain_acum = cumsum(Gain_total),
    gain_acum_pct_subset = 100 * gain_acum / sum(Gain_total),
    gain_pct_full = 100 * Gain_total / sum(imp_agregada$Gain_total)
  )

write.csv(
  imp_top15,
  here(paths$tables, "top15_importancia.csv"),
  row.names = FALSE
)

write.csv(
  imp_17,
  here(paths$tables, "top17_importancia.csv"),
  row.names = FALSE
)

write.csv(
  imp_min,
  here(paths$models, "min_importancia.csv"),
  row.names = FALSE
)

write.csv(
  imp_min,
  here(paths$tables, "min_importancia.csv"),
  row.names = FALSE
)

tabla_cobertura <- data.frame(
  conjunto = c("11 raw", "17 completas"),
  n_vars = c(nrow(imp_min), nrow(imp_17)),
  gain_cubierto = c(sum(imp_min$Gain_total), sum(imp_17$Gain_total)),
  cobertura_pct = c(
    100 * sum(imp_min$Gain_total) / sum(imp_agregada$Gain_total),
    100 * sum(imp_17$Gain_total) / sum(imp_agregada$Gain_total)
  )
)

write.csv(
  tabla_cobertura,
  here(paths$tables, "cobertura_gain.csv"),
  row.names = FALSE
)

cat("\n>>> Tabla top 15 guardada en 04_outputs/tables/top15_importancia.csv\n")
cat(">>> Tabla top 17 guardada en 04_outputs/tables/top17_importancia.csv\n")
cat(">>> Tabla cobertura guardada en 04_outputs/tables/cobertura_gain.csv\n")

# ============================================================
# PASO 4 — Gráficas de metodología
# ============================================================

cat("\n>>> [4/4] Generando gráficas de metodología...\n")

# --- Gráfico 1: Top 15 variables globales -------------------
p_top15 <- imp_top15 |>
  mutate(var_base = factor(var_base, levels = rev(var_base))) |>
  ggplot(aes(x = var_base, y = Gain_total, fill = tipo)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = round(Gain_total, 3)),
            hjust = -0.1, size = 3.3, color = "#333333") +
  coord_flip() +
  scale_fill_manual(values = c(
    "Variable raw" = "#1B3A6B",
    "Feature derivada" = "#C8972B"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Top 15 variables por importancia",
    subtitle = "Ranking global de Gain total del XGBoost completo",
    x        = NULL,
    y        = "Gain total",
    fill     = NULL,
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle      = element_text(color = "#555555"),
    plot.caption       = element_text(color = "#888888", size = 8),
    legend.position    = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave(
  here(paths$figures, "top15_importancia.png"),
  plot = p_top15, width = 8, height = 5.5, dpi = 150
)

cat("    top15_importancia.png\n")

# --- Gráfico 2: Curva de concentración del gain -------------
x80 <- tabla_cortes$n_variables[tabla_cortes$corte_pct == 80][1]
x90 <- tabla_cortes$n_variables[tabla_cortes$corte_pct == 90][1]

p_cum <- imp_agregada |>
  ggplot(aes(x = rank, y = gain_acum_pct)) +
  geom_line(color = "#1B3A6B", linewidth = 1) +
  geom_point(color = "#1B3A6B", size = 1.2) +
  geom_hline(yintercept = c(80, 90), linetype = "dashed", color = "#C8972B") +
  geom_vline(xintercept = c(x80, x90), linetype = "dotted", color = "#666666") +
  annotate("text", x = x80, y = 82, label = paste0("80% en ", x80), hjust = -0.05, size = 3.2) +
  annotate("text", x = x90, y = 92, label = paste0("90% en ", x90), hjust = -0.05, size = 3.2) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  labs(
    title    = "Curva de concentración del gain acumulado",
    subtitle = "Muestra cuántas variables explican la mayor parte de la señal",
    x        = "Variables ordenadas por importancia",
    y        = "% acumulado del Gain",
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle   = element_text(color = "#555555"),
    plot.caption    = element_text(color = "#888888", size = 8)
  )

ggsave(
  here(paths$figures, "gain_acumulado.png"),
  plot = p_cum, width = 8, height = 5, dpi = 150
)

cat("    gain_acumulado.png\n")

# --- Gráfico 3: Gap entre variables consecutivas ------------
imp_gap <- imp_top15 |>
  mutate(
    gain_gap = ifelse(is.na(gain_gap), 0, gain_gap),
    var_base = factor(var_base, levels = rev(var_base))
  )

p_gap <- imp_gap |>
  ggplot(aes(x = var_base, y = gain_gap, fill = tipo)) +
  geom_col(width = 0.75) +
  coord_flip() +
  scale_fill_manual(values = c(
    "Variable raw" = "#1B3A6B",
    "Feature derivada" = "#C8972B"
  )) +
  labs(
    title    = "Brecha de importancia entre variables consecutivas",
    subtitle = "Permite ver el punto de quiebre del ranking",
    x        = NULL,
    y        = "Gain gap respecto a la variable superior",
    fill     = NULL,
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle      = element_text(color = "#555555"),
    plot.caption       = element_text(color = "#888888", size = 8),
    legend.position    = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave(
  here(paths$figures, "gap_top15_importancia.png"),
  plot = p_gap, width = 8, height = 5.5, dpi = 150
)

cat("    gap_top15_importancia.png\n")

# --- Gráfico 4: Importancia raw 11 --------------------------
p_raw11 <- imp_min |>
  mutate(var_base = factor(var_base, levels = rev(var_base))) |>
  ggplot(aes(x = var_base, y = gain_pct_full, fill = tipo)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = sprintf("%.1f%%", gain_pct_full)),
            hjust = -0.1, size = 3.3, color = "#333333") +
  coord_flip() +
  scale_fill_manual(values = c("Variable raw" = "#1B3A6B")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Importancia relativa — 11 variables raw",
    subtitle = "Participación en el gain total del XGBoost completo",
    x        = NULL,
    y        = "% del Gain total",
    fill     = NULL,
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle      = element_text(color = "#555555"),
    plot.caption       = element_text(color = "#888888", size = 8),
    legend.position    = "none",
    panel.grid.major.y = element_blank()
  )

ggsave(
  here(paths$figures, "min_importancia.png"),
  plot = p_raw11, width = 8, height = 5.5, dpi = 150
)

cat("    min_importancia.png\n")

# --- Gráfico 5: Importancia del modelo mínimo completo ------
p_17 <- imp_17 |>
  mutate(var_base = factor(var_base, levels = rev(var_base))) |>
  ggplot(aes(x = var_base, y = Gain_total, fill = tipo)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = round(Gain_total, 3)),
            hjust = -0.1, size = 3.3, color = "#333333") +
  coord_flip() +
  scale_fill_manual(values = c(
    "Variable raw" = "#1B3A6B",
    "Feature derivada" = "#C8972B"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Importancia del modelo mínimo completo",
    subtitle = "11 variables raw seleccionadas + 6 features derivadas",
    x        = NULL,
    y        = "Gain total",
    fill     = NULL,
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle      = element_text(color = "#555555"),
    plot.caption       = element_text(color = "#888888", size = 8),
    legend.position    = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave(
  here(paths$figures, "min_importancia_completa.png"),
  plot = p_17, width = 8, height = 5.5, dpi = 150
)

cat("    min_importancia_completa.png\n")

# --- Gráfico 6: Cobertura del gain --------------------------
p_cov <- tabla_cobertura |>
  mutate(conjunto = factor(conjunto, levels = conjunto)) |>
  ggplot(aes(x = conjunto, y = cobertura_pct, fill = conjunto)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.1f%%", cobertura_pct)),
            vjust = -0.4, size = 3.5, color = "#333333") +
  scale_fill_manual(values = c("11 raw" = "#1B3A6B", "17 completas" = "#C8972B")) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.08))) +
  labs(
    title    = "Cobertura del gain por conjunto de variables",
    subtitle = "Cuánta señal del modelo completo conserva cada subconjunto",
    x        = NULL,
    y        = "% del Gain total",
    fill     = NULL,
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle   = element_text(color = "#555555"),
    plot.caption    = element_text(color = "#888888", size = 8),
    legend.position = "none"
  )

ggsave(
  here(paths$figures, "cobertura_gain.png"),
  plot = p_cov, width = 7, height = 4.5, dpi = 150
)

cat("    cobertura_gain.png\n")

# ============================================================
# TABLAS DE PRESENTACIÓN ADICIONALES
# ============================================================

resumen_modelos_importancia <- data.frame(
  conjunto = c("11 raw", "17 completas"),
  n_vars = c(nrow(imp_min), nrow(imp_17)),
  gain_total = c(sum(imp_min$Gain_total), sum(imp_17$Gain_total)),
  cobertura_pct = c(
    100 * sum(imp_min$Gain_total) / sum(imp_agregada$Gain_total),
    100 * sum(imp_17$Gain_total) / sum(imp_agregada$Gain_total)
  )
)

write.csv(
  resumen_modelos_importancia,
  here(paths$tables, "resumen_modelos_importancia.csv"),
  row.names = FALSE
)

# ============================================================
# Guardar resumen de la base mínima
# ============================================================

saveRDS(
  imp_top15,
  here(paths$processed, "top15_importancia.rds")
)

saveRDS(
  imp_17,
  here(paths$processed, "top17_importancia.rds")
)

saveRDS(
  imp_min,
  here(paths$processed, "min_importancia.rds")
)

# ============================================================
# Limpieza y cierre
# ============================================================

rm(train_full, test_full, train_clean, test_clean,
   dummy_recipe, X_full, y_full, dtrain, params, xgb_full,
   imp_raw, imp_agregada, imp_top15, imp_17, imp_min,
   imp_gap, tabla_cortes, tabla_cobertura, resumen_modelos_importancia,
   todas_vars, mapear_dummy_a_base,
   MIN_RAW_VARS, MIN_ENG_VARS, DEPS_MIN,
   aplicar_eng_min, p_top15, p_cum, p_gap, p_raw11, p_17, p_cov,
   cortes_gain)
gc()

cat("\n>>> 00_features_min.R completado\n")
