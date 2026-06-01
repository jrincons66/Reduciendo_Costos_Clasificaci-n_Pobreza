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
#   extrae la importancia por Gain, selecciona las top-8
#   variables y construye train_min.rds / test_min.rds con
#   solo esas variables. Estos datasets alimentan los modelos
#   de la encuesta mínima.
#
# Outputs:
#   - 00_data/01_processed/train_min.rds
#   - 00_data/01_processed/test_min.rds
#   - 00_data/01_processed/top_vars.rds
#   - 02_models/top8_importancia.csv
#   - 04_outputs/figures/top8_importancia.png
# ============================================================

train_full <- readRDS(here(paths$processed, "train_features.rds"))
test_full  <- readRDS(here(paths$processed, "test_features.rds"))

train_full <- train_full |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

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
imp_raw <- xgb.importance(feature_names = colnames(X_full),
                          model         = xgb_full)

# ============================================================
# PASO 2 — Mapear dummies → variables base y seleccionar top
# ============================================================

cat(">>> [2/4] Seleccionando top variables...\n")

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
  arrange(desc(Gain_total))

imp_agregada <- imp_agregada |>
  mutate(
    gain_gap = lag(Gain_total) - Gain_total
  )

imp_agregada <- imp_agregada |>
  mutate(
    gain_acum = cumsum(Gain_total),
    gain_acum_pct = 100 * gain_acum / sum(Gain_total)
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
  df <- df |>
    mutate(
      ratio_dependencia = (nper - n_ocupados) / pmax(nper, 1),
      hacinamiento      = nper / pmax(p5000, 1),
      nper_sq           = nper^2,
      edad_prom_sq      = edad_promedio^2,
      calidad_empleo    = horas_trabajo_prom * prop_cotiza_pension,
      doble_proteccion  = prop_cotiza_pension^2
    )
  
  df
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

# Guardar lista de variables mínimas
saveRDS(MIN_RAW_VARS, here(paths$processed, "min_raw_vars.rds"))

saveRDS(
  c(MIN_RAW_VARS, MIN_ENG_VARS),
  here(paths$processed, "top_vars.rds")
)

# Guardar tabla de importancia solo para las variables mínimas
imp_min <- imp_agregada |>
  filter(var_base %in% MIN_RAW_VARS) |>
  arrange(desc(Gain_total))

write.csv(
  imp_min,
  here(paths$models, "min_importancia.csv"),
  row.names = FALSE
)

# ============================================================
# PASO 4 — Gráfico de importancia
# ============================================================

cat("\n>>> [4/4] Generando gráfico de importancia...\n")

p_imp <- imp_min |>
  mutate(var_base = reorder(var_base, Gain_total)) |>
  ggplot(aes(x = var_base, y = Gain_total)) +
  geom_col(fill = "#1B3A6B", width = 0.7) +
  geom_text(aes(label = round(Gain_total, 3)),
            hjust = -0.1, size = 3.5, color = "#333333") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Variables mínimas por importancia (XGBoost — Gain)",
    subtitle = "Variables originales mínimas y features reconstruibles",
    x        = NULL,
    y        = "Gain",
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle      = element_text(color = "#555555"),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(
  here(paths$figures, "min_importancia.png"),
  plot = p_imp, width = 8, height = 5, dpi = 150
)

cat("    Gráfico guardado: min_importancia.png\n")

# ============================================================
# Limpiar entorno
# ============================================================
rm(train_full, test_full, train_clean, test_clean,
   dummy_recipe, X_full, y_full, dtrain, params, xgb_full,
   imp_raw, imp_agregada, imp_min, todas_vars, mapear_dummy_a_base,
   MIN_RAW_VARS, MIN_ENG_VARS, DEPS_MIN,
   aplicar_eng_min, p_imp)
gc()
