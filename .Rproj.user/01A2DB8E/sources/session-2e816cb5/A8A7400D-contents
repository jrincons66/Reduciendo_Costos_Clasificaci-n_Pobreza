# ============================================================
# 00_features_min.R
# Encuesta mínima — selección top-8 por importancia XGBoost
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
#   - 00_data/01_processed/top8_vars.rds
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
# PASO 2 — Mapear dummies → variables base y seleccionar top-8
# ============================================================

cat(">>> [2/4] Seleccionando top-8 variables...\n")

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

top8_vars <- imp_agregada$var_base[1:TOP_N]

cat("\n>>> Top-8 variables seleccionadas:\n")
for (i in seq_along(top8_vars)) {
  cat(sprintf("    [%d] %-25s Gain = %.4f\n",
              i, top8_vars[i], imp_agregada$Gain_total[i]))
}

# Guardar lista y tabla
saveRDS(top8_vars, here(paths$processed, "top8_vars.rds"))
write.csv(imp_agregada[1:TOP_N, ],
          here(paths$models, "top8_importancia.csv"),
          row.names = FALSE)

# ============================================================
# PASO 3 — Construir dataset mínimo
# ============================================================

cat("\n>>> [3/4] Construyendo dataset mínimo...\n")

# Features engineered y sus dependencias en variables raw
FEATURES_ENG <- c(
  "ratio_dependencia", "hacinamiento", "educ_x_ocup",
  "rural_x_ocup", "formal_x_salud", "nper_sq", "edad_prom_sq",
  "mujeres_x_inact", "jefe_mujer_inact", "tasa_inactivos",
  "sin_ocupados", "educ_jefe_x_ocup", "calidad_empleo",
  "presion_habitacional", "jefe_vulnerable", "doble_proteccion",
  "ratio_mayores_65", "jefe_mayor_inactivo"
)

DEPS_ENG <- list(
  ratio_dependencia    = c("nper", "n_ocupados"),
  hacinamiento         = c("nper", "p5000"),
  educ_x_ocup          = c("nivel_educ_max", "tasa_ocupacion"),
  rural_x_ocup         = c("clase", "tasa_ocupacion"),
  formal_x_salud       = c("prop_cotiza_pension", "prop_afiliado_salud"),
  nper_sq              = c("nper"),
  edad_prom_sq         = c("edad_promedio"),
  mujeres_x_inact      = c("prop_mujeres", "tasa_ocupacion"),
  jefe_mujer_inact     = c("jefe_mujer", "tasa_ocupacion"),
  tasa_inactivos       = c("n_inactivos", "n_pet"),
  sin_ocupados         = c("n_ocupados"),
  educ_jefe_x_ocup     = c("educ_jefe", "ocup_jefe"),
  calidad_empleo       = c("horas_trabajo_prom", "prop_cotiza_pension"),
  presion_habitacional = c("p5010", "p5000"),
  jefe_vulnerable      = c("ocup_jefe", "educ_jefe"),
  doble_proteccion     = c("prop_cotiza_pension"),
  ratio_mayores_65     = c("n_mayores_65", "n_pet"),
  jefe_mayor_inactivo  = c("edad_jefe", "ocup_jefe")
)

top8_eng <- intersect(top8_vars, FEATURES_ENG)
top8_raw <- setdiff(top8_vars, FEATURES_ENG)

# Recalcular solo las engineered del top-8 desde train_clean
train_clean <- readRDS(here(paths$processed, "train_clean.rds"))
test_clean  <- readRDS(here(paths$processed, "test_clean.rds"))

eng_fns <- list(
  ratio_dependencia    = ~ mutate(.x, ratio_dependencia =
                                    (nper - n_ocupados) / pmax(nper, 1)),
  hacinamiento         = ~ mutate(.x, hacinamiento =
                                    nper / pmax(p5000, 1)),
  educ_x_ocup          = ~ mutate(.x, educ_x_ocup =
                                    as.integer(nivel_educ_max) * tasa_ocupacion),
  rural_x_ocup         = ~ mutate(.x, rural_x_ocup =
                                    as.integer(clase == "2") * tasa_ocupacion),
  formal_x_salud       = ~ mutate(.x, formal_x_salud =
                                    prop_cotiza_pension * prop_afiliado_salud),
  nper_sq              = ~ mutate(.x, nper_sq = nper^2),
  edad_prom_sq         = ~ mutate(.x, edad_prom_sq = edad_promedio^2),
  mujeres_x_inact      = ~ mutate(.x, mujeres_x_inact =
                                    prop_mujeres * (1 - tasa_ocupacion)),
  jefe_mujer_inact     = ~ mutate(.x, jefe_mujer_inact =
                                    jefe_mujer * (1 - tasa_ocupacion)),
  tasa_inactivos       = ~ mutate(.x, tasa_inactivos =
                                    n_inactivos / pmax(n_pet, 1)),
  sin_ocupados         = ~ mutate(.x, sin_ocupados =
                                    as.integer(n_ocupados == 0)),
  educ_jefe_x_ocup     = ~ mutate(.x, educ_jefe_x_ocup =
                                    as.integer(educ_jefe) * ocup_jefe),
  calidad_empleo       = ~ mutate(.x, calidad_empleo =
                                    horas_trabajo_prom * prop_cotiza_pension),
  presion_habitacional = ~ mutate(.x, presion_habitacional =
                                    p5010 / pmax(p5000, 1)),
  jefe_vulnerable      = ~ mutate(.x, jefe_vulnerable =
                                    as.integer(ocup_jefe == 0 & educ_jefe <= 3)),
  doble_proteccion     = ~ mutate(.x, doble_proteccion =
                                    prop_cotiza_pension^2),
  ratio_mayores_65     = ~ mutate(.x, ratio_mayores_65 =
                                    n_mayores_65 / pmax(n_pet, 1)),
  jefe_mayor_inactivo  = ~ mutate(.x, jefe_mayor_inactivo =
                                    as.integer(edad_jefe > 60 & ocup_jefe == 0))
)

aplicar_eng_min <- function(df) {
  for (fn_name in top8_eng) {
    df <- switch(fn_name,
                 ratio_dependencia    = mutate(df, ratio_dependencia =
                                                 (nper - n_ocupados) / pmax(nper, 1)),
                 hacinamiento         = mutate(df, hacinamiento =
                                                 nper / pmax(p5000, 1)),
                 educ_x_ocup          = mutate(df, educ_x_ocup =
                                                 as.integer(nivel_educ_max) * tasa_ocupacion),
                 rural_x_ocup         = mutate(df, rural_x_ocup =
                                                 as.integer(clase == "2") * tasa_ocupacion),
                 formal_x_salud       = mutate(df, formal_x_salud =
                                                 prop_cotiza_pension * prop_afiliado_salud),
                 nper_sq              = mutate(df, nper_sq = nper^2),
                 edad_prom_sq         = mutate(df, edad_prom_sq = edad_promedio^2),
                 mujeres_x_inact      = mutate(df, mujeres_x_inact =
                                                 prop_mujeres * (1 - tasa_ocupacion)),
                 jefe_mujer_inact     = mutate(df, jefe_mujer_inact =
                                                 jefe_mujer * (1 - tasa_ocupacion)),
                 tasa_inactivos       = mutate(df, tasa_inactivos =
                                                 n_inactivos / pmax(n_pet, 1)),
                 sin_ocupados         = mutate(df, sin_ocupados =
                                                 as.integer(n_ocupados == 0)),
                 educ_jefe_x_ocup     = mutate(df, educ_jefe_x_ocup =
                                                 as.integer(educ_jefe) * ocup_jefe),
                 calidad_empleo       = mutate(df, calidad_empleo =
                                                 horas_trabajo_prom * prop_cotiza_pension),
                 presion_habitacional = mutate(df, presion_habitacional =
                                                 p5010 / pmax(p5000, 1)),
                 jefe_vulnerable      = mutate(df, jefe_vulnerable =
                                                 as.integer(ocup_jefe == 0 & educ_jefe <= 3)),
                 doble_proteccion     = mutate(df, doble_proteccion =
                                                 prop_cotiza_pension^2),
                 ratio_mayores_65     = mutate(df, ratio_mayores_65 =
                                                 n_mayores_65 / pmax(n_pet, 1)),
                 jefe_mayor_inactivo  = mutate(df, jefe_mayor_inactivo =
                                                 as.integer(edad_jefe > 60 & ocup_jefe == 0)),
                 df  # default: no hacer nada
    )
  }
  df
}

train_min <- aplicar_eng_min(train_clean)
test_min  <- aplicar_eng_min(test_clean)

train_min <- train_min |> select(any_of(c("id", "pobre", top8_vars)))
test_min  <- test_min  |> select(any_of(c("id", top8_vars)))

saveRDS(train_min, here(paths$processed, "train_min.rds"))
saveRDS(test_min,  here(paths$processed, "test_min.rds"))

cat("    train_min:", nrow(train_min), "x", ncol(train_min), "\n")
cat("    test_min: ", nrow(test_min),  "x", ncol(test_min),  "\n")
# ============================================================
# PASO 4 — Gráfico de importancia
# ============================================================

cat("\n>>> [4/4] Generando gráfico de importancia...\n")

p_imp <- imp_agregada[1:TOP_N, ] |>
  mutate(var_base = reorder(var_base, Gain_total)) |>
  ggplot(aes(x = var_base, y = Gain_total)) +
  geom_col(fill = "#1B3A6B", width = 0.7) +
  geom_text(aes(label = round(Gain_total, 3)),
            hjust = -0.1, size = 3.5, color = "#333333") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Top-8 variables por importancia (XGBoost — Gain)",
    subtitle = "Variables seleccionadas para la encuesta mínima de focalización",
    x        = NULL,
    y        = "Gain acumulado",
    caption  = "Fuente: DANE MESE 2018, Bogotá"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B"),
    plot.subtitle      = element_text(color = "#555555"),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "top8_importancia.png"),
       plot = p_imp, width = 8, height = 5, dpi = 150)

cat("    Gráfico guardado: top8_importancia.png\n")

# ============================================================
# Limpiar entorno
# ============================================================
rm(train_full, test_full, train_clean, test_clean,
   dummy_recipe, X_full, y_full, dtrain, params, xgb_full,
   imp_raw, imp_agregada, todas_vars, mapear_dummy_a_base,
   FEATURES_ENG, DEPS_ENG, top8_eng, top8_raw,
   eng_fns, aplicar_eng_min, p_imp)
gc()

cat("\n>>> 00_features_min.R completado\n")