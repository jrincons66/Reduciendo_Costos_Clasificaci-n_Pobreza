# ============================================================
# 05_Comparativa.R
# Tabla comparativa final - datos reales vs sintéticos
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
#
# Descripción:
#   Consolida resultados de todos los modelos (reales y
#   sintéticos), genera tabla comparativa, tablas resumen y
#   gráficos para el artículo final.
#
# Outputs:
#   - 04_outputs/tables/comparativa_final.csv
#   - 04_outputs/tables/comparativa_wide.csv
#   - 04_outputs/tables/resumen_importancia_raw11.csv
#   - 04_outputs/tables/resumen_importancia_17.csv
#   - 04_outputs/figures/comparativa_f1.png
#   - 04_outputs/figures/comparativa_diferencia.png
#   - 04_outputs/figures/importancia_raw11.png
#   - 04_outputs/figures/importancia_17.png
#   - 04_outputs/figures/comparativa_panel.png
# ============================================================

TIPO <- "05_Comparativa"

cat("\n─────────────────────────────────────────────────────────\n")
cat("  Comparativa final - reales vs sintéticos\n")
cat("─────────────────────────────────────────────────────────\n")

# ============================================================
# PASO 1 — Cargar y limpiar resultados
# ============================================================

cat("\n>>> [1/4] Cargando resultados...\n")

# Limpiar duplicados del log
log_cv <- read.csv(here(paths$models, "log.csv")) |>
  group_by(tipo, modelo) |>
  slice_max(order_by = timestamp, n = 1) |>
  ungroup()

write.csv(log_cv, here(paths$models, "log.csv"), row.names = FALSE)

# Mejor F1 por tipo en datos reales
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

# Resultados sintéticos
resultados_sint <- readRDS(here(paths$processed, "resultados_sint.rds")) |>
  mutate(datos = "Sintéticos") |>
  select(modelo, f1, threshold, datos)

cat("    Modelos reales cargados:     ", nrow(mejores_reales), "\n")
cat("    Modelos sintéticos cargados: ", nrow(resultados_sint), "\n")

# ============================================================
# PASO 2 — Tabla comparativa
# ============================================================

cat("\n>>> [2/4] Construyendo tabla comparativa...\n")

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
  pivot_wider(
    names_from  = datos,
    values_from = f1,
    values_fn   = max
  ) |>
  mutate(
    diferencia   = round(`Reales (DANE)` - `Sintéticos`, 4),
    gap_relativo = round(diferencia / `Reales (DANE)` * 100, 2)
  ) |>
  arrange(desc(`Reales (DANE)`))

cat("\n>>> Tabla comparativa:\n")
print(comparativa)

cat("\n>>> Diferencia reales vs sintéticos:\n")
print(comparativa_wide)

write.csv(comparativa,
          here(paths$tables, "comparativa_final.csv"),
          row.names = FALSE)

write.csv(comparativa_wide,
          here(paths$tables, "comparativa_wide.csv"),
          row.names = FALSE)

cat("\n>>> CSVs guardados en 04_outputs/tables/\n")

# ============================================================
# PASO 3 — Gráficos de desempeño
# ============================================================

cat("\n>>> [3/4] Generando gráficos de desempeño...\n")

colores_datos <- c("Reales (DANE)" = "#1B3A6B",
                   "Sintéticos"    = "#C8972B")

# --- Gráfico 1: F1 por modelo y tipo de datos ---------------
p1 <- comparativa |>
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
  scale_y_continuous(limits = c(0, 0.80),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(
    title    = "F1-score por modelo",
    subtitle = "Datos reales (DANE MESE 2018) vs datos sintéticos",
    x        = NULL,
    y        = "F1-score (CV 5 folds, threshold OOF)",
    fill     = NULL,
    caption  = "Encuesta mínima: 11 variables raw + 6 features derivadas | Tópicos de IA - Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", color = "#1B3A6B", size = 13),
    plot.subtitle   = element_text(color = "#555555", size = 10),
    plot.caption    = element_text(color = "#888888", size = 8),
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "comparativa_f1.png"),
       plot = p1, width = 8, height = 5, dpi = 150)
cat("    comparativa_f1.png\n")

# --- Gráfico 2: Diferencia F1 reales vs sintéticos ----------
p2 <- comparativa_wide |>
  ggplot(aes(x = reorder(modelo, diferencia),
             y = diferencia,
             fill = diferencia >= 0)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "#888888", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%+.3f", diferencia)),
            hjust = ifelse(comparativa_wide$diferencia >= 0, -0.1, 1.1),
            size = 3.5, color = "#333333") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE"  = "#1B3A6B",
                               "FALSE" = "#C8972B"),
                    guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.2, 0.2))) +
  labs(
    title    = "Brecha de generalización por modelo",
    subtitle = "F1 reales - F1 sintéticos  |  + = mejor en datos reales",
    x        = NULL,
    y        = "Diferencia F1",
    caption  = "Tópicos de IA - Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B", size = 13),
    plot.subtitle      = element_text(color = "#555555", size = 10),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "comparativa_diferencia.png"),
       plot = p2, width = 7, height = 4, dpi = 150)
cat("    comparativa_diferencia.png\n")

# ============================================================
# PASO 4 — Importancias: raw 11 y modelo completo 17
# ============================================================

cat("\n>>> [4/4] Generando gráficas de importancia...\n")

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

MIN_ENG_VARS <- c(
  "ratio_dependencia",
  "hacinamiento",
  "nper_sq",
  "edad_prom_sq",
  "calidad_empleo",
  "doble_proteccion"
)

# Importancia completa del XGB global.
# Debe existir como archivo guardado desde 00_features_min.R.
archivo_imp_full <- "imp_agregada.csv"
if (!file.exists(here(paths$models, archivo_imp_full))) {
  stop(
    "No encuentro ", archivo_imp_full, " en 04_outputs/02_models/.\n",
    "Guarda la tabla completa de importancia del XGBoost global como imp_agregada.csv."
  )
}

imp_full <- read.csv(here(paths$models, archivo_imp_full))

if (!("var_base" %in% names(imp_full))) {
  stop("imp_agregada.csv debe contener la columna 'var_base'.")
}

if (!("Gain_total" %in% names(imp_full))) {
  if ("Gain" %in% names(imp_full)) {
    imp_full <- imp_full |>
      rename(Gain_total = Gain)
  } else {
    stop("imp_agregada.csv debe contener 'Gain_total' o 'Gain'.")
  }
}

full_gain_total <- sum(imp_full$Gain_total, na.rm = TRUE)

preparar_importancia <- function(df, full_total) {
  df |>
    arrange(desc(Gain_total)) |>
    mutate(
      gain_gap = lag(Gain_total) - Gain_total,
      gain_acum = cumsum(Gain_total),
      gain_acum_pct_subset = 100 * gain_acum / sum(Gain_total),
      gain_acum_pct_full   = 100 * gain_acum / full_total,
      gain_pct_full        = 100 * Gain_total / full_total
    )
}

# Tabla y gráfico para las 11 raw
imp_raw11 <- imp_full |>
  filter(var_base %in% MIN_RAW_VARS) |>
  preparar_importancia(full_gain_total)

write.csv(
  imp_raw11,
  here(paths$tables, "resumen_importancia_raw11.csv"),
  row.names = FALSE
)

# Tabla y gráfico para las 17 completas
imp_17 <- imp_full |>
  filter(var_base %in% c(MIN_RAW_VARS, MIN_ENG_VARS)) |>
  preparar_importancia(full_gain_total)

write.csv(
  imp_17,
  here(paths$tables, "resumen_importancia_17.csv"),
  row.names = FALSE
)

cat("\n>>> Resumen raw 11:\n")
print(imp_raw11)

cat("\n>>> Resumen 17 completas:\n")
print(imp_17)

# --- Gráfico 3: Importancia raw 11 ---------------------------
p3 <- imp_raw11 |>
  mutate(var_base = reorder(var_base, gain_pct_full)) |>
  ggplot(aes(x = var_base, y = gain_pct_full)) +
  geom_col(fill = "#1B3A6B", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", gain_pct_full)),
            hjust = -0.1, size = 3.5, color = "#333333") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Importancia relativa — 11 variables raw",
    subtitle = "Participación en el gain total del XGBoost completo",
    x        = NULL,
    y        = "% del Gain total",
    caption  = "Fuente: DANE MESE 2018, Bogotá | Tópicos de IA - Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B", size = 13),
    plot.subtitle      = element_text(color = "#555555", size = 10),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "importancia_raw11.png"),
       plot = p3, width = 8, height = 5, dpi = 150)
cat("    importancia_raw11.png\n")

# --- Gráfico 4: Importancia 17 completas --------------------
p4 <- imp_17 |>
  mutate(var_base = reorder(var_base, gain_pct_full)) |>
  ggplot(aes(x = var_base, y = gain_pct_full)) +
  geom_col(fill = "#C8972B", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", gain_pct_full)),
            hjust = -0.1, size = 3.5, color = "#333333") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Importancia relativa — 17 variables del modelo mínimo",
    subtitle = "11 variables raw + 6 features derivadas",
    x        = NULL,
    y        = "% del Gain total",
    caption  = "Fuente: DANE MESE 2018, Bogotá | Tópicos de IA - Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B", size = 13),
    plot.subtitle      = element_text(color = "#555555", size = 10),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "importancia_17.png"),
       plot = p4, width = 8, height = 5, dpi = 150)
cat("    importancia_17.png\n")

# ============================================================
# TABLAS ADICIONALES ÚTILES PARA RESULTADOS
# ============================================================

cat("\n>>> Generando tablas adicionales...\n")

# Resumen compacto de desempeño
resumen_modelos <- comparativa |>
  group_by(datos) |>
  slice_max(order_by = f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(datos, modelo, f1, threshold) |>
  arrange(datos)

write.csv(
  resumen_modelos,
  here(paths$tables, "resumen_modelos.csv"),
  row.names = FALSE
)

# Cobertura del gain por subconjunto
cobertura_gain <- data.frame(
  conjunto = c("11 raw", "17 completas"),
  n_vars    = c(nrow(imp_raw11), nrow(imp_17)),
  gain_total_subconjunto = c(sum(imp_raw11$Gain_total), sum(imp_17$Gain_total)),
  gain_total_modelo      = c(full_gain_total, full_gain_total),
  cobertura_pct = c(
    100 * sum(imp_raw11$Gain_total) / full_gain_total,
    100 * sum(imp_17$Gain_total) / full_gain_total
  )
)

write.csv(
  cobertura_gain,
  here(paths$tables, "cobertura_gain.csv"),
  row.names = FALSE
)

cat("\n>>> Tablas guardadas:\n")
cat("    · comparativa_final.csv\n")
cat("    · comparativa_wide.csv\n")
cat("    · resumen_importancia_raw11.csv\n")
cat("    · resumen_importancia_17.csv\n")
cat("    · resumen_modelos.csv\n")
cat("    · cobertura_gain.csv\n")

# ============================================================
# PASO 5 — Panel combinado para artículo
# ============================================================

cat("\n>>> [5/5] Generando panel combinado...\n")

panel <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title   = "Clasificación de pobreza con encuesta mínima",
    subtitle = paste(
      "11 variables raw + 6 features derivadas |",
      "DANE MESE 2018, Bogotá"
    ),
    caption = paste(
      "Jose Rincón · Lucas Rodríguez · María Paula Osuna |",
      "Tópicos de IA · Uniandes 2026 · Profesor: Álvaro Riascos"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", color = "#1B3A6B",
                                   size = 14, hjust = 0.5),
      plot.subtitle = element_text(color = "#555555", size = 10,
                                   hjust = 0.5),
      plot.caption  = element_text(color = "#888888", size = 8,
                                   hjust = 0.5)
    )
  )

ggsave(here(paths$figures, "comparativa_panel.png"),
       plot = panel, width = 14, height = 12, dpi = 150)
cat("    comparativa_panel.png\n")

# ============================================================
# RESUMEN FINAL EN CONSOLA
# ============================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║   RESULTADOS FINALES                                    ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n")

cat("\n  Datos reales (DANE MESE 2018, Bogotá):\n")
comparativa |>
  filter(datos == "Reales (DANE)") |>
  arrange(desc(f1)) |>
  as.data.frame() |>
  print()

cat("\n  Datos sintéticos:\n")
comparativa |>
  filter(datos == "Sintéticos") |>
  arrange(desc(f1)) |>
  as.data.frame() |>
  print()

cat("\n  Brecha de generalización:\n")
comparativa_wide |>
  as.data.frame() |>
  print()

cat("\n  Cobertura del gain total del modelo completo:\n")
cobertura_gain |>
  as.data.frame() |>
  print()

cat("\n  Mejor modelo overall:\n")
comparativa |>
  slice_max(f1, n = 1, with_ties = FALSE) |>
  as.data.frame() |>
  print()

cat("\n>>> Outputs guardados:\n")
cat("    · 04_outputs/tables/comparativa_final.csv\n")
cat("    · 04_outputs/tables/comparativa_wide.csv\n")
cat("    · 04_outputs/tables/resumen_importancia_raw11.csv\n")
cat("    · 04_outputs/tables/resumen_importancia_17.csv\n")
cat("    · 04_outputs/tables/resumen_modelos.csv\n")
cat("    · 04_outputs/tables/cobertura_gain.csv\n")
cat("    · 04_outputs/figures/comparativa_f1.png\n")
cat("    · 04_outputs/figures/comparativa_diferencia.png\n")
cat("    · 04_outputs/figures/importancia_raw11.png\n")
cat("    · 04_outputs/figures/importancia_17.png\n")
cat("    · 04_outputs/figures/comparativa_panel.png\n")

# --- Limpiar entorno ----------------------------------------
rm(log_cv, mejores_reales, resultados_sint,
   comparativa, comparativa_wide, imp_full,
   imp_raw11, imp_17, cobertura_gain, resumen_modelos,
   colores_datos, p1, p2, p3, p4, panel, TIPO,
   MIN_RAW_VARS, MIN_ENG_VARS, archivo_imp_full, full_gain_total)
gc()

cat("\n>>> 05_Comparativa.R completado\n")
