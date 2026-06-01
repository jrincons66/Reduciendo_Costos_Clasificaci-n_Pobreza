# ============================================================
# 05_Comparativa.R
# Tabla comparativa final — datos reales vs sintéticos
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
#
# Descripción:
#   Consolida resultados de todos los modelos (reales y
#   sintéticos), genera tabla comparativa y gráficos para
#   el artículo final.
#
# Outputs:
#   - 04_outputs/tables/comparativa_final.csv
#   - 04_outputs/figures/comparativa_f1.png
#   - 04_outputs/figures/comparativa_diferencia.png
#   - 04_outputs/figures/top8_importancia_final.png
#   - 04_outputs/figures/comparativa_panel.png
# ============================================================

TIPO <- "05_Comparativa"

cat("\n─────────────────────────────────────────────────────────\n")
cat("  Comparativa final — reales vs sintéticos\n")
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
      tipo == "01_Logit"         ~ "Logit",
      tipo == "02_RandomForest"  ~ "Random Forest",
      tipo == "03_XGBoost"       ~ "XGBoost"
    ),
    datos = "Reales (DANE)"
  ) |>
  select(modelo, f1 = cv_f1, threshold, datos)

# Resultados sintéticos
resultados_sint <- readRDS(here(paths$processed, "resultados_sint.rds")) |>
  mutate(datos = "Sintéticos") |>
  select(modelo, f1, threshold, datos)

cat("    Modelos reales cargados:    ", nrow(mejores_reales), "\n")
cat("    Modelos sintéticos cargados:", nrow(resultados_sint), "\n")

# ============================================================
# PASO 2 — Tabla comparativa
# ============================================================

cat("\n>>> [2/4] Construyendo tabla comparativa...\n")

comparativa <- bind_rows(mejores_reales, resultados_sint) |>
  mutate(
    datos     = factor(datos,
                       levels = c("Reales (DANE)", "Sintéticos")),
    f1        = round(f1, 4),
    threshold = round(threshold, 3)
  ) |>
  arrange(datos, desc(f1))

# Tabla wide para análisis
comparativa_wide <- comparativa |>
  group_by(modelo, datos) |>
  slice_max(order_by = f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(modelo, f1, datos) |>
  pivot_wider(names_from  = datos,
              values_from = f1,
              values_fn   = max) |>
  mutate(
    diferencia   = round(`Reales (DANE)` - `Sintéticos`, 4),
    gap_relativo = round(diferencia / `Reales (DANE)` * 100, 2)
  ) |>
  arrange(desc(`Reales (DANE)`))

cat("\n>>> Tabla comparativa:\n")
print(comparativa)

cat("\n>>> Diferencia reales vs sintéticos:\n")
print(comparativa_wide)

# Guardar ambas
write.csv(comparativa,
          here(paths$tables, "comparativa_final.csv"),
          row.names = FALSE)
write.csv(comparativa_wide,
          here(paths$tables, "comparativa_wide.csv"),
          row.names = FALSE)

cat("\n>>> CSVs guardados en 04_outputs/tables/\n")

# ============================================================
# PASO 3 — Gráficos individuales
# ============================================================

cat("\n>>> [3/4] Generando gráficos...\n")

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
    caption  = "Encuesta mínima: top-8 variables | Tópicos de IA — Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", color = "#1B3A6B",
                                   size = 13),
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
    subtitle = "F1 reales − F1 sintéticos  |  + = mejor en datos reales",
    x        = NULL,
    y        = "Diferencia F1",
    caption  = "Tópicos de IA — Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B",
                                      size = 13),
    plot.subtitle      = element_text(color = "#555555", size = 10),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "comparativa_diferencia.png"),
       plot = p2, width = 7, height = 4, dpi = 150)
cat("    comparativa_diferencia.png\n")

# --- Gráfico 3: Importancia top-8 ---------------------------
imp_top8 <- read.csv(here(paths$models, "top8_importancia.csv"))

p3 <- imp_top8 |>
  mutate(
    var_base  = reorder(var_base, Gain_total),
    pct_gain  = Gain_total / sum(Gain_total) * 100
  ) |>
  ggplot(aes(x = var_base, y = pct_gain)) +
  geom_col(fill = "#1B3A6B", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", pct_gain)),
            hjust = -0.1, size = 3.5, color = "#333333") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Importancia relativa — top-8 variables",
    subtitle = "Contribución al Gain total del XGBoost completo",
    x        = NULL,
    y        = "% del Gain total",
    caption  = "Fuente: DANE MESE 2018, Bogotá | Tópicos de IA — Uniandes 2026"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", color = "#1B3A6B",
                                      size = 13),
    plot.subtitle      = element_text(color = "#555555", size = 10),
    plot.caption       = element_text(color = "#888888", size = 8),
    panel.grid.major.y = element_blank()
  )

ggsave(here(paths$figures, "top8_importancia_final.png"),
       plot = p3, width = 8, height = 5, dpi = 150)
cat("    top8_importancia_final.png\n")

# ============================================================
# PASO 4 — Panel combinado para artículo
# ============================================================

cat("\n>>> [4/4] Generando panel combinado...\n")

panel <- (p1 | p2) / p3 +
  plot_annotation(
    title   = "Reducción de Costos de Clasificación de Pobreza en Colombia",
    subtitle = paste(
      "Encuesta mínima (top-8 variables observables) |",
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
       plot = panel, width = 14, height = 10, dpi = 150)
cat("    comparativa_panel.png\n")

# ============================================================
# RESUMEN FINAL EN CONSOLA
# ============================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║   RESULTADOS FINALES                                     ║\n")
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

cat("\n  Mejor modelo overall:\n")
comparativa |>
  slice_max(f1, n = 1, with_ties = FALSE) |>
  as.data.frame() |>
  print()

cat("\n>>> Outputs guardados:\n")
cat("    · 04_outputs/tables/comparativa_final.csv\n")
cat("    · 04_outputs/tables/comparativa_wide.csv\n")
cat("    · 04_outputs/figures/comparativa_f1.png\n")
cat("    · 04_outputs/figures/comparativa_diferencia.png\n")
cat("    · 04_outputs/figures/top8_importancia_final.png\n")
cat("    · 04_outputs/figures/comparativa_panel.png\n")

# --- Limpiar entorno ----------------------------------------
rm(log_cv, mejores_reales, resultados_sint,
   comparativa, comparativa_wide, imp_top8,
   colores_datos, p1, p2, p3, panel, TIPO)
gc()

cat("\n>>> 05_Comparativa.R completado\n")