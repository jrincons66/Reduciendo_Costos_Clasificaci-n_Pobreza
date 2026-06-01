# ============================================================
# 00_clean.R
# Limpieza general del dataset y exclusión de leakage
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
#
# Descripción:
#   Carga los 4 archivos del DANE MESE 2018 (train/test ×
#   hogares/personas), excluye todas las variables de ingreso
#   para evitar data leakage, agrega personas al nivel de
#   hogar y guarda train_clean.rds / test_clean.rds.
# ============================================================

# --- Verificar archivos -------------------------------------
archivos_necesarios <- c(
  "train_hogares.csv", "test_hogares.csv",
  "train_personas.csv", "test_personas.csv"
)

faltantes <- archivos_necesarios[
  !file.exists(here(paths$raw, archivos_necesarios))
]

if (length(faltantes) > 0) {
  stop(
    "\n========================================================\n",
    "  Faltan archivos en 00_data/00_raw/:\n",
    paste(" -", faltantes, collapse = "\n"), "\n\n",
    "  Descárgalos desde Kaggle:\n",
    "  kaggle competitions download -c uniandes-bdml-2026-10-ps-2\n",
    "  y descomprime los CSV en 00_data/00_raw/\n",
    "========================================================\n"
  )
} else {
  cat(">>> Archivos encontrados en 00_data/00_raw/\n")
}

# --- Cargar datos -------------------------------------------
train_h <- read.csv(here(paths$raw, "train_hogares.csv"))
test_h  <- read.csv(here(paths$raw, "test_hogares.csv"))
train_p <- read.csv(here(paths$raw, "train_personas.csv"))
test_p  <- read.csv(here(paths$raw, "test_personas.csv"))

cat("Dimensiones originales:\n")
cat("  train_hogares: ", dim(train_h), "\n")
cat("  test_hogares:  ", dim(test_h),  "\n")
cat("  train_personas:", dim(train_p), "\n")
cat("  test_personas: ", dim(test_p),  "\n")

# --- Variables a excluir — hogares --------------------------
# Se excluyen todas las variables de ingreso y líneas de
# pobreza para evitar leakage. El modelo debe predecir
# pobreza SIN conocer el ingreso del hogar.
excluir_hogares <- c(
  # Ingresos y líneas de pobreza
  "Ingtotug", "Ingtotugarr", "Ingpcug",
  "Lp", "Li",
  # Conteos derivados del outcome
  "Indigente", "Npobres", "Nindigentes",
  # Factores de expansión y administrativas
  "Fex_c", "Fex_dpto", "Mes",
  # Montos de arriendo imputado
  "P5100", "P5130", "P5140"
)

# --- Variables a excluir — personas -------------------------
excluir_personas <- c(
  # Estrato (no disponible en test)
  "Estrato1",
  # Salarios y componentes de ingreso laboral
  "P6500", "P6510s1", "P6510s2",
  "P6545s1", "P6545s2",
  "P6580s1", "P6580s2",
  "P6585s1a1", "P6585s1a2",
  "P6585s2a1", "P6585s2a2",
  "P6585s3a1", "P6585s3a2",
  "P6585s4a1", "P6585s4a2",
  "P6590s1", "P6600s1", "P6610s1", "P6620s1",
  "P6630s1a1", "P6630s2a1", "P6630s3a1",
  "P6630s4a1", "P6630s6a1",
  "P6750", "P6760", "P550", "P7070",
  "P7140s1", "P7140s2",
  "P7422s1", "P7472s1",
  "P7500s1", "P7500s1a1", "P7500s2a1", "P7500s3a1",
  "P7510s1a1", "P7510s2a1", "P7510s3a1",
  "P7510s5a1", "P7510s6a1", "P7510s7a1",
  # Ingresos agregados e imputados
  "Impa", "Isa", "Ie", "Imdi",
  "Iof1", "Iof2", "Iof3h", "Iof3i", "Iof6",
  # Flags de imputación
  "Cclasnr2", "Cclasnr3", "Cclasnr4", "Cclasnr5",
  "Cclasnr6", "Cclasnr7", "Cclasnr8", "Cclasnr11",
  # Versiones imputadas
  "Impaes", "Isaes", "Iees", "Imdies",
  "Iof1es", "Iof2es", "Iof3hes", "Iof3ies", "Iof6es",
  # Ingresos totales
  "Ingtotob", "Ingtotes", "Ingtot",
  # Administrativas
  "Fex_c", "Fex_dpto", "Mes"
)

# --- Aplicar exclusiones ------------------------------------
train_h <- train_h |> select(-any_of(excluir_hogares))
test_h  <- test_h  |> select(-any_of(excluir_hogares))
train_p <- train_p |> select(-any_of(excluir_personas))
test_p  <- test_p  |> select(-any_of(excluir_personas))

# --- Limpiar nombres de columnas ----------------------------
train_h <- train_h |> janitor::clean_names()
test_h  <- test_h  |> janitor::clean_names()
train_p <- train_p |> janitor::clean_names()
test_p  <- test_p  |> janitor::clean_names()

# --- Agregar personas al nivel de hogar ---------------------
agregar_personas <- function(df_personas) {
  df_personas |>
    mutate(
      p6210 = na_if(p6210, 9),
      p6090 = na_if(p6090, 9)
    ) |>
    group_by(id) |>
    summarise(
      # Demografía
      prop_mujeres       = mean(p6020 == 2, na.rm = TRUE),
      edad_promedio      = mean(p6040, na.rm = TRUE),
      edad_max           = max(p6040,  na.rm = TRUE),
      edad_min           = min(p6040,  na.rm = TRUE),
      n_menores_18       = sum(p6040 < 18, na.rm = TRUE),
      n_mayores_65       = sum(p6040 > 65, na.rm = TRUE),
      jefe_mujer         = as.integer(
        any(p6050 == 1 & p6020 == 2, na.rm = TRUE)),
      
      # Educación
      nivel_educ_max = {
        val <- suppressWarnings(max(p6210, na.rm = TRUE))
        factor(ifelse(is.infinite(val), NA_real_, val), levels = 1:6)
      },
      n_sin_educacion    = sum(p6210 == 1, na.rm = TRUE),
      
      # Estado laboral
      n_ocupados         = sum(oc  == 1, na.rm = TRUE),
      n_desocupados      = sum(des == 1, na.rm = TRUE),
      n_inactivos        = sum(ina == 1, na.rm = TRUE),
      n_pet              = sum(pet == 1, na.rm = TRUE),
      tasa_ocupacion     = sum(oc  == 1, na.rm = TRUE) /
        pmax(sum(pet == 1, na.rm = TRUE), 1),
      
      # Características laborales
      prop_cuenta_propia   = mean(p6430 == 4,       na.rm = TRUE),
      horas_trabajo_prom   = mean(p6800,             na.rm = TRUE),
      prop_empresa_pequena = mean(p6870 %in% c(1,2), na.rm = TRUE),
      prop_segundo_trabajo = mean(p7040 == 1,        na.rm = TRUE),
      
      # Seguridad social
      prop_cotiza_pension  = mean(p6920 == 1, na.rm = TRUE),
      prop_afiliado_salud  = mean(p6090 == 1, na.rm = TRUE),
      prop_reg_subsidiado  = mean(p6100 == 3, na.rm = TRUE),
      
      # Jefe del hogar
      educ_jefe  = first(p6210[p6050 == 1]),
      ocup_jefe  = first(oc[p6050 == 1]),
      edad_jefe  = first(p6040[p6050 == 1]),
      
      .groups = "drop"
    )
}

cat("\n>>> Agregando personas al nivel de hogar...\n")
train_p_agg <- agregar_personas(train_p)
test_p_agg  <- agregar_personas(test_p)

# --- Join hogares + personas --------------------------------
train <- train_h |> left_join(train_p_agg, by = "id")
test  <- test_h  |> left_join(test_p_agg,  by = "id")

# --- Factores -----------------------------------------------
vars_factor <- c("clase", "dominio", "depto", "p5090")

train <- train |> mutate(across(all_of(vars_factor), as.factor))
test  <- test  |> mutate(across(all_of(vars_factor), as.factor))

# --- Imputación ---------------------------------------------
# Variables laborales: NA = nadie en el hogar → 0
vars_laborales <- c(
  "prop_cuenta_propia", "horas_trabajo_prom",
  "prop_segundo_trabajo", "prop_cotiza_pension",
  "prop_reg_subsidiado", "prop_empresa_pequena"
)
train <- train |> mutate(across(all_of(vars_laborales), ~ replace_na(., 0)))
test  <- test  |> mutate(across(all_of(vars_laborales), ~ replace_na(., 0)))

# nivel_educ_max: sin info → primaria (2)
train <- train |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "2"))
test  <- test  |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "2"))

# educ_jefe → primaria (2)
train <- train |> mutate(educ_jefe = replace_na(educ_jefe, 2))
test  <- test  |> mutate(educ_jefe = replace_na(educ_jefe, 2))

# ocup_jefe → 0
train <- train |> mutate(ocup_jefe = replace_na(ocup_jefe, 0))
test  <- test  |> mutate(ocup_jefe = replace_na(ocup_jefe, 0))

# prop_afiliado_salud → 0
train <- train |> mutate(prop_afiliado_salud = replace_na(prop_afiliado_salud, 0))
test  <- test  |> mutate(prop_afiliado_salud = replace_na(prop_afiliado_salud, 0))

# --- Winsorización ------------------------------------------
winsorizr <- function(x, p = 0.99) {
  cap <- quantile(x, p, na.rm = TRUE)
  pmin(x, cap)
}

vars_winsorizar <- c(
  "p5000", "p5010", "nper", "npersug",
  "n_pet", "n_menores_18", "n_ocupados",
  "n_inactivos", "horas_trabajo_prom"
)

train <- train |> mutate(across(all_of(vars_winsorizar), winsorizr))
test  <- test  |> mutate(across(all_of(vars_winsorizar), winsorizr))

# --- Guardar ------------------------------------------------
saveRDS(train, here(paths$processed, "train_clean.rds"))
saveRDS(test,  here(paths$processed, "test_clean.rds"))

cat("\n>>> 00_clean.R completado\n")
cat("    train:", nrow(train), "filas x", ncol(train), "columnas\n")
cat("    test: ", nrow(test),  "filas x", ncol(test),  "columnas\n")

# --- Limpiar entorno ----------------------------------------
rm(train_h, test_h, train_p, test_p, train_p_agg, test_p_agg,
   excluir_hogares, excluir_personas, vars_laborales,
   vars_factor, vars_winsorizar, agregar_personas, winsorizr)
gc()
