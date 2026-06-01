# ============================================================
# 00_features.R
# Feature engineering — 18 variables proxy de pobreza
# ============================================================
#
# Tópicos de IA · Universidad de los Andes · 2026-10
# Profesor: Álvaro Riascos
# Autores: Jose Rincón · Lucas Rodríguez · María Paula Osuna
#
# Descripción:
#   Construye 18 variables proxy que capturan dimensiones
#   de pobreza sin usar ingresos directamente. Todas son
#   observables y difíciles de manipular por el hogar.
#
# Referencias:
#   - IDB Kaggle Costa Rica (2018): hacinamiento, dependencia
#   - Corral et al. (2024): mercado laboral Colombia
#   - Banerjee & Duflo (2011): patrones hogares pobres
#   - World Bank Poverty & Shared Prosperity (2019)
#   - UNDP & ECLAC Social Panorama LA (2024)
# ============================================================

train <- readRDS(here(paths$processed, "train_clean.rds"))
test  <- readRDS(here(paths$processed, "test_clean.rds"))

feature_engineer <- function(df) {
  df |>
    mutate(
      
      # [01] Ratio de dependencia
      # Fracción del hogar que no genera ingreso laboral.
      # A mayor ratio, mayor presión sobre los ocupados.
      ratio_dependencia = (nper - n_ocupados) / pmax(nper, 1),
      
      # [02] Hacinamiento
      # Personas por cuarto. Umbral crítico >= 3 pers/cuarto.
      hacinamiento = nper / pmax(p5000, 1),
      
      # [03] Interacción educación × ocupación
      # Capital humano solo genera valor si hay empleo.
      educ_x_ocup = as.integer(nivel_educ_max) * tasa_ocupacion,
      
      # [04] Interacción zona rural × ocupación
      # En zonas rurales el empleo tiene menor productividad.
      rural_x_ocup = as.integer(clase == "2") * tasa_ocupacion,
      
      # [05] Formalidad × seguridad social
      # Proxy de inserción real en mercado formal.
      formal_x_salud = prop_cotiza_pension * prop_afiliado_salud,
      
      # [06] Tamaño del hogar al cuadrado
      # Rendimientos decrecientes del tamaño del hogar.
      nper_sq = nper^2,
      
      # [07] Edad promedio al cuadrado
      # Relación en U invertida entre edad y pobreza.
      edad_prom_sq = edad_promedio^2,
      
      # [08] Género × inactividad
      # Hogares con más mujeres y baja ocupación.
      mujeres_x_inact = prop_mujeres * (1 - tasa_ocupacion),
      
      # [09] Jefatura femenina × inactividad
      # Jefatura femenina + baja inserción laboral.
      jefe_mujer_inact = jefe_mujer * (1 - tasa_ocupacion),
      
      # [10] Tasa de inactividad
      # Fracción de la PET que no busca ni tiene empleo.
      tasa_inactivos = n_inactivos / pmax(n_pet, 1),
      
      # [11] Sin ocupados (dummy)
      # Hogar con cero ocupados: umbral duro de vulnerabilidad.
      sin_ocupados = as.integer(n_ocupados == 0),
      
      # [12] Educación jefe × ocupación jefe
      # Capital humano del jefe solo activa si trabaja.
      educ_jefe_x_ocup = as.integer(educ_jefe) * ocup_jefe,
      
      # [13] Calidad del empleo
      # Horas trabajadas × formalidad: proxy de ingreso laboral.
      calidad_empleo = horas_trabajo_prom * prop_cotiza_pension,
      
      # [14] Presión habitacional
      # Cuartos usados sobre cuartos disponibles.
      presion_habitacional = p5010 / pmax(p5000, 1),
      
      # [15] Jefe vulnerable
      # Jefe sin empleo y sin educación más allá de primaria.
      jefe_vulnerable = as.integer(ocup_jefe == 0 & educ_jefe <= 3),
      
      # [16] Doble protección social
      # Alta cotización pensional: ingreso estable.
      doble_proteccion = prop_cotiza_pension * prop_cotiza_pension,
      
      # [17] Ratio adultos mayores sobre PET
      # Carga específica de vejez dentro de la población activa.
      ratio_mayores_65 = n_mayores_65 / pmax(n_pet, 1),
      
      # [18] Jefe mayor inactivo
      # Jefe mayor de 60 años sin empleo.
      jefe_mayor_inactivo = as.integer(edad_jefe > 60 & ocup_jefe == 0)
    )
}

cat(">>> Aplicando feature engineering...\n")
train <- feature_engineer(train)
test  <- feature_engineer(test)

# --- Guardar ------------------------------------------------
saveRDS(train, here(paths$processed, "train_features.rds"))
saveRDS(test,  here(paths$processed, "test_features.rds"))

cat(">>> 00_features.R completado\n")
cat("    train:", nrow(train), "filas x", ncol(train), "columnas\n")
cat("    test: ", nrow(test),  "filas x", ncol(test),  "columnas\n")
cat("\n    Features generadas (18):\n")
cat("    [01] ratio_dependencia      [10] tasa_inactivos\n")
cat("    [02] hacinamiento           [11] sin_ocupados\n")
cat("    [03] educ_x_ocup            [12] educ_jefe_x_ocup\n")
cat("    [04] rural_x_ocup           [13] calidad_empleo\n")
cat("    [05] formal_x_salud         [14] presion_habitacional\n")
cat("    [06] nper_sq                [15] jefe_vulnerable\n")
cat("    [07] edad_prom_sq           [16] doble_proteccion\n")
cat("    [08] mujeres_x_inact        [17] ratio_mayores_65\n")
cat("    [09] jefe_mujer_inact       [18] jefe_mayor_inactivo\n")

# --- Limpiar entorno ----------------------------------------
rm(feature_engineer)
gc()