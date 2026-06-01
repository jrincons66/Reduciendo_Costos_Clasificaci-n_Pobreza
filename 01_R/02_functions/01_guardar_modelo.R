guardar_modelo <- function(modelo, nombre, tipo, dir_modelo,
                           threshold, f1) {
  saveRDS(modelo, file.path(dir_modelo, paste0(nombre, ".rds")))
  
  log_path <- here(paths$models, "log.csv")
  entrada  <- data.frame(
    tipo      = tipo,
    modelo    = nombre,
    cv_f1     = round(f1, 6),
    threshold = round(threshold, 4),
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    stringsAsFactors = FALSE
  )
  
  if (file.exists(log_path)) {
    log_actual <- read.csv(log_path, stringsAsFactors = FALSE)
    log_nuevo  <- rbind(log_actual, entrada)
  } else {
    log_nuevo <- entrada
  }
  
  write.csv(log_nuevo, log_path, row.names = FALSE)
  cat(sprintf("    Modelo guardado: %s | F1=%.4f | threshold=%.3f\n",
              nombre, f1, threshold))
}
