generar_submission <- function(modelo, test_data, threshold,
                               tipo, nombre = NULL) {
  if (inherits(modelo, "ranger")) {
    probs <- predict(modelo, data = test_data)$predictions[, "pobre"]
  } else {
    probs <- predict(modelo, newdata = test_data, type = "prob")[, "pobre"]
  }
  
  preds <- as.integer(probs >= threshold)
  sub   <- data.frame(id = test_data$id, pobre = preds)
  
  dir_sub <- here(paths$submissions, tipo)
  dir.create(dir_sub, recursive = TRUE, showWarnings = FALSE)
  
  nombre_archivo <- if (!is.null(nombre)) nombre else "submission"
  ruta <- file.path(dir_sub, paste0(nombre_archivo, ".csv"))
  
  write.csv(sub, ruta, row.names = FALSE)
  cat("    Submission guardada:", basename(ruta), "\n")
}
