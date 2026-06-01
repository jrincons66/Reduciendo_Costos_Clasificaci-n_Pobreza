optimizar_threshold <- function(modelo, train_data, y_real) {
  
  if (inherits(modelo, "ranger")) {
    probs <- modelo$predictions[, "pobre"]
  } else {
    probs <- predict(modelo, newdata = train_data, type = "prob")[, "pobre"]
  }
  
  if (is.factor(y_real)) {
    y_bin <- as.integer(y_real == "pobre")
  } else {
    y_bin <- as.integer(y_real)
  }
  
  thresh_grid <- seq(0.1, 0.9, by = 0.005)
  f1_grid <- map_dbl(thresh_grid, function(t) {
    preds <- as.integer(probs >= t)
    tp    <- sum(preds == 1 & y_bin == 1)
    fp    <- sum(preds == 1 & y_bin == 0)
    fn    <- sum(preds == 0 & y_bin == 1)
    prec  <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rec   <- if (tp + fn == 0) 0 else tp / (tp + fn)
    if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
  })
  
  list(
    threshold = thresh_grid[which.max(f1_grid)],
    f1        = max(f1_grid)
  )
}
