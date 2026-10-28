# Normalization of the count matrix kept by scripts/qc.sh: genes are rows, samples columns.

PSEUDOCOUNT <- 1

# Median-of-ratios size factors: each sample's counts against the geometric mean over the samples, per gene.
size_factors <- function(counts) {
  logs <- log(counts)
  log_geomean <- rowMeans(logs)
  usable <- is.finite(log_geomean)
  apply(logs[usable, , drop = FALSE], 2, function(x) exp(median(x - log_geomean[usable])))
}

log_normalize <- function(counts, factors = size_factors(counts)) {
  log2(sweep(counts, 2, factors, "/") + PSEUDOCOUNT)
}
