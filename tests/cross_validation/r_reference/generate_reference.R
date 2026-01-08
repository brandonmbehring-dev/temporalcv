#!/usr/bin/env Rscript
# =============================================================================
# Generate Reference Values for Cross-Validation Tests
# =============================================================================
#
# This script generates baseline values from R's forecast package for
# cross-validation against temporalcv's Python implementation.
#
# Requirements:
#   install.packages("forecast")
#
# Usage:
#   Rscript generate_reference.R
#
# Output:
#   - dm_reference.csv: Diebold-Mariano test reference values
#
# Note:
# The generated CSVs should be committed to the repository.
# Python tests read these pre-computed values (no R required in CI).
# =============================================================================

library(forecast)

set.seed(42)

# =============================================================================
# DM Test Reference Values
# =============================================================================

# Generate reproducible test cases
generate_dm_reference <- function() {
  results <- data.frame()

  # Test Case 1: IID errors, equal performance (null true)
  n <- 100
  e1 <- rnorm(n, 0, 1)
  e2 <- rnorm(n, 0, 1)

  dm <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 2)
  results <- rbind(results, data.frame(
    case = "iid_equal",
    n = n,
    h = 1,
    alternative = "two.sided",
    statistic = dm$statistic,
    pvalue = dm$p.value,
    seed = 42
  ))

  # Test Case 2: Model 1 clearly better
  set.seed(43)
  e1 <- rnorm(n, 0, 0.5)  # Smaller errors
  e2 <- rnorm(n, 0, 1.5)  # Larger errors

  dm <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 2)
  results <- rbind(results, data.frame(
    case = "model1_better",
    n = n,
    h = 1,
    alternative = "two.sided",
    statistic = dm$statistic,
    pvalue = dm$p.value,
    seed = 43
  ))

  # Test Case 3: Multi-step horizon (h=4)
  set.seed(44)
  e1 <- rnorm(n, 0, 1)
  e2 <- rnorm(n, 0.3, 1)  # Biased

  dm <- dm.test(e1, e2, alternative = "two.sided", h = 4, power = 2)
  results <- rbind(results, data.frame(
    case = "h4_horizon",
    n = n,
    h = 4,
    alternative = "two.sided",
    statistic = dm$statistic,
    pvalue = dm$p.value,
    seed = 44
  ))

  # Test Case 4: One-sided test
  set.seed(45)
  e1 <- rnorm(n, 0, 0.8)
  e2 <- rnorm(n, 0, 1.2)

  dm <- dm.test(e1, e2, alternative = "greater", h = 1, power = 2)
  results <- rbind(results, data.frame(
    case = "one_sided_greater",
    n = n,
    h = 1,
    alternative = "greater",
    statistic = dm$statistic,
    pvalue = dm$p.value,
    seed = 45
  ))

  # Test Case 5: Small sample (n=30)
  set.seed(46)
  n_small <- 30
  e1 <- rnorm(n_small, 0, 1)
  e2 <- rnorm(n_small, 0, 1.5)

  dm <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 2)
  results <- rbind(results, data.frame(
    case = "small_sample",
    n = n_small,
    h = 1,
    alternative = "two.sided",
    statistic = dm$statistic,
    pvalue = dm$p.value,
    seed = 46
  ))

  # Test Case 6: MAE (power=1)
  set.seed(47)
  e1 <- rnorm(n, 0, 1)
  e2 <- rnorm(n, 0, 1.3)

  dm <- dm.test(e1, e2, alternative = "two.sided", h = 1, power = 1)
  results <- rbind(results, data.frame(
    case = "mae_power1",
    n = n,
    h = 1,
    alternative = "two.sided",
    statistic = dm$statistic,
    pvalue = dm$p.value,
    seed = 47
  ))

  return(results)
}

# Generate and save
dm_ref <- generate_dm_reference()
write.csv(dm_ref, "dm_reference.csv", row.names = FALSE)

cat("Generated dm_reference.csv with", nrow(dm_ref), "test cases\n")
cat("\nPreview:\n")
print(dm_ref)
