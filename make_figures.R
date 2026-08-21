# =============================================================================
# make_figures.R  —  Paper figures for the federated ICB-response study
#
# Reads the three pipeline outputs:
#   Comprehensive_Results.csv   (scalar metrics + CIs per fold)
#   Predictions_Detailed.csv    (per-sample y_true, y_pred_prob)
#   Coefficients_Detailed.csv   (per-gene logistic weights per fold)
#
# If a file is missing it generates schema-matching synthetic data, so the
# script runs standalone for a dry run. Figures are written to ./figures/.
#
# Deps: ggplot2, dplyr, tidyr, scales
#   install.packages(c("ggplot2","dplyr","tidyr","scales"))
# Run:  Rscript make_figures.R
# =============================================================================

suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(scales)})

# ----------------------------- setup -----------------------------------------
OUTDIR <- "figures"; dir.create(OUTDIR, showWarnings = FALSE)
TRAFFICKING <- c("CD274","CMTM6","CMTM4","DRG2","RAB11A","RAB8A","SNX1","SNX2","SNX27","VPS35")
IFN         <- c("IFNG","STAT1","CXCL9","CXCL10","IDO1","HLA-DRA")
COHORTS     <- c("Hugo","Riaz","Gide","Liu","IMvigor210")
FEATSETS    <- c("Trafficking_Only","IFN_Gamma_Only","Combined")
PAL_MODEL   <- c(Within_Cohort_Local_Baseline="#9e9e9e", FL_FedAvg="#4C9F70",
                 FL_FedProx="#2C6E9B", Centralized_Reference="#B5651D")

theme_pub <- function(base = 12) {
  theme_bw(base_size = base) +
    theme(panel.grid.minor = element_blank(),
          strip.background = element_rect(fill = "grey92", colour = NA),
          strip.text = element_text(face = "bold"),
          legend.position = "bottom",
          plot.title = element_text(face = "bold"))
}
save_fig <- function(p, name, w = 8, h = 5.5) {
  ggsave(file.path(OUTDIR, paste0(name, ".png")), p, width = w, height = h, dpi = 300)
  ggsave(file.path(OUTDIR, paste0(name, ".pdf")), p, width = w, height = h)
  message("  wrote ", name)
}
parse_eps <- function(x) {
  x <- as.character(x)
  ifelse(tolower(x) %in% c("inf","infinity"), Inf, suppressWarnings(as.numeric(x)))
}

# --------------------- load (or synthesise) inputs ---------------------------
synth_results <- function() {
  set.seed(1); rows <- list()
  fs_bump <- c(Trafficking_Only = -0.03, IFN_Gamma_Only = 0.0, Combined = 0.04)
  for (fs in FEATSETS) for (h in COHORTS) {
    base <- 0.70 + fs_bump[fs] + rnorm(1, 0, 0.03)
    add <- function(model, eps, auroc) {
      lo <- pmax(0, auroc - runif(1, .05, .1)); hi <- pmin(1, auroc + runif(1, .05, .1))
      rows[[length(rows)+1]] <<- data.frame(
        Feature_Set = fs, Holdout = h, Model = model, Cumulative_Epsilon = eps,
        Prevalence = runif(1,.25,.4), AUROC = auroc, AUROC_CI_Low = lo, AUROC_CI_High = hi,
        AUPRC = auroc - .05, AUPRC_CI_Low = lo-.05, AUPRC_CI_High = hi-.05,
        Brier_Score = 0.25 - (auroc-.5)*0.2 + rnorm(1,0,.01),
        Log_Loss = 0.6, Calibration_Slope = rnorm(1,1,.2), Calibration_Intercept = rnorm(1,0,.1),
        Sensitivity = auroc-.05, Specificity = auroc-.03, F1 = auroc-.08, MCC = (auroc-.5)*1.2,
        Delta_AUROC = NA, Delta_AUROC_CI_Low = NA, Delta_AUROC_CI_High = NA,
        Delta_AUPRC = NA, Delta_AUPRC_CI_Low = NA, Delta_AUPRC_CI_High = NA)
    }
    add("Centralized_Reference", Inf, base)
    add("Within_Cohort_Local_Baseline", Inf, base - runif(1,.05,.15))
    for (strat in c("FL_FedAvg","FL_FedProx")) {
      prox <- if (strat == "FL_FedProx") 0.01 else 0
      for (eps in c(Inf,10,8,4,2,1)) {
        pen <- if (is.infinite(eps)) 0 else (0.18 / eps)          # DP cost grows as eps shrinks
        a <- max(0.5, base - 0.02 + prox - pen + rnorm(1,0,.01))
        r <- nrow(do.call(rbind, rows)) # placeholder
        add(strat, eps, a)
        d <- a - base
        rows[[length(rows)]]$Delta_AUROC <- d
        rows[[length(rows)]]$Delta_AUROC_CI_Low <- d - runif(1,.03,.06)
        rows[[length(rows)]]$Delta_AUROC_CI_High <- d + runif(1,.03,.06)
        rows[[length(rows)]]$Delta_AUPRC <- d - .01
        rows[[length(rows)]]$Delta_AUPRC_CI_Low <- d - .06
        rows[[length(rows)]]$Delta_AUPRC_CI_High <- d + .04
      }
    }
  }
  do.call(rbind, rows)
}
synth_predictions <- function() {
  set.seed(2); out <- list()
  for (h in COHORTS) for (model in c("Centralized_Reference","FL_FedProx"))
    for (eps in c(Inf, 4)) {
      if (model == "Centralized_Reference" && is.finite(eps)) next
      n <- sample(40:140, 1); y <- rbinom(n, 1, 0.33)
      signal <- if (is.infinite(eps)) 1.4 else 0.7
      p <- plogis(rnorm(n, (y - 0.5) * signal, 1))
      out[[length(out)+1]] <- data.frame(Feature_Set="Combined", Holdout=h, Model=model,
                                         Cumulative_Epsilon=eps, y_true=y, y_pred_prob=p, y_pred_recal=pmin(pmax(p*0.9+0.03*mean(y),0),1))
    }
  do.call(rbind, out)
}
synth_coeffs <- function() {
  set.seed(3); out <- list()
  for (h in COHORTS) for (model in c("Centralized_Reference","FL_FedProx")) {
    eps <- if (model=="Centralized_Reference") Inf else Inf
    for (g in c(TRAFFICKING, IFN)) {
      mu <- if (g %in% IFN) 0.5 else if (g %in% c("CD274","CMTM6")) 0.35 else 0.05
      out[[length(out)+1]] <- data.frame(Feature_Set="Combined", Holdout=h, Model=model,
                                         Gene=g, Weight=rnorm(1, mu, 0.15), Cumulative_Epsilon=eps, Seed=sample(0:4,1))
    }
  }
  do.call(rbind, out)
}
load_or <- function(path, synth) {
  if (file.exists(path)) { message("loading ", path); read.csv(path, stringsAsFactors = FALSE) }
  else { message("** ", path, " not found - using synthetic stand-in **"); synth() }
}

results <- load_or("Comprehensive_Results.csv", synth_results)
preds   <- load_or("Predictions_Detailed.csv",  synth_predictions)
coeffs  <- load_or("Coefficients_Detailed.csv", synth_coeffs)
results$Cumulative_Epsilon <- parse_eps(results$Cumulative_Epsilon)
preds$Cumulative_Epsilon   <- parse_eps(preds$Cumulative_Epsilon)
coeffs$Cumulative_Epsilon  <- parse_eps(coeffs$Cumulative_Epsilon)
results$Feature_Set <- factor(results$Feature_Set, levels = FEATSETS)

# ============================================================================
# FIG 1 — Privacy-utility curve (AUROC vs cumulative epsilon)
# ============================================================================
fl <- results %>% filter(grepl("^FL_", Model), is.finite(Cumulative_Epsilon))
agg <- fl %>% group_by(Feature_Set, Model, Cumulative_Epsilon) %>%
  summarise(m = mean(AUROC, na.rm=TRUE), s = sd(AUROC, na.rm=TRUE), .groups="drop") %>%
  mutate(s = ifelse(is.na(s), 0, s))
ceil <- results %>% filter(Model == "Centralized_Reference") %>%
  group_by(Feature_Set) %>% summarise(m = mean(AUROC, na.rm=TRUE), .groups="drop")
p1 <- ggplot(agg, aes(Cumulative_Epsilon, m, colour = Model, fill = Model)) +
  geom_ribbon(aes(ymin = m - s, ymax = m + s), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_hline(data = ceil, aes(yintercept = m), linetype = "dashed", colour = "grey30") +
  facet_wrap(~ Feature_Set) +
  scale_x_log10(breaks = c(1,2,4,8,10)) +
  scale_colour_manual(values = PAL_MODEL) + scale_fill_manual(values = PAL_MODEL) +
  labs(title = "Privacy-utility tradeoff",
       subtitle = "Dashed line = non-private centralized ceiling; smaller epsilon = stronger privacy",
       x = expression("Cumulative privacy budget " * epsilon * " (log scale)"),
       y = "AUROC (mean +/- SD across cohorts)") +
  theme_pub()
save_fig(p1, "fig1_privacy_utility_auroc", w = 9)

# Brier version (calibration cost of privacy)
aggb <- fl %>% group_by(Feature_Set, Model, Cumulative_Epsilon) %>%
  summarise(m = mean(Brier_Score, na.rm=TRUE), s = sd(Brier_Score, na.rm=TRUE), .groups="drop") %>%
  mutate(s = ifelse(is.na(s), 0, s))
p1b <- ggplot(aggb, aes(Cumulative_Epsilon, m, colour = Model, fill = Model)) +
  geom_ribbon(aes(ymin = m - s, ymax = m + s), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  facet_wrap(~ Feature_Set) + scale_x_log10(breaks = c(1,2,4,8,10)) +
  scale_colour_manual(values = PAL_MODEL) + scale_fill_manual(values = PAL_MODEL) +
  labs(title = "Privacy cost to calibration",
       x = expression("Cumulative " * epsilon * " (log scale)"),
       y = "Brier score (lower = better)") +
  theme_pub()
save_fig(p1b, "fig1b_privacy_utility_brier", w = 9)

# ============================================================================
# FIG 2 — Federation vs baselines (non-private): floor / federation / ceiling
# ============================================================================
cmp <- results %>%
  filter(Model %in% c("Within_Cohort_Local_Baseline","Centralized_Reference") |
         (Model == "FL_FedProx" & is.infinite(Cumulative_Epsilon))) %>%
  mutate(Model = factor(Model, levels = c("Within_Cohort_Local_Baseline","FL_FedProx","Centralized_Reference")))
dodge <- position_dodge(width = 0.6)
p2 <- ggplot(cmp, aes(Holdout, AUROC, colour = Model)) +
  geom_point(position = dodge, size = 2) +
  geom_errorbar(aes(ymin = AUROC_CI_Low, ymax = AUROC_CI_High), position = dodge, width = 0.4) +
  facet_wrap(~ Feature_Set) + coord_flip() +
  scale_colour_manual(values = PAL_MODEL) +
  labs(title = "Local control vs federated (non-private) vs centralized ceiling",
       y = "AUROC (95% bootstrap CI)", x = NULL) +
  theme_pub()
save_fig(p2, "fig2_federation_vs_baselines", w = 9, h = 6)

# ============================================================================
# FIG 3 — Delta-AUROC forest (federated - centralized), non-private
# ============================================================================
dl <- results %>% filter(Model %in% c("FL_FedAvg","FL_FedProx"),
                         is.infinite(Cumulative_Epsilon), !is.na(Delta_AUROC))
p3 <- ggplot(dl, aes(Delta_AUROC, Holdout, colour = Model)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  geom_point(position = dodge, size = 2) +
  geom_errorbarh(aes(xmin = Delta_AUROC_CI_Low, xmax = Delta_AUROC_CI_High),
                 position = dodge, height = 0.3) +
  facet_wrap(~ Feature_Set) +
  scale_colour_manual(values = PAL_MODEL) +
  labs(title = "Cost of federation: change in AUROC (federated - centralized)",
       subtitle = "Left of 0 = federation underperforms the pooled ceiling",
       x = expression(Delta * "AUROC"), y = NULL) +
  theme_pub()
save_fig(p3, "fig3_delta_forest", w = 9, h = 5)

# ============================================================================
# FIG 4 — Biology ablation: AUROC by feature set (centralized ceiling)
# ============================================================================
bio <- results %>% filter(Model == "Centralized_Reference")
p4 <- ggplot(bio, aes(Feature_Set, AUROC, colour = Feature_Set)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = AUROC_CI_Low, ymax = AUROC_CI_High), width = 0.3) +
  facet_wrap(~ Holdout, nrow = 1) +
  labs(title = "Biology ablation: does the trafficking module add signal?",
       x = NULL, y = "AUROC (95% CI)") +
  theme_pub() + theme(axis.text.x = element_text(angle = 35, hjust = 1),
                      legend.position = "none")
save_fig(p4, "fig4_biology_ablation", w = 11, h = 4.5)

# ============================================================================
# FIG 5 — Coefficient plot: gene weights coloured by module (the moat)
# ============================================================================
cf <- coeffs %>% filter(Feature_Set == "Combined", Model == "Centralized_Reference") %>%
  group_by(Gene) %>% summarise(m = mean(Weight), s = sd(Weight), .groups="drop") %>%
  mutate(s = ifelse(is.na(s), 0, s),
         Module = ifelse(Gene %in% IFN, "IFN-g", "Trafficking"))
p5 <- ggplot(cf, aes(m, reorder(Gene, m), colour = Module)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  geom_pointrange(aes(xmin = m - s, xmax = m + s)) +
  scale_colour_manual(values = c("Trafficking" = "#2C6E9B", "IFN-g" = "#B5651D")) +
  labs(title = "Standardized logistic weights (mean +/- SD across folds)",
       subtitle = "Combined model: the coefficients ARE the interpretability layer",
       x = "Weight", y = NULL) +
  theme_pub()
save_fig(p5, "fig5_coefficients", w = 8.5, h = 6)

# ============================================================================
# FIG 6 — ROC curves per cohort (centralized vs federated, non-private)
# ============================================================================
roc_points <- function(y, p) {
  o <- order(p, decreasing = TRUE); y <- y[o]
  P <- sum(y); N <- sum(1 - y)
  if (P == 0 || N == 0) return(NULL)
  data.frame(fpr = c(0, cumsum(1 - y) / N), tpr = c(0, cumsum(y) / P))
}
roc_all <- preds %>% filter(Feature_Set == "Combined", is.infinite(Cumulative_Epsilon)) %>%
  group_by(Holdout, Model) %>% group_modify(~ {
    r <- roc_points(.x$y_true, .x$y_pred_prob); if (is.null(r)) tibble() else r
  }) %>% ungroup()
p6 <- ggplot(roc_all, aes(fpr, tpr, colour = Model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ Holdout, nrow = 1) + coord_equal() +
  scale_colour_manual(values = PAL_MODEL) +
  labs(title = "ROC by held-out cohort (non-private)", x = "FPR", y = "TPR") +
  theme_pub()
save_fig(p6, "fig6_roc_by_cohort", w = 12, h = 3.6)

# ============================================================================
# FIG 7 — Reliability (calibration) curves: non-private vs private
# ============================================================================
calib_bins <- function(y, p, nb = 8) {
  br <- seq(0, 1, length.out = nb + 1); b <- cut(p, br, include.lowest = TRUE)
  data.frame(y = y, p = p, b = b) %>% group_by(b) %>%
    summarise(pred = mean(p), obs = mean(y), n = n(), .groups = "drop")
}
cal <- preds %>% filter(Feature_Set == "Combined",
                        Model %in% c("Centralized_Reference","FL_FedProx")) %>%
  mutate(eps_lab = ifelse(is.infinite(Cumulative_Epsilon), "non-private",
                          paste0("eps=", Cumulative_Epsilon))) %>%
  group_by(Model, eps_lab) %>% group_modify(~ calib_bins(.x$y_true, .x$y_pred_prob)) %>% ungroup()
p7 <- ggplot(cal, aes(pred, obs, colour = interaction(Model, eps_lab, sep=" / "))) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line() + geom_point(aes(size = n)) +
  coord_equal(xlim = c(0,1), ylim = c(0,1)) +
  labs(title = "Reliability curves", x = "Mean predicted probability",
       y = "Observed frequency", colour = NULL, size = "n") +
  theme_pub()
save_fig(p7, "fig7_calibration", w = 7.5, h = 6)

# ============================================================================
# FIG 8 — Summary heatmap: AUROC over feature set x epsilon (FedProx)
# ============================================================================
hm <- results %>% filter(Model == "FL_FedProx", is.finite(Cumulative_Epsilon)) %>%
  group_by(Feature_Set, Cumulative_Epsilon) %>%
  summarise(AUROC = mean(AUROC, na.rm=TRUE), .groups="drop")
p8 <- ggplot(hm, aes(factor(Cumulative_Epsilon), Feature_Set, fill = AUROC)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.2f", AUROC)), size = 3.5) +
  scale_fill_gradient(low = "#f7fbff", high = "#08519c") +
  labs(title = "FedProx AUROC across feature set x privacy budget",
       x = expression("Cumulative " * epsilon), y = NULL) +
  theme_pub()
save_fig(p8, "fig8_summary_heatmap", w = 7, h = 4)


# ============================================================================
# FIG 9 - PCA of pooled expression: cohort (non-IID) heterogeneity
# ============================================================================
if (file.exists("Expression_Pooled.csv")) {
  ex <- read.csv("Expression_Pooled.csv", stringsAsFactors = FALSE)
} else {
  message("** Expression_Pooled.csv not found - synthetic stand-in **")
  set.seed(9); genes_all <- c(TRAFFICKING, IFN)
  ex <- do.call(rbind, lapply(COHORTS, function(cc) {
    n <- sample(40:120, 1)
    shift <- rnorm(length(genes_all), 0, 0.7)   # per-cohort batch offset
    M <- matrix(rnorm(n * length(genes_all)), n) + matrix(shift, n, length(genes_all), byrow = TRUE)
    d <- as.data.frame(M); names(d) <- genes_all
    d$Cohort <- cc; d$response_binary <- rbinom(n, 1, 0.35); d
  }))
}
genes <- intersect(c(TRAFFICKING, IFN), names(ex))
pca <- prcomp(ex[, genes], scale. = TRUE)
ve <- round(100 * pca$sdev^2 / sum(pca$sdev^2), 1)
pcs <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], Cohort = ex$Cohort)
p9 <- ggplot(pcs, aes(PC1, PC2, colour = Cohort)) +
  geom_point(alpha = 0.7, size = 1.8) + stat_ellipse(level = 0.68) +
  labs(title = "Cohort heterogeneity (non-IID structure)",
       subtitle = "PCA of standardized expression; cross-cohort separation motivates federated, batch-robust modeling",
       x = paste0("PC1 (", ve[1], "%)"), y = paste0("PC2 (", ve[2], "%)")) +
  theme_pub()
save_fig(p9, "fig9_pca_heterogeneity", w = 7.5, h = 5.5)

# ============================================================================
# FIG 10 - calibration before vs after Platt recalibration (non-private FL)
# ============================================================================
rc <- preds %>% filter(Feature_Set == "Combined", Model == "FL_FedProx", is.infinite(Cumulative_Epsilon))
if (!("y_pred_recal" %in% names(rc))) rc$y_pred_recal <- rc$y_pred_prob
calib_bins2 <- function(y, p, nb = 8) {
  br <- seq(0, 1, length.out = nb + 1); b <- cut(p, br, include.lowest = TRUE)
  data.frame(y = y, p = p, b = b) %>% group_by(b) %>%
    summarise(pred = mean(p), obs = mean(y), .groups = "drop")
}
cc <- bind_rows(calib_bins2(rc$y_true, rc$y_pred_prob)  %>% mutate(stage = "Raw"),
                calib_bins2(rc$y_true, rc$y_pred_recal) %>% mutate(stage = "Recalibrated"))
p10 <- ggplot(cc, aes(pred, obs, colour = stage)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line() + geom_point(size = 2) + coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_colour_manual(values = c("Raw" = "#B5651D", "Recalibrated" = "#2C6E9B")) +
  labs(title = "Calibration before vs after recalibration",
       subtitle = "Platt scaling fit on training cohorts, applied to held-out cohort",
       x = "Mean predicted probability", y = "Observed frequency", colour = NULL) +
  theme_pub()
save_fig(p10, "fig10_recalibration", w = 6.5, h = 6)


# ============================================================================
# FIG 11 - coefficient stability across seeds, folds, and privacy levels
# ============================================================================
cs <- coeffs %>% filter(Feature_Set == "Combined", Model == "FL_FedProx")
if (!("Seed" %in% names(cs))) cs$Seed <- 0
stab <- cs %>% group_by(Gene) %>%
  summarise(med = median(Weight),
            lo = quantile(Weight, 0.25), hi = quantile(Weight, 0.75),
            sign_consistency = max(mean(Weight > 0), mean(Weight < 0)),  # 0.5=random, 1=always same sign
            .groups = "drop") %>%
  mutate(Module = ifelse(Gene %in% IFN, "IFN-g", "Trafficking"))
p11 <- ggplot(stab, aes(med, reorder(Gene, med), colour = Module)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  geom_pointrange(aes(xmin = lo, xmax = hi)) +
  geom_text(aes(label = sprintf("%.0f%%", 100 * sign_consistency)),
            hjust = -0.35, size = 2.6, show.legend = FALSE) +
  scale_colour_manual(values = c("Trafficking" = "#2C6E9B", "IFN-g" = "#B5651D")) +
  labs(title = "Coefficient stability (FL, combined model)",
       subtitle = "Median +/- IQR across seeds x folds x privacy levels; % = sign consistency",
       x = "Weight (median, IQR)", y = NULL) +
  theme_pub()
save_fig(p11, "fig11_coefficient_stability", w = 8, h = 6)

message("\nDone. ", length(list.files(OUTDIR, pattern = "png$")), " PNG figures in ./", OUTDIR, "/")
