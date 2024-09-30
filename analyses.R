#!/usr/bin/env Rscript

# load the necessary packages
source("packages.R")

# clear all variables
rm(list = ls())

set.seed(123)
theme_set(theme_light())
options(digits = 8)
options(dplyr.summarise.inform = TRUE)

# Load the YAML file
config <- yaml.load_file("CONSTANTS.yaml")
# Access the 'julia_home' value
julia_home <- config$julia_home

library(jglmm)
options(JULIA_HOME = julia_home)
# run these lines twice
julia_setup(JULIA_HOME = julia_home)
jglmm_setup()

z_score <- function(x) {
    return((x - mean(x)) / sd(x))
}

z_score_test <- function(x, sd) {
    return((x - mean(x)) / sd)
}

remove_outlier <- function(df, reading_measure) {
    reading_times <- as.numeric(df[[reading_measure]])
    z_score <- z_score(reading_times)
    abs_z_score <- abs(z_score)
    df$outlier <- abs_z_score > 3
    # print number of outliers / total number of reading times
    print(paste(sum(df$outlier), "/", length(df$outlier)))
    # remove outliers
    df <- df[df$outlier == FALSE, ]
    return(df)
}

model_cross_val <- function(form, df_in, predicted_var, mixed_effects, num_folds = 10, shuffle_folds = FALSE, remove_outliers = FALSE, log_transform = FALSE, is_linear = TRUE) {
    df <- df_in
    if (predicted_var %in% CONT_RESP_VARIABLES) {
        if (log_transform == TRUE) {
            # remove 0s
            df <- df[df[[predicted_var]] != 0, ]
            df[[predicted_var]] <- log(df[[predicted_var]])
            if (remove_outliers == TRUE) {
                df <- remove_outlier(df, predicted_var)
            }
        }
    } else {
        is_linear <- FALSE
    }

    folds <- cut(seq(1, nrow(df)), breaks = num_folds, labels = FALSE)
    if (shuffle_folds == TRUE) {
        folds <- sample(folds)
    }
    estimates <- c()
    for (i in 1:num_folds) {
        test_indices <- which(folds == i, arr.ind = TRUE)
        test_data <- df[test_indices, ]
        train_data <- df[-test_indices, ]
        # todo take sigma from train data 
        test_data <- preprocess(test_data, PREDICTORS_TO_NORMALIZE)    
        train_data <- preprocess(train_data, PREDICTORS_TO_NORMALIZE)
        if (mixed_effects) {
            if (is_linear) {
                model <- jglmm(as.formula(form), data = train_data)
            } else {
                model <- jglmm(as.formula(form), data = train_data, family = "binomial", link="logit")
            }
        } else {
            model <- lm(as.formula(form), data = train_data)
        }
        if (is_linear) {
            stdev <- sigma(model)
            lh <- dnorm(
                test_data[[predicted_var]],
                mean = predict(model, newdata = test_data, allow.new.levels = TRUE),
                sd = stdev,
                log = TRUE
            )
        }
        else {
            probs <- predict(model, newdata = test_data, type = "response", allow.new.levels = TRUE)
            lh = ((test_data[[predicted_var]] * log(probs)) +  ((1 - test_data[[predicted_var]]) * log(1 - probs)))
        }
        estimates <- c(estimates, lh)
    }
    return(estimates)
}

preprocess <- function(df, predictors_to_normalize, is_linear) {
    # first, copy df in order to not overwrite original
    df_copy <- df
    df_copy$subj_id <- as.factor(df_copy$subject_id)
    #  convert to log lex freq
    df_copy$log_lex_freq <- as.numeric(df_copy$zipf_freq)

    # normalize baseline predictors
    df_copy$log_lex_freq <- scale(df_copy$log_lex_freq)
    df_copy$word_length <- scale(df_copy$word_length_with_punct)

    # normalize surprisal/entropy predictors
    for (predictor in predictors_to_normalize) {
        df_copy[[predictor]] <- as.numeric(df_copy[[predictor]])
        df_copy[[predictor]] <- scale(df_copy[[predictor]])
    }
    return(df_copy)
}

is_significant <- function(p_value, alpha = 0.05) {
    ifelse(p_value < alpha, "sig.", "not sig.")
}

# Prepare experiments
# log-transform response variable?
LOG_TF <- TRUE
CONT_RESP_VARIABLES <- c("FFD", "SFD", "FD", "FPRT", "FRT", "TFT", "RRT", "word_rt", "RPD_inc")

decoding_df <- read.csv("data/rms_scores_surp_ent.csv", header = TRUE, sep = "\t")
colnames(decoding_df) <- gsub("\\.", "_", colnames(decoding_df))
decoding_df$model <- as.factor(decoding_df$model)
decoding_df$decoding_strategy <- as.factor(decoding_df$decoding_strategy)
reading_measures <- c("FPRT", "TFT", "RRT", "RPD_inc", "Fix")
# find all columns that include "surprisal" or "entropy"
PREDICTORS_TO_NORMALIZE <- grep("surprisal|entropy|entropies", colnames(decoding_df), value = TRUE)


# Create the main "results" directory if it doesn't already exist
if (!dir.exists("results")) {
  dir.create("results")
} else {
  cat("Directory 'results' already exists.\n")
}

# Create the subdirectories under "results" if they don't already exist
if (!dir.exists("results/baseline")) {
  dir.create("results/baseline")
} else {
  cat("Directory 'results/baseline' already exists.\n")
}

if (!dir.exists("results/e1")) {
  dir.create("results/e1")
} else {
  cat("Directory 'results/e1' already exists.\n")
}

if (!dir.exists("results/e2")) {
  dir.create("results/e2")
} else {
  cat("Directory 'results/e2' already exists.\n")
}

if (!dir.exists("results/e3")) {
  dir.create("results/e3")
} else {
  cat("Directory 'results/e3' already exists.\n")
}




##############
###Baseline ###
##############


get_dlls_baseline_exp <- function(model_df, baseline_predictors, reading_measures) {
    all_df <- data.frame()
    for (reading_measure in reading_measures) {
        print(reading_measure)
        for (baseline_predictor in baseline_predictors) {
            surp <- baseline_predictor[1]
            ent <- baseline_predictor[2]
            formulas <- c(
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq"),
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", surp),
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", ent),
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", surp, " + ", ent)
            )
            print("   baseline")
            baseline_dll <- model_cross_val(formulas[1], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            print("   target_surp")
            target_surp_dll <- model_cross_val(formulas[2], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            print("   target_ent")

            target_ent_dll <- model_cross_val(formulas[3], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            print("   target_surp_ent")
            target_surp_ent_dll <- model_cross_val(formulas[4], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)

            loglik_df_surp <- data.frame(
                delta_basline = baseline_dll,
                delta_target = target_surp_dll,
                delta_loglik = target_surp_dll - baseline_dll,
                target_predictor = surp,
                reading_measure = reading_measure
            )
            loglik_df_ent <- data.frame(
                delta_basline = baseline_dll,
                delta_target = target_ent_dll,
                delta_loglik = target_ent_dll - baseline_dll,
                target_predictor = ent,
                reading_measure = reading_measure
            )
            loglik_df_surp_ent <- data.frame(
                delta_basline = baseline_dll,
                delta_target = target_surp_ent_dll,
                delta_loglik = target_surp_ent_dll - baseline_dll,
                target_predictor = paste(surp, ent, sep = " + "),
                reading_measure = reading_measure
            )
            all_df <- rbind(all_df, loglik_df_surp, loglik_df_ent, loglik_df_surp_ent)
        }
    }
    return(all_df)
}

baseline_predictor_combinations <- list(
    c("surprisal_mistral_base", "entropy_mistral_base"),
    c("surprisal_p_mistral_base", "entropy_p_mistral_base"),
    c("surprisal_mistral_instruct", "entropy_mistral_instruct"),
    c("surprisal_p_mistral_instruct", "entropy_p_mistral_instruct"),
    c("surprisal_phi2", "entropy_phi2"),
    c("surprisal_p_phi2", "entropy_p_phi2"),
    c("surprisal_gpt2", "entropy_gpt2"),
    c("surprisal_p_gpt2", "entropy_p_gpt2"),
    c("surprisal_wizardlm", "entropy_wizardlm"),
    c("surprisal_p_wizardlm", "entropy_p_wizardlm")
)

all_dlls_baseline <- get_dlls_baseline_exp(decoding_df, baseline_predictor_combinations, reading_measures)

write.csv(all_dlls_baseline, "results/baseline/dll_baseline.csv", row.names = FALSE)


# do a paired permutation test to test whether the delta loglik is significantly different from zero
permt_baseline <- all_dlls_baseline %>%
    group_by(target_predictor, reading_measure) %>%
    do(tidy((paired.perm.test(.$delta_loglik, n.perm = 500, pval = TRUE))))
# rename the column name to p-value
colnames(permt_baseline)[3] <- 'p.value'

# get mean and sd of delta loglik
dll_baseline_summarized <- all_dlls_baseline %>%
    group_by(target_predictor, reading_measure) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()

# add p value from permutation test to dll_baseline_summarized
dll_baseline_summarized <- merge(dll_baseline_summarized, permt_baseline, by = c("target_predictor", "reading_measure"), all = TRUE)

# add column that indicates significance
dll_baseline_summarized$significance <- is_significant(dll_baseline_summarized$p.value)

# column to indicate whether prompt included or not
dll_baseline_summarized$prompt <- ifelse(grepl("_p_", dll_baseline_summarized$target_predictor), "w/prompt", "wo/prompt")

# add new column 
dll_baseline_summarized <- dll_baseline_summarized %>%
  mutate(predictability_metric = case_when(
    grepl("entropy", target_predictor) & grepl("surprisal", target_predictor) ~ "combined",
    grepl("entropy", target_predictor) & !grepl("surprisal", target_predictor) ~ "entropy",
    grepl("surprisal", target_predictor) & !grepl("entropy", target_predictor) ~ "surprisal",
    TRUE ~ NA_character_  # default case if none of the conditions match
  ))

# write to csv
write.csv(dll_baseline_summarized, "results/baseline/dll_baseline_summarized.csv", row.names = FALSE)

# # plot each reading measure separately, only those without prompt
# for (reading_measure in unique(dll_baseline_summarized$reading_measure)) {
#     ggplot(data = dll_baseline_summarized[dll_baseline_summarized$reading_measure == reading_measure & dll_baseline_summarized$prompt == "wo/prompt", ], aes(x = target_predictor, y = m, colour = target_predictor, shape=significance)) +
#         geom_point(
#             position = position_dodge(width = .5), size = 3
#         ) +
#         geom_errorbar(aes(ymin = lower, ymax = upper),
#             width = .1, position = position_dodge(width = .5), linewidth = 0.4
#         ) +
#         # scale_x_discrete(guide = guide_axis(angle = 45)) +
#         scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
#         scale_shape_manual(values = c(1, 19)) +
#         scale_linetype_manual(values=c("solid", "dotted")) +
#         geom_hline(yintercept = 0, linetype = "dashed") +
#         ylab("Delta PP") +
#         xlab("Model") + 
#         facet_grid(~predictability_metric) +
#         theme(text = element_text(family = "sans")) +
#         theme(legend.position = "bottom", axis.ticks.x = element_blank(), axis.text.x = element_blank())
#     ggsave(paste0("results/baseline/dll_baseline_", reading_measure, ".pdf"), width = 12, height = 10, dpi = 200)
# }
# For plotting: create a new column for custom legend labels based on the predictability_metric
dll_baseline_summarized_labels <- dll_baseline_summarized %>%
  mutate(legend_label = case_when(
    predictability_metric == "entropy" ~ sub("^[^_]+_", "", target_predictor),
    predictability_metric == "surprisal" ~ sub("^[^_]+_", "", target_predictor),
    predictability_metric == "combined" ~ {
      parts <- strsplit(target_predictor, " \\+ ")
      sapply(parts, function(x) {
        entropy_part <- sub("^[^_]+_", "", x[1])
        surprisal_part <- sub("^[^_]+_", "", x[2])
        paste(surprisal_part)
      })
    },
    TRUE ~ target_predictor  # Default to original if no match
  ))

# if target_predictor contains the substring "gpt2", rename to GPT-2, if phi2, rename to Phi2, if mistral_base, rename to Mistral, if mistral_instruct, rename to Mistral Instruct
# check if target_predictor contains "gpt2"
dll_baseline_summarized_labels$target_predictor <- ifelse(grepl("gpt2", dll_baseline_summarized_labels$target_predictor), "GPT-2", dll_baseline_summarized_labels$target_predictor)
# check if target_predictor contains "phi2"
dll_baseline_summarized_labels$target_predictor <- ifelse(grepl("phi2", dll_baseline_summarized_labels$target_predictor), "Phi2", dll_baseline_summarized_labels$target_predictor)
# check if target_predictor contains "mistral_base"
dll_baseline_summarized_labels$target_predictor <- ifelse(grepl("mistral_base", dll_baseline_summarized_labels$target_predictor), "Mistral", dll_baseline_summarized_labels$target_predictor)
# check if target_predictor contains "mistral_instruct"
dll_baseline_summarized_labels$target_predictor <- ifelse(grepl("mistral_instruct", dll_baseline_summarized_labels$target_predictor), "Mistral Instruct", dll_baseline_summarized_labels$target_predictor)
# check if target_predictor contains "wizardlm"
dll_baseline_summarized_labels$target_predictor <- ifelse(grepl("wizardlm", dll_baseline_summarized_labels$target_predictor), "WizardLM", dll_baseline_summarized_labels$target_predictor)


# Initialize a list to store the plots
plot_list <- list()

# Loop over each reading measure
for (reading_measure in unique(dll_baseline_summarized_labels$reading_measure)) {
    # Loop over each unique predictability_metric value
    metric_values <- unique(dll_baseline_summarized_labels$predictability_metric)
        
        # Create a plot for each metric
    p <- ggplot(data = dll_baseline_summarized_labels[dll_baseline_summarized_labels$reading_measure == reading_measure & 
                                                 dll_baseline_summarized_labels$prompt == "wo/prompt",],
                                                 # dll_baseline_summarized_labels$predictability_metric == metric, , 
                aes(x = target_predictor, y = m, shape = significance, group = predictability_metric, colour = predictability_metric)) +
        geom_point(position = position_dodge(width = .5), size = 3) +
        geom_errorbar(aes(ymin = lower, ymax = upper), width = .25, position = position_dodge(width = .5), linewidth = 0.4) +
        scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
        # rotate x axis 45 degrees
        # theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
        scale_shape_manual(values = c("not sig." = 21, "sig." = 19)) +  # Hollow for not significant, filled for significant
        # scale_fill_manual(values = c("not sig." = "white", "sig." = "black")) +  # White for not significant, black for significant
        scale_linetype_manual(values = c("solid", "dotted")) +
        geom_hline(yintercept = 0, linetype = "dashed") +
        facet_wrap(~reading_measure) +
        ylab("Delta LL") +
        xlab("Surprisal extraction model") +
        # increase font size of x axis labels, y axis labels, x axis title, y axis title, legend title, legend text
        theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
            axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
            legend.title = element_text(size = 12), legend.text = element_text(size = 12),
            strip.text.x = element_text(size = 10)) +
        ggtitle(paste(reading_measure)) +
        labs(colour = "Predictor", shape = "Significance")  # Rename the legend titles
    
    # Add the plot to the list
    plot_list[[paste(reading_measure)]] <- p
    
    # Combine the plots for the current reading measure into one figure
    combined_plot <- plot_grid(plotlist = plot_list, ncol = 1)  # or use wrap_plots(plot_list) if using patchwork
    
    # Save the combined figure
    ggsave(paste0("results/baseline/dll_baseline_", reading_measure, "_combined.pdf"), plot = combined_plot, width = 7, height = 8, dpi = 200)
    
    # Clear the plot list for the next reading measure
    plot_list <- list()
}

# Initialize a list to store the plots
plot_list <- list()
        
# One plot for all rm
ggplot(data = dll_baseline_summarized_labels[dll_baseline_summarized_labels$prompt == "wo/prompt",],
                                                 # dll_baseline_summarized_labels$predictability_metric == metric, , 
                aes(x = target_predictor, y = m, shape = significance, group = predictability_metric, colour = predictability_metric)) +
        geom_point(position = position_dodge(width = .7), size = 2) +
        geom_errorbar(aes(ymin = lower, ymax = upper), width = .5, position = position_dodge(width = .7), linewidth = 0.4) +
        scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
        # rotate x axis 45 degrees
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        scale_shape_manual(values = c("not sig." = 21, "sig." = 19)) +  # Hollow for not significant, filled for significant
        # scale_fill_manual(values = c("not sig." = "white", "sig." = "black")) +  # White for not significant, black for significant
        scale_linetype_manual(values = c("solid", "dotted")) +
        geom_hline(yintercept = 0, linetype = "dashed") +
        facet_wrap(~reading_measure, nrow=1) +
        ylab("Delta LL") +
        xlab("Surprisal extraction model") +
        # increase font size of x axis labels, y axis labels, x axis title, y axis title, legend title, legend text
        theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
            axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
            legend.title = element_text(size = 12), legend.text = element_text(size = 12),
            strip.text.x = element_text(size = 10)) +
        # legend at bottom
        theme(legend.position = "bottom") +
        labs(colour = "Predictor", shape = "Significance")  # Rename the legend titles
    
    # Save the combined figure
ggsave(paste0("results/baseline/dll_baseline_all_combined.pdf"), width = 12, height = 8, dpi = 200)





####################
### Experiment 1 ###
####################

get_dlls_decoding_e1 <- function(model_df, target_predictors, reading_measures, decoding_strategy, model) {
    all_df <- data.frame()
    for (reading_measure in reading_measures) {
        for (target_predictor in target_predictors) {
            # if target predictor is surprisal, counter_part is same string but replace surprisal with entropy and vice versa
            if (grepl("surprisal", target_predictor)) {
                counter_part <- gsub("surprisal", "entropy", target_predictor)
            } else {
                counter_part <- gsub("entropy", "surprisal", target_predictor)
            }
            formulas <- c(
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", counter_part),
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", counter_part, " + ", target_predictor)
            )
            baseline_dll <- model_cross_val(formulas[1], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            target_dll <- model_cross_val(formulas[2], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            loglik_df <- data.frame(
                delta_basline = baseline_dll,
                delta_target = target_dll,
                delta_loglik = target_dll - baseline_dll,
                target_predictor = target_predictor,
                reading_measure = reading_measure,
                decoding_strategy = decoding_strategy,
                model = model
            )
            all_df <- rbind(all_df, loglik_df)
        }
    }
    return(all_df)
}

model_decoding_combos <- data.frame(unique(decoding_df [,c('model','decoding_strategy')]))

gpt_measures <- c(
    "surprisal_gpt2", "entropy_gpt2"
    # "surprisal_p_gpt2", "entropy_p_gpt2"
)

all_dlls_e1 <- data.frame()
# for each row, get model and decoding strategy
# for (i in 1:2) {
for (i in 1:nrow(model_decoding_combos)) {
    model <- model_decoding_combos[i, 'model']
    decoding_strategy <- model_decoding_combos[i, 'decoding_strategy']
    print(model)
    print(decoding_strategy)
    # get subset of decoding_df
    decoding_subset <- decoding_df[decoding_df$model == model & decoding_df$decoding_strategy == decoding_strategy, ]
    # shuffle
    decoding_subset <- decoding_subset[sample(1:nrow(decoding_subset)), ]
    # call get_dlls_decoding_e1
    dlls_e1 <- get_dlls_decoding_e1(decoding_subset, gpt_measures, reading_measures, decoding_strategy, model)
    dlls_e1$model <- model
    dlls_e1$decoding_strategy <- decoding_strategy
    all_dlls_e1 <- rbind(all_dlls_e1, dlls_e1)
}


# do a paired permutation test to test whether the delta loglik is significantly different from zero
permt_e1 <- all_dlls_e1 %>%
    group_by(target_predictor, reading_measure, decoding_strategy, model) %>%
    do(tidy((paired.perm.test(.$delta_loglik, n.perm = 500, pval = TRUE))))
# rename the column name to p-value
colnames(permt_e1)[5] <- 'p.value'



dll_xmodel_summarized_e1 <- all_dlls_e1 %>%
    group_by(model, target_predictor, decoding_strategy, reading_measure) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()

# add the p value from the permutation test to dll_xmodel_summarized
dll_xmodel_summarized_e1 <- merge(dll_xmodel_summarized_e1, permt_e1, by = c("model", "target_predictor", "decoding_strategy", "reading_measure"), all = TRUE)

# add column that indicates significance
dll_xmodel_summarized_e1$significance <- is_significant(dll_xmodel_summarized_e1$p.value)

# rename labels for target predictor
dll_xmodel_summarized_e1$target_predictor <- factor(dll_xmodel_summarized_e1$target_predictor, labels = c("Entropy GPT-2", "Surprisal GPT-2"))
# rename labels for decoding strategy
dll_xmodel_summarized_e1$decoding_strategy <- factor(dll_xmodel_summarized_e1$decoding_strategy, labels = c("Beam search", "Greedy search", "Sampling", "Top-k", "Top-p"))
# rename labels for model
dll_xmodel_summarized_e1$model <- factor(dll_xmodel_summarized_e1$model, labels = c("Mistral", "Phi2", "WizardLM"))

# save to csv
write.csv(dll_xmodel_summarized_e1, "results/e1/dll_xmodel_summarized_e1.csv", row.names = FALSE)


ggplot(data = dll_xmodel_summarized_e1, aes(x = model, y = m, colour = decoding_strategy, shape=significance)) +
    geom_point(
        position = position_dodge(width = .5), size = 3
    ) +
    geom_errorbar(aes(ymin = lower, ymax = upper),
        width = .1, position = position_dodge(width = .5), linewidth = 0.4
    ) +
    # scale_x_discrete(guide = guide_axis(angle = 45)) +
    scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
    scale_shape_manual(values = c(1, 19)) +
    scale_linetype_manual(values=c("solid", "dotted")) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ylab("Delta LL") +
    xlab("Model") +
    facet_grid(reading_measure~target_predictor) +
    theme(text = element_text(family = "sans")) +
    theme(legend.position = "bottom") +
    guides(colour = guide_legend(title = "Decoding strategy")) +
    guides(shape = guide_legend(title = "Significance")) +
    theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
            axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
            legend.title = element_text(size = 12), legend.text = element_text(size = 12),
            strip.text.x = element_text(size = 10), strip.text.y = element_text(size = 10))

ggsave("results/e1/dll_xmodel_e1.pdf", width = 12, height = 10, dpi = 200)


# run experiment 1 again but without counter part in baseline

get_dlls_decoding_e1_no_counterpart <- function(model_df, target_predictors, reading_measures, decoding_strategy, model) {
    all_df <- data.frame()
    for (reading_measure in reading_measures) {
        for (target_predictor in target_predictors) {
            formulas <- c(
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq"),
                paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", target_predictor)
            )
            baseline_dll <- model_cross_val(formulas[1], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            target_dll <- model_cross_val(formulas[2], model_df, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
            loglik_df <- data.frame(
                delta_basline = baseline_dll,
                delta_target = target_dll,
                delta_loglik = target_dll - baseline_dll,
                target_predictor = target_predictor,
                reading_measure = reading_measure,
                decoding_strategy = decoding_strategy,
                model = model
            )
            all_df <- rbind(all_df, loglik_df)
        }
    }
    return(all_df)
}

all_dlls_e1_no_counterpart <- data.frame()
# for each row, get model and decoding strategy
# for (i in 1:2) {
for (i in 1:nrow(model_decoding_combos)) {
    model <- model_decoding_combos[i, 'model']
    decoding_strategy <- model_decoding_combos[i, 'decoding_strategy']
    print(model)
    print(decoding_strategy)
    # get subset of decoding_df
    decoding_subset <- decoding_df[decoding_df$model == model & decoding_df$decoding_strategy == decoding_strategy, ]
    # shuffle
    decoding_subset <- decoding_subset[sample(1:nrow(decoding_subset)), ]
    # call get_dlls_decoding_e1
    dlls_e1_no_counterpart <- get_dlls_decoding_e1_no_counterpart(decoding_subset, gpt_measures, reading_measures, decoding_strategy, model)
    dlls_e1_no_counterpart$model <- model
    dlls_e1_no_counterpart$decoding_strategy <- decoding_strategy
    all_dlls_e1_no_counterpart <- rbind(all_dlls_e1_no_counterpart, dlls_e1_no_counterpart)
}


# do a paired permutation test to test whether the delta loglik is significantly different from zero
permt_e1_no_counterpart <- all_dlls_e1_no_counterpart %>%
    group_by(target_predictor, reading_measure, decoding_strategy, model) %>%
    do(tidy((paired.perm.test(.$delta_loglik, n.perm = 500, pval = TRUE))))
# rename the column name to p-value
colnames(permt_e1_no_counterpart)[5] <- 'p.value'



dll_xmodel_summarized_e1_no_counterpart <- all_dlls_e1_no_counterpart %>%
    group_by(model, target_predictor, decoding_strategy, reading_measure) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()

# add the p value from the permutation test to dll_xmodel_summarized
dll_xmodel_summarized_e1_no_counterpart <- merge(dll_xmodel_summarized_e1_no_counterpart, permt_e1_no_counterpart, by = c("model", "target_predictor", "decoding_strategy", "reading_measure"), all = TRUE)

# add column that indicates significance
dll_xmodel_summarized_e1_no_counterpart$significance <- is_significant(dll_xmodel_summarized_e1_no_counterpart$p.value)

# rename labels for target predictor
dll_xmodel_summarized_e1_no_counterpart$target_predictor <- factor(dll_xmodel_summarized_e1_no_counterpart$target_predictor, labels = c("Entropy GPT-2", "Surprisal GPT-2"))

# rename labels for decoding strategy
dll_xmodel_summarized_e1_no_counterpart$decoding_strategy <- factor(dll_xmodel_summarized_e1_no_counterpart$decoding_strategy, labels = c("Beam search", "Greedy search", "Sampling", "Top-k", "Top-p"))

# rename labels for model
dll_xmodel_summarized_e1_no_counterpart$model <- factor(dll_xmodel_summarized_e1_no_counterpart$model, labels = c("Mistral", "Phi2", "WizardLM"))

# save to csv
write.csv(dll_xmodel_summarized_e1_no_counterpart, "results/e1/dll_xmodel_summarized_e1_no_counterpart.csv", row.names = FALSE)

ggplot(data = dll_xmodel_summarized_e1_no_counterpart, aes(x = model, y = m, colour = decoding_strategy, shape=significance)) +
    geom_point(
        position = position_dodge(width = .5), size = 3
    ) +
    geom_errorbar(aes(ymin = lower, ymax = upper),
        width = .1, position = position_dodge(width = .5), linewidth = 0.4
    ) +
    # scale_x_discrete(guide = guide_axis(angle = 45)) +
    scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
    scale_shape_manual(values = c(1, 19)) +
    scale_linetype_manual(values=c("solid", "dotted")) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ylab("Delta LL") +
    xlab("Text generation model") +
    facet_grid(reading_measure~target_predictor) +
    theme(text = element_text(family = "sans")) +
    theme(legend.position = "bottom") +
    guides(colour = guide_legend(title = "Decoding strategy")) +
    guides(shape = guide_legend(title = "Significance")) +
    theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
            axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
            legend.title = element_text(size = 12), legend.text = element_text(size = 12),
            strip.text.x = element_text(size = 10), strip.text.y = element_text(size = 10))

ggsave("results/e1/dll_xmodel_e1_no_counterpart.pdf", width = 12, height = 10, dpi = 200)



####################
### Experiment 2 ###
####################
## Test interaction effects of decoding strategy and surprisal/entropy (GPT2-based) on reading measures

named.contr.sum<-function(x, ...) {
    if (is.factor(x)) {
        x <- levels(x)
    } else if (is.numeric(x) & length(x)==1L) {
        stop("cannot create names with integer value. Pass factor levels")
    }
    x<-contr.sum(x, ...)
    colnames(x) <- paste0(rownames(x)[1:(nrow(x)-1)],'_vs_grandmean')
    return(x)
}

# reorder levels of decoding strategy
decoding_df$decoding_strategy <- factor(decoding_df$decoding_strategy, levels = c("topk", "greedy_search", "beam_search", "sampling", "topp"))
unique_models <- unique(decoding_df$model)

reading_measures <- c("Fix", "FPReg", "FPRT", "TFT", "RRT", "RPD_inc")

all_results_e2 <- data.frame()
# for each model
for(model in unique_models) {
    print(model)
    # get subset of model
    model_df <- decoding_df[decoding_df$model == model, ]
    model_df <- preprocess(model_df, PREDICTORS_TO_NORMALIZE)
    model_df$decoding_ <- factor(model_df$decoding_strategy)
    contrasts(model_df$decoding_str) <- named.contr.sum(levels(model_df$decoding_str))
    for(rm in reading_measures) {
        print(rm)
        # log transform if cont
        if (LOG_TF & rm %in% CONT_RESP_VARIABLES) {
            # remove 0s
            model_df_in <- model_df[model_df[[rm]] != 0, ]
            model_df_in[[rm]] <- log(model_df_in[[rm]])
        } else {
            model_df_in <- model_df
        }
        for(metric in c("surprisal_gpt2", "entropy_gpt2")){ 
            form <- paste0(rm, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", metric, " * decoding_")
            if (rm %in% CONT_RESP_VARIABLES) {
                reg_model <- lmer(form, data = model_df_in)
                effects <- data.frame(summary(reg_model)$coefficients[-1, c("Estimate", "Pr(>|t|)", "Std. Error") ])
                colnames(effects)[2] <- "pval"
            }
            else {
                reg_model <- glmer(form, data = model_df_in, family = binomial)
                effects <- data.frame(summary(reg_model)$coefficients[-1, c("Estimate", "Pr(>|z|)", "Std. Error") ])
                colnames(effects)[2] <- "pval"
            }
            effects$effect <- rownames(effects)
            # move effect to first column
            effects <- effects[, c(ncol(effects), 1:(ncol(effects)-1))]
            row.names(effects) <- NULL
            # add rm info and model info
            effects$reading_measure <- rm
            effects$model <- model
            effects$metric <- metric
            all_results_e2 <- rbind(all_results_e2, effects)
        }
    }
}

# check 
all_results_e2$significance <- is_significant(all_results_e2$pval)
# remove "surprisal_gpt2" and "entropy_gpt2" from effect column
all_results_e2$effect <- gsub("surprisal_gpt2", "", all_results_e2$effect)
all_results_e2$effect <- gsub("entropy_gpt2", "", all_results_e2$effect)
# refactor effect
all_results_e2$effect <- factor(all_results_e2$effect)
# drop unused levels
all_results_e2$effect <- droplevels(all_results_e2$effect)


# write results to csv
write.csv(all_results_e2, "results/e2/interaction_effects_e2.csv", row.names = FALSE)


sub_results_e2 <- all_results_e2[grepl(":", all_results_e2$effect), ]
# remove "decoding_" from effect
sub_results_e2$effect <- gsub(":decoding_", "", sub_results_e2$effect)
# assign new lables
sub_results_e2$effect <- factor(sub_results_e2$effect, labels = c("Beam search", "Greedy search", "Sampling", "Top-p"))
# rename metric
sub_results_e2$metric <- factor(sub_results_e2$metric, labels = c("Surprisal GPT-2", "Entropy GPT-2"))
# rename models
sub_results_e2$model <- factor(sub_results_e2$model, labels = c("Mistral", "Phi2", "WizardLM"))

ggplot(data = sub_results_e2, aes(x = effect, y = Estimate, colour = metric, shape=significance)) +
    geom_point(aes(colour = metric), position = position_dodge(width = .5), size = 2) +
    geom_errorbar(aes(ymin = Estimate - Std..Error, ymax = Estimate + Std..Error), width = 0.1, position = position_dodge(width = .5)) +
    # angle x axis 45 degres
    theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
    facet_grid(reading_measure~model, scales="free_y") +
    theme(text = element_text(family = "sans")) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(shape = "Significance", colour = "Metric") +
    scale_shape_manual(values = c(1, 19)) +
    xlab("Effect of decoding strategy (sum-contrast coded)") +
    ylab("Coefficient estimate") +
    theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
            axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
            legend.title = element_text(size = 12), legend.text = element_text(size = 12),
            strip.text.x = element_text(size = 10), strip.text.y = element_text(size = 10))

ggsave("results/e2/interaction_effects_e2_labels.pdf", width = 12, height = 10, dpi = 200)





####################
### Experiment 3 ###
####################
# Assess delta-loglikelihood of transition-score based measures vs "traditional" measures

reading_measures <- c("Fix", "FPReg", "FPRT", "TFT", "RRT", "RPD_inc")

get_dlls_e3 <- function(model_df, all_baselines, reading_measures) {
    # decoding_df, all_pairs, reading_measures, baseline
    all_df <- data.frame()
    for (baseline_predictor in all_baselines) {
        print(baseline_predictor)
        decoding_strategies <- unique(model_df$decoding_strategy)
        # check if surprisal or entropy
        if (grepl("surprisal", baseline_predictor)) {
            target_predictor <- "surprisal_trunc_wo_nl_wl_sum"
            pred_effect <- "surprisal"
        } else {
            target_predictor <- "entropies_trunc_wo_nl_wl_joint"
            pred_effect <- "entropy"
        }
        # if phi2, get phi2 subset and remove beam_search
        if (grepl("phi2", baseline_predictor)) {
            decoding_strategies <- decoding_strategies[decoding_strategies != "beam_search"]
            model <- "phi2"
        } else if (grepl("mistral", baseline_predictor)) {
            model <- "mistral"
        } else if (grepl("wizardlm", baseline_predictor)) {
            model <- "wizardlm"
        } else {
            model <- "unknown" 
        }
        for (decoding_strategy in decoding_strategies) {
            print(decoding_strategy)
            for (reading_measure in reading_measures) {
                print(reading_measure)
                decoding_model_subset <- model_df[model_df$decoding_strategy == decoding_strategy & model_df$model == model, ]
                decoding_model_subset <- decoding_model_subset[sample(1:nrow(decoding_model_subset)), ]
                formulas <- c(
                    paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", baseline_predictor),
                    paste0(reading_measure, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", target_predictor)
                )
                baseline_dll <- model_cross_val(formulas[1], decoding_model_subset, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
                target_dll <- model_cross_val(formulas[2], decoding_model_subset, reading_measure, mixed_effects = TRUE, log_transform = TRUE)
                loglik_df <- data.frame(
                    model = model,
                    decoding_strategy = decoding_strategy,
                    predictability_effect = pred_effect,
                    delta_basline = baseline_dll,
                    delta_target = target_dll,
                    delta_loglik = target_dll - baseline_dll,
                    baseline_predictor = baseline_predictor,
                    reading_measure = reading_measure
                )
                all_df <- rbind(all_df, loglik_df)
            }
        }
    }
    return(all_df)
}

mistral_surp <- c("surprisal_mistral_base", "surprisal_p_mistral_base", 
                     "surprisal_mistral_instruct", "surprisal_p_mistral_instruct")
mistral_ent <- c("entropy_mistral_base", "entropy_p_mistral_base", 
                    "entropy_mistral_instruct", "entropy_p_mistral_instruct")
phi_surp <- c("surprisal_p_phi2", "surprisal_phi2")
phi_ent <- c("entropy_p_phi2", "entropy_phi2")
wizardlm_surp <- c("surprisal_p_wizardlm", "surprisal_wizardlm")
wizardlm_ent <- c("entropy_p_wizardlm", "entropy_wizardlm")
all_baselines <- c(mistral_surp, mistral_ent, phi_surp, phi_ent, wizardlm_surp, wizardlm_ent)

all_dll_e3 <- get_dlls_e3(decoding_df, all_baselines, reading_measures)

write.csv(all_dll_e3, "results/e3/dll_xmodel_e3.csv", row.names = FALSE)


# permutation tests
permt_e3 <- all_dll_e3 %>%
    group_by(model, decoding_strategy, predictability_effect, baseline_predictor, reading_measure) %>%
    do(tidy((paired.perm.test(.$delta_loglik, n.perm = 500, pval = TRUE))))
colnames(permt_e3)[6] <- "p.value"

# t test
tt_e3 <- all_dll_e3 %>%
    group_by(model, decoding_strategy, predictability_effect, baseline_predictor, reading_measure) %>%
    do(tidy((t.test(.$delta_loglik, mu = 0, alternative = "two.sided"))))


dll_xmodel_summarized_e3 <- all_dll_e3 %>%
    group_by(model, decoding_strategy, predictability_effect, baseline_predictor, reading_measure) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()


# merge tt and dll_xscore_summarized on model and score and keep all columns
dll_xmodel_summarized_e3 <- merge(dll_xmodel_summarized_e3, tt_e3, by = c("model", "decoding_strategy", "baseline_predictor", "predictability_effect", "reading_measure"), all = TRUE)

dll_xmodel_summarized_e3$significance <- is_significant(dll_xmodel_summarized_e3$p.value)
dll_xmodel_summarized_e3$significance <- as.factor(dll_xmodel_summarized_e3$significance)

# new variable if "surprisal" in baseline_predictor then "surprisal" else "entropy"
dll_xmodel_summarized_e3$predictability_effect <- ifelse(grepl("surprisal", dll_xmodel_summarized_e3$baseline_predictor), "surprisal", "entropy")
# new variable if "_p_" in baseline_predictor then "w/prompt" else "wo/prompt"
dll_xmodel_summarized_e3$prompt <- ifelse(grepl("_p_", dll_xmodel_summarized_e3$baseline_predictor), "w/prompt", "wo/prompt")

# remove _entropy, _surprisal from baseline_predictor
dll_xmodel_summarized_e3$baseline_predictor <- gsub("entropy_", "", dll_xmodel_summarized_e3$baseline_predictor)
# remove _p from baseline_predictor
dll_xmodel_summarized_e3$baseline_predictor <- gsub("p_", "", dll_xmodel_summarized_e3$baseline_predictor)

write.csv(dll_xmodel_summarized_e3, "results/e3/dll_xmodel_summarized_e3.csv", row.names = FALSE)


# rename labels for target predictor
dll_xmodel_summarized_e3$baseline_predictor <- factor(dll_xmodel_summarized_e3$baseline_predictor, labels = c("Mistral", "Mistral Instruct", "Phi2", "Mistral", "Mistral Instruct", "Phi2", "WizardLM", "WizardLM"))
# rename labels for decoding strategy
dll_xmodel_summarized_e3$decoding_strategy <- factor(dll_xmodel_summarized_e3$decoding_strategy, labels = c("Beam search", "Greedy search", "Sampling", "Top-k", "Top-p"))



for (red_mes in unique(dll_xmodel_summarized_e3$reading_measure)) {
    
    # Create the plot with separate panels for model and predictability_effect
    p <- ggplot(data = dll_xmodel_summarized_e3[dll_xmodel_summarized_e3$reading_measure == red_mes & dll_xmodel_summarized_e3$prompt == "w/prompt", ], 
                aes(x = decoding_strategy, y = m, colour = baseline_predictor, shape = significance)) +
        geom_point(position = position_dodge(width = .5), size = 3) +
        geom_errorbar(aes(ymin = lower, ymax = upper),
                      width = .1, position = position_dodge(width = .5), linewidth = 0.4) +
        scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
        scale_shape_manual(values = c(1, 19)) +
        geom_hline(yintercept = 0, linetype = "dashed") +
        ylab("Delta LL") +
        xlab("Decoding Strategy") +
        # rename label for color
        labs(colour = "Surprisal extraction model", shape = "Significance") +
        
        # Facet grid by model and predictability_effect
        facet_grid(model ~ predictability_effect, scales = "free_x") +
        
        theme(text = element_text(family = "sans")) +
        theme(legend.position = "bottom") + 
        theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
            axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
            legend.title = element_text(size = 12), legend.text = element_text(size = 12),
            strip.text.x = element_text(size = 10), strip.text.y = element_text(size = 10))
    
    # Save the plot
    ggsave(paste0("results/e3/dll_xmodel_e3_", red_mes, ".pdf"), plot = p, width = 12, height = 6, dpi = 200)
}
