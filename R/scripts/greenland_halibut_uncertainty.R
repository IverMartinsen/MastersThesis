################################################################
### Script for analysis of the Greenland halibut test results###
################################################################

library(ggplot2)
library(reshape2)
library(viridis)
library(gridExtra)
source('../libraries/utils.R')

# Load individual and summary results
results = read.csv(r'(C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith images\Artikkel om blåkveiteotolitter\Resultater\results.csv)')
summary = read.csv(r'(C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith images\Artikkel om blåkveiteotolitter\Resultater\summary.csv)')


draw_differences = function(y, title){
    #' Highlights the area which differs between the labeled distribution and predicted distribution.
    #'
    #' y, character: predictions to analyze, e.g. 'y1'
    #' title, character: plot title, e.g. 'Predictions by deep learning'
  
    # Create data frame that compares age with prediction
    df.temp = melt(results, measure.vars = c('age', y))

    plot.temp = ggplot(df.temp, aes(x = value, fill = interaction(sex, variable))) + 
        geom_density()

    # Create values for difference plots
    dense = ggplot_build(plot.temp)$data[[1]]

    # Create vector of min/max values for each sex that defines the ribbons to be drawn
    ymin_male = pmin(dense[dense$group == 2, ]$y, dense[dense$group == 4, ]$y)
    ymax_male = pmax(dense[dense$group == 2, ]$y, dense[dense$group == 4, ]$y)
    ymin_female = pmin(dense[dense$group == 1, ]$y, dense[dense$group == 3, ]$y)
    ymax_female = pmax(dense[dense$group == 1, ]$y, dense[dense$group == 3, ]$y)
  
    # Vector of x values for ribbons
    x = dense$x[1:512]

    # Plot difference in age distributions
    p = ggplot() +
        # Plot ribbons showing age distribution differences
        geom_ribbon(aes(x = x, ymin = ymin_male, ymax = ymax_male, fill = 'Male')) + 
        geom_ribbon(aes(x = x, ymin = ymin_female, ymax = ymax_female, fill = 'Female')) +
        # Plot density line for labeled age
        geom_density(aes(x = value, color = sex, linetype = variable), size = 1, data = melt(results, measure.vars = c('age', y))) + 
        # Layout configurations
        theme_classic() + 
        labs(x = 'Age', fill = 'Sex', title = title) + 
        theme(text = element_text(size = 10), plot.title = element_text(hjust = 0.5), legend.position = c(0.85, 0.7)) + 
        scale_fill_discrete(name = '') + 
        scale_color_manual(values = c('black', 'black'), guide = 'none') + 
        scale_linetype_discrete(name = '', labels = c('Read age', 'Predictions'))
    
    return(p)
}


# Display difference in age distributions
p1 = draw_differences('y1', 'Image based estimates')
p2 = draw_differences('y2', 'Length based estimates')

grid.arrange(p1, p2, ncol = 2)


# Compute and print accuracy and loss for all models/sexes 
# by iterating over the columns that store the loss.
for(i in 6:7){
    results_male = results[results$sex == 'male', ]
    results_female = results[results$sex == 'female', ]
    
    print(colnames(results)[i])
    print('')
    print(paste0('MSE total:   ', mean((results$age - results[, i])**2)))
    print(paste0('MSE males:   ', mean((results_male$age - results_male[, i])**2)))
    print(paste0('MSE females: ', mean((results_female$age - results_female[, i])**2)))
    
    print('')
    print(paste0('CV total:   ', mean(sqrt((2/(1 + results[, i]/results$age) - 1)**2 + (2/(1 + results$age/results[, i]) - 1)**2))))
    print(paste0('CV males:   ', mean(sqrt((2/(1 + results_male[, i]/results_male$age) - 1)**2 + (2/(1 + results_male$age/results_male[, i]) - 1)**2))))
    print(paste0('CV females: ', mean(sqrt((2/(1 + results_female[, i]/results_female$age) - 1)**2 + (2/(1 + results_female$age/results_female[, i]) - 1)**2))))
    
    for(j in 0:2){
        print('')
    
        # Total j-off percentage and mean squared error
        print(paste0(j, '-off percentage total:   ', sum(abs(results$age - round(results[, i])) <= j)*100 / dim(results)[1])) 
    
        # j-off and mean squared error for males
        print(paste0(j, '-off percentage males:   ', sum(abs(results_male$age - round(results_male[, i])) <= j)*100 / dim(results_male)[1]))
    
        # j-off percentage and mean squared error for females
        print(paste0(j, '-off percentage females: ', sum(abs(results_female$age - round(results_female[, i])) <= j)*100 / dim(results_female)[1]))
        }
    print('')
    }


# Add a small term to the male age to enable pairwise plotting of of both sexes
df.temp = results
df.temp$age[which(df.temp$sex == 'male')] = df.temp$age[which(df.temp$sex == 'male')] + 0.4

# Scatter plot predicted age vs labeled age
scatterplot = ggplot(df.temp, aes(x = age, y = round(y1), color = sex)) + 
    geom_count(aes(colour = sex), alpha = 1) + 
    scale_size_area(max_size = 6) + 
    theme_minimal() + 
    geom_abline(slope = 1, intercept = 0, linetype = 'dotted') + 
    scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
    scale_y_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
    labs(x = 'True age', y = 'Predicted age', title = 'A') + 
    theme(text = element_text(size = 10), plot.title = element_text(hjust = 0.5), legend.position = c(0.1, 0.7)) + 
    scale_color_discrete(name = 'Sex', labels = c('Female', 'Male'))




plot_residuals = function(y, title){
    #' Produce ggplot of residuals vs predictions
    #' 
    #' y: prediction to use, e.g. 'y1'
    #' title: plot title, e.g. 'Deep learning predictions'
  
    # Compute the mean residual for each age
    df.temp = results
    df.temp$mean = 0
  
    for(i in 1:26){
        idx = which(round(df.temp[, y]) == i)
        df.temp$mean[idx] = mean(df.temp[idx, ]$age - round(df.temp[idx, y]))
        }
  
    # Round age predictions and compute rounded residuals
    df.temp[, y] = round(df.temp[, y])
    df.temp$res = df.temp[, y] - df.temp$age
  
    # Plot residuals
    p = ggplot(df.temp, aes_string(x = y, y = 'res')) + 
        geom_count(aes(color = -abs(res + mean)), show.legend = FALSE) + 
        scale_color_viridis(guide = 'none') +
        scale_size_area(max_size = 4) + 
        theme_minimal() + 
        scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
        scale_y_continuous(limits=c(-11, 11)) + 
        theme(text = element_text(size = 10), plot.title = element_text(hjust = 0.5)) + 
        labs(x = 'Predicted age', y = 'Residual', title = title) + 
        geom_line(aes(x = y1, -mean))
  
    return(p)
}



# Display residuals
residual_plot = plot_residuals('y1', 'B')


# Plot standard errors vs predicted age
y = 'y1'
title = 'C'

for(sex in c('male', 'female')){
    variances_ = data.frame(y1 = rep(0, 26), y2 = rep(0, 26), y3 = rep(0, 26))
    for(i in 1:26){
        variances_[i, y] = sd(results[which(round(results[, y]) == i & results$sex == sex), y])
        variances_$sex = sex
        variances_$age = 1:26
    }
    if(sex == 'male'){
        variances = variances_  
    }else if(sex == 'female'){
        variances = rbind(variances, variances_)  
    }
}

standard_errors = ggplot(variances, aes_string(x = 'age', y = y, color = 'sex')) + 
    geom_point(size = 2) + 
    theme_minimal() + 
    scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
    theme(text = element_text(size = 10), plot.title = element_text(hjust = 0.5), legend.position = c(0.9, 0.2)) + 
    labs(x = 'Age', y = 'Standard deviation', title = title) + 
    scale_color_discrete(name = '', labels = c('Female', 'Male'))

grid.arrange(scatterplot, residual_plot, standard_errors, layout_matrix = rbind(c(1, 1), c(2, 3)))


# For each model, for each age, compute MSE contribution
y_vals = c()
for(y in c('y1', 'y2')){
    temp = rep(0, 26)
    for(i in 1:26){
        idx = which(results$age == i)
        temp[i] = sum((results[idx, 'age'] - results[idx, y])**2 / 3540)
        }
    y_vals = c(y_vals, temp)
}


# Create dataframe with age-wise MSE contributions with model as factor variable
df.temp = data.frame(
    x = c(1:26, 1:26), 
    y = y_vals, 
    f = c(rep('deep_learning', 26), rep('length', 26))
    )


# Plot MSE contributions against age for all three models
ggplot(df.temp, aes(x = x, y = y, fill = factor(f, levels = c('length', 'deep_learning')), color = f)) + 
    geom_area(position = 'identity', alpha = 0.6) + 
    theme_classic() + 
    scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
    theme(text = element_text(size = 10), plot.title = element_text(hjust = 0.5), legend.position = c(0.9, 0.9)) + 
    labs(x = 'Age', y = 'Squared error') + 
    scale_fill_discrete(name = '', labels = c('Length', 'Image')) + 
    scale_color_manual(values = c('black', 'black'), guide = 'none')


#########################
### Confidence bounds ###
#########################

# wrapper function to get true ages based on sex
get_x_vals = function(sex) seq(min(results$age[which(results$sex == sex)]), max(results$age[which(results$sex == sex)]))


get_lower_bound = function(sex){
    #' Produce lower confidence bound for sex
    #' 
    #' 
    x = get_x_vals(sex)
    
    lower = numeric(length(x))
    for(i in 1:length(x)){
        estimates = results$y1[which(results$age == x[i] & results$sex == sex)]
        n = length(estimates)
        lower[i] = sort(estimates)[floor(n*0.05) + 1]
    }
    return(lower)
}


get_upper_bound = function(sex){
    #' Produce upper confidence bound for sex
    #' 
    #' 
    x = get_x_vals(sex)
    
    upper = numeric(length(x))
    for(i in 1:length(x)){
        estimates = results$y1[which(results$age == x[i] & results$sex == sex)]
        n = length(estimates)
        upper[i] = sort(estimates)[ceiling(n*0.95)]
    }
    return(upper)
}


# Plot confidence bounds
ggplot() + 
    geom_line(aes(get_x_vals('female'), get_x_vals('female'))) + 
    geom_line(aes(get_x_vals('male'), get_lower_bound('male')), linetype = 2) + 
    geom_line(aes(get_x_vals('male'), get_upper_bound('male')), linetype = 2) + 
    geom_line(aes(get_x_vals('female'), get_lower_bound('female'), color = 'test'), linetype = 2) + 
    geom_line(aes(get_x_vals('female'), get_upper_bound('female')), linetype = 2) + 
    theme_classic()


# Create dataframe for comparing age vs age for length model and deep learning model
df.temp = melt(results, measure.vars = c('y2', 'y1'))
# Add a small term to the x-axis for the length predictions to enable pairwise comparison
df.temp$age[which(df.temp$variable == 'y1')] = df.temp$age[which(df.temp$variable == 'y1')] + 0.3


# Plot difference in age-vs-age for length and deep learning
ggplot(df.temp[which(df.temp$sex == 'female'), ], aes(x = age, y = round(value), color = variable)) + 
    geom_count() + 
    scale_size_area(max_size = 16) + 
    geom_abline(slope = 1, intercept = 0, size = 1) + 
    theme_classic() + 
    scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
    scale_y_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
    labs(x = 'True age', y = 'Predicted age', title = 'Female predictions') + 
    theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
    scale_colour_discrete(name = 'Model', labels = c('Length', 'Deep learning'))
