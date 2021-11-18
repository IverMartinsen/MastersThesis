library(ggplot2)
library(reshape2)
library(viridis)
source(integral_numerical)

results = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Grønlandskveiteotolitter\Validation Procedure\results.csv)')
linear_results = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Grønlandskveiteotolitter\Validation Procedure\linear_results.csv)')
summary = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Grønlandskveiteotolitter\Validation Procedure\summary.csv)')

results$y2 = linear_results$y2

# Create data frame that compares age with prediction
df = melt(results, measure.vars = c('age', 'y3'))

p = ggplot(df, aes(x = value, fill = interaction(sex, variable))) + 
  geom_density(alpha = .1)

# Create values for difference plots
dense = ggplot_build(p)$data[[1]]

ymin_m = pmin(dense[dense$group==2, ]$y, dense[dense$group==4, ]$y)
ymax_m = pmax(dense[dense$group==2, ]$y, dense[dense$group==4, ]$y)
ymin_f = pmin(dense[dense$group==1, ]$y, dense[dense$group==3, ]$y)
ymax_f = pmax(dense[dense$group==1, ]$y, dense[dense$group==3, ]$y)
x = dense$x[1:512]

# Plot difference in distributions
ggplot() +
  geom_ribbon(aes(x = x, ymin = ymin_m, ymax = ymax_m, fill = 'Male')) + 
  geom_ribbon(aes(x = x, ymin = ymin_f, ymax = ymax_f, fill = 'Female')) +
  geom_density(aes(x = results$age[which(results$sex == 'male')]), size = 1) + 
  geom_density(aes(x = results$age[which(results$sex == 'female')]), size = 1) + 
  theme_classic() + 
  labs(x = 'Age', fill = 'Sex', title = 'Predictions by deep learning and length') + 
  scale_fill_manual(values = viridis(20)[c(10, 19)]) + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) 

# Computing accuracies
n = dim(results)[1]

# Total x-off
sum(abs(round(results$y2) - results$age) <= 2)*100 / n
mean((results$age - results$y2)**2)

# x-off males
results_male = results[results$sex == 'male', ]
num_males = dim(results_male)[1]
sum(abs(round(results_male$y2) - results_male$age) <= 2)*100 / num_males
mean((results_male$age - results_male$y2)**2)

# x-off females
results_female = results[results$sex == 'female', ]
num_females = dim(results_female)[1]
sum(abs(round(results_female$y2) - results_female$age) <= 2)*100 / num_females
mean((results_female$age - results_female$y2)**2)

# Correlations
cor(results[, c(3, 6, 7, 8)])

temp = results
temp$age[which(temp$sex == 'male')] = temp$age[which(temp$sex == 'male')] + 0.3

# Scatter plot age vs age
ggplot(temp, aes(x = age, y = round(y3))) + 
  geom_count(aes(colour = factor(sex)), alpha = 1) + 
  scale_size_area(max_size = 16) + 
  theme_classic() + 
  geom_abline(slope = 1, intercept = 0, linetype = 'dotted') + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
  scale_y_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
  labs(x = 'True age', y = 'Predicted age', title = 'Combined predictions') + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  scale_colour_discrete(name = 'Sex', labels = c('Female', 'Male'))


spread = melt(results, measure.vars = c('y2', 'y1'))
spread$age[which(spread$variable == 'y1')] = spread$age[which(spread$variable == 'y1')] + 0.3

ggplot(spread[which(spread$sex == 'female'), ], aes(x = age, y = round(value), color = variable)) + 
  geom_count() + 
  scale_size_area(max_size = 16) + 
  geom_abline(slope = 1, intercept = 0, size = 1) + 
  theme_classic() + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
  scale_y_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
  labs(x = 'True age', y = 'Predicted age', title = 'Female predictions') + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  scale_colour_discrete(name = 'Model', labels = c('Length', 'Deep learning'))


results$mean = 0
for(i in 1:26){
  idx = which(round(results$y1) == i)
  results$mean[idx] = mean(results[idx, ]$age - round(results[idx, ]$y1))
}

# Plot residuals
ggplot(results, aes(x = round(y1), y = round(y1) - age)) + 
  geom_count(aes(color = -abs(round(y1) - age + mean))) + 
  #geom_point(aes(x = round(y2), y = -mean), size = 4) + 
  scale_color_viridis(guide = FALSE) +
  scale_size_area(max_size = 16) + 
  theme_minimal() + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
  scale_y_continuous(limits=c(-15, 15)) + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  labs(x = 'Predicted age', y = 'Residual', title = 'Deep learning predictions')

for(i in 1:26){
  print(mean((results$age - results$y1)[which(results$age == i)]**2) - 
  mean((results$age - results$y3)[which(results$age == i)]**2))
}    

sum((results$age - results$y3)[which(results$age < 11 | results$age > 15)]**2) / 3540

sum((abs(results$age - round(results$y1)) < 3)[which(results$age >= 11 & results$age <= 15)]) / length(which(results$age >= 11 & results$age <= 15))

a = -1
b = 24
sex = 'female'
p = density(results[which(results$sex == sex), ]$age, from = a, to = b)$y
q = density(results[which(results$sex == sex), ]$y1, from = a, to = b)$y

-integrate_num(p*log(q/p), a, b, 'simpson')

max(density(results[which(results$sex == sex), ]$y3)$x)

sigma1 = sd(results[which(results$sex == 'female'), ]$age)
sigma2 = sd(results[which(results$sex == 'female'), ]$y3)
mu1 = mean(results[which(results$sex == 'female'), ]$age)
mu2 = mean(results[which(results$sex == 'female'), ]$y3)

log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2*sigma2**2) - 0.5

y3 = rep(0, 26)
for(i in 1:26){
  idx = which(results$age == i)
  y3[i] = sum((results[idx, ]$age - results[idx, ]$y3)**2 / 3540)
}

test = data.frame(x = c(1:26, 1:26, 1:26), y = c(y1, y2, y3), f = c(rep('a', 26), rep('b', 26), rep('c', 26)))

ggplot(test, aes(x = x, y = y, fill = factor(f, levels = c('b', 'a', 'c')))) + 
  geom_line(position = 'identity', size = 2) + 
  geom_area(position = 'identity', alpha = 0.6) + 
  theme_classic() + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  labs(x = 'Age', y = 'Squared error') + 
  scale_fill_discrete(name = '', labels = c('Length', 'Image', 'Combined'))

model = lm(age ~ y1, results)

mean((sqrt(2*(results$age - results$y3)**2) / (results$age + results$y3))[dataset$sex == 'female'])*100


std = rep(0, 26)

observed_variances_male = data.frame(y1 = rep(0, 26), y2 = rep(0, 26), y3 = rep(0, 26))
observed_variances_female = data.frame(y1 = rep(0, 26), y2 = rep(0, 26), y3 = rep(0, 26))


for(i in 1:26){
  observed_variances_male$y1[i] = sd(results[which(round(results$y1) == i & results$sex == 'male'), ]$y1)
  observed_variances_female$y1[i] = sd(results[which(round(results$y1) == i & results$sex == 'female'), ]$y1)
}

observed_variances_male$sex = 'male'
observed_variances_female$sex = 'female'

test = rbind(observed_variances_male, observed_variances_female)
test$age = c(1:26, 1:26)

ggplot(test, aes(x = age, y = y1, color = sex)) + 
  geom_point(size = 8) + 
  theme_classic() + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  labs(x = 'Age', y = 'Standard deviation', title = 'Deep learning standard errors') + 
  scale_color_discrete(name = '', labels = c('Female', 'Male'))
