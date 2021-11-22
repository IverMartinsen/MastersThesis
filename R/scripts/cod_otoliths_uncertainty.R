# Script for analysis of cod otolith uncertainty

library(reshape2)
library(ggplot2)

# Load dataframe of scores and summary results
individual_scores = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Deep learning applied to fish otolith images\Torskeotolitter\Forsøk\Forsøk 13.08.2021\individual_scores.csv)', sep = ';', dec = ',')
colnames(individual_scores)[colnames(individual_scores) == colnames(individual_scores)[1]] = 'filename'
summary_results = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Deep learning applied to fish otolith images\Torskeotolitter\Forsøk\Forsøk 13.08.2021\summary_results.csv)', sep = ';', dec = ',')

# Number of samples
num_samples = length(individual_scores$filename)

# Create dataframe that squeezes the individual scores into 4 colums
squeezed_scores = data.frame(
  filename = individual_scores$filename,
  'trial1' = rep(0, num_samples),
  'trial2' = rep(0, num_samples),
  'trial3' = rep(0, num_samples),
  'trial4' = rep(0, num_samples),
  ncc = grepl('cc', individual_scores$filename)
)

for(i in 1:num_samples){
  squeezed_scores[i, 2:5] = individual_scores[i, seq(2, 21)][is.na(individual_scores[i, seq(2, 21)]) == FALSE]
}

# Create dataframe of scores, use class label as factor variabel
melted_scores = melt(individual_scores, na.rm = TRUE)

# Merge all relevant data into one dataframe
df = data.frame(
  filename = melted_scores$filename,
  score = melted_scores$value,
  ncc = grepl('cc', melted_scores$filename),
  trial = melted_scores$variable
)

# Create histogram of class-wise score distributions
ggplot(df, aes(x = score, fill = ncc)) +
  geom_density(position = 'identity', binwidth = 1, aes(x = score), alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = 'dashed') + 
  scale_fill_discrete(name = 'Stock', labels = c('NEAC', 'NCC')) + 
  theme_classic() + 
  theme(text = element_text(size = 20))

# create plot of weighted density
ncc_curve = density(df$score[which(df$ncc == TRUE)])
neac_curve = density(df$score[which(df$ncc == FALSE)])
x.temp = c(neac_curve$x, ncc_curve$x)
y.temp = c(neac_curve$y*243/610, ncc_curve$y*367/610)
f.temp = c(rep('neac', length(neac_curve$x)), rep('ncc', length(ncc_curve$x)))
df.scaled = data.frame(x = x.temp, y = y.temp, f = f.temp)
ggplot(df.scaled, aes(x = x, y = y, fill = f)) + 
  geom_area(alpha=.6) + 
  geom_vline(xintercept = 0, linetype = 'dashed') + 
  scale_fill_discrete(name = 'Stock', labels = c('NCC', 'NEAC')) + 
  theme_classic() + 
  theme(text = element_text(size = 20))

# Create histogram of trial-wise score distributions for trial 1 to 4
ggplot(df[which((df$trial == 'X1'|df$trial == 'X2'|df$trial == 'X3'|df$trial == 'X4') & df$ncc == TRUE), ], aes(x = score, fill = trial)) + 
  geom_density(position = 'identity', aes(x = score), alpha = .8) +
  scale_fill_discrete(name = 'Trial no.', labels = c(1, 2, 3, 4)) +
  theme_classic() + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  labs(title = 'Norwegian Coastal Cod') + 
  ylim(0, 0.3) + 
  xlim(-10, 10)

# Print mean of scores
for(bool in c(TRUE, FALSE)){
  print(paste0('NCC: ', bool))
  print('')
  n = 4*length(which(squeezed_scores$ncc == bool))
  mean_score = sum(squeezed_scores[which(squeezed_scores$ncc == bool), 2:5]) / (n)
  print(paste0('Mean score: ', mean_score))
  print(paste0('SST: ', var(unlist(squeezed_scores[which(squeezed_scores$ncc == bool), 2:5]))*(n - 1)))
  print(paste0('SSA: ', 4*sum((apply(squeezed_scores[which(squeezed_scores$ncc == bool), seq(2, 5)], 1, mean) - mean_score)**2)))
  print('')
}

# Number of samples with ambigous classifications
length(which(apply(sign(squeezed_scores[, 1:4]), 1, (function(x) length(unique(x)) != 1))))

# x-axis for the pdf's
x = seq(-10, 10, 0.01)

# y-axis for the pdf's
y1 = 367*dnorm(x, mean = -3, sd = sqrt(6.25))/610
y2 = 243*dnorm(x, mean = 1.97, sd = sqrt(6.30))/610

# Make plot showing the probability of misclassification
ggplot() + 
  geom_line(size = 2, aes(c(x, x), c(y1, y2), color = c(rep('NCC', length(x)), rep('NEAC', length(x))))) +
  geom_area(aes(x[x>=0], y1[x>=0])) + 
  geom_area(aes(x[x<=0], y2[x<=0])) + 
  theme_classic() + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  labs(x = 'Score', y = 'Density', title = 'Probability of misclassification') + 
  scale_color_discrete(name = '')