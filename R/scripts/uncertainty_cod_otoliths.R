library(reshape2)
library(ggplot2)

# Load dataframe of scores
individual_scores = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Torskeotolitter\Forsøk\Forsøk 13.08.2021\individual_scores.csv)', sep = ';', dec = ',')
colnames(individual_scores)[colnames(individual_scores) == colnames(individual_scores)[1]] = 'filename'

# Number of samples
num_samples = length(individual_scores$filename)

# Squeeze individual scores
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
  geom_density(position = 'identity', aes(x = score), alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = 'dashed') + 
  scale_fill_discrete(name = 'Stock', labels = c('NEAC', 'NCC')) + 
  theme_classic() + 
  theme(text = element_text(size = 20))

# Create histogram of trial-wise score distributions
ggplot(df[which((df$trial == 'X9'|df$trial == 'X10'|df$trial == 'X11'|df$trial == 'X12') & df$ncc == FALSE), ], aes(x = score, fill = trial)) + 
  geom_density(position = 'identity', aes(x = score), alpha = .8) +
  scale_fill_discrete(name = 'Trial no.', labels = c(5, 6, 7, 8)) +
  theme_classic() + 
  theme(text = element_text(size = 20), plot.title = element_text(hjust = 0.5)) + 
  labs(title = 'Norwegian Coastal Cod')

# Pooled variance
mean(apply(squeezed_scores[which(squeezed_scores$ncc == TRUE), seq(2, 5)], 1, var))

# Number of samples with ambigous classifications
length(which(apply(sign(squeezed_scores[, 1:4]), 1, (function(x) length(unique(x)) != 1))))
