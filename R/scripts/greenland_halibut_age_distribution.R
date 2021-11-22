# Script for analysis of the Greenland halibut age distribution

library(ggplot2)
library(viridis)

# Load dataframe
dataset = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Deep learning applied to fish otolith images\Data\Blåkveiteotolitter\dataframe.csv)')

# Set colors for the two sexes
my_colors = viridis(2, begin = 0.5, end = 0.9)

# Create a temporary dataframe adding a term to the male length
# to make a plot with non-overlapping points.
df.temp = dataset
df.temp$length[df.temp$sex == 'male'] = df.temp$length[df.temp$sex == 'male'] + .5

# Plot age vs length for both groups
ggplot(df.temp, aes(y = age, x = length)) + 
  geom_count(aes(color = sex), alpha = 0.6) + 
  scale_color_manual(values = my_colors, labels=c('Females', 'Males')) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20)) + 
  labs(color='Sex')

# Plot age vs length for females
ggplot(subset(dataset, sex == 'female')) + 
  geom_count(aes(y = age, x = length), alpha = 0.5, color = my_colors[1]) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20))

# Plot age vs length for males
ggplot(subset(dataset, sex == 'male')) + 
  geom_count(aes(y = age, x = length), alpha = 0.5, color = my_colors[2]) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20))

# Discard data with incomplete features so we can fit a model 
# predicting the age using length and sex.
dataset = dataset[rowSums(is.na(dataset)) == 0, ]

# Fit Von Bertalenffy model to the data
# Define VBGF loss function
loss = function(params){
  labels = dataset$age
  predictions = params[1] + params[2]*log(1 - dataset$length/params[3]) + 
    (params[4]*log(1 - dataset$length/params[5]))*(dataset$sex == 'male')
  return(sum((labels - predictions)**2) / (dim(dataset)[1] - 1 - 5))
}

# Find model parameters using numerical optimization
params = optim(c(1, 1, 10000, 1, 10000), loss)$par

# Function generator that returns an age prediction function
pred_func = function(params, sex){
  return(function(x){
    return(
      params[1] + params[2]*log(1 - x/params[3]) + (params[4]*log(1 - x/params[5]))*(sex == 'male')
    )})
}

# Define an age prediction function for each sex
male_age = pred_func(params, 'male')
female_age = pred_func(params, 'female')

# Display age distribution for each sex
ggplot(dataset, aes(x = age, fill = sex)) + 
  geom_bar(aes(y = ..count..), binwidth = 1, alpha = 0.6, position = 'identity') + 
  geom_text(stat='count', aes(label=..count..), vjust = -1) + 
  theme_classic() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2)) + 
  theme(text = element_text(size = 20)) + 
  labs(fill='Sex')

# Display length distribution for each sex
ggplot(dataset, aes(x = length, fill = sex)) + 
  geom_histogram(aes(y = ..density..), binwidth = 2, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3, show.legend = FALSE, aes(color = sex)) + 
  theme_classic() + 
  xlab('Length (cm)') + 
  ylab('') + 
  scale_x_continuous() + 
  theme(text = element_text(size = 20)) + 
  labs(fill='Sex')