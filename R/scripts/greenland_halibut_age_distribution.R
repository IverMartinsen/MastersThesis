# Script for analysis of the Greenland halibut age distribution

library(ggplot2)
library(viridis)
library(dplyr)
library(gridExtra)

# Load dataframe with all available features arranged by filename
dataset = read.csv(r'(C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith images\Data\Blåkveiteotolitter\dataframe_new.csv)')

# Helper function to generate colors from the standard ggplot2 palette
get_colors = function(n) hcl(h = seq(15, 375, length = n + 1), l = 65, c = 100)[1:n]

# Display age distribution for each sex
p1 = ggplot(dataset[which(dataset$sex != "unknown"), ], aes(x = age, fill = sex)) + 
  geom_bar(aes(y = ..count..), alpha = 0.5, position = 'identity') + 
  geom_text(stat='count', aes(label=..count..), vjust = -1, size = 2) + 
  theme_classic() + 
  xlab('') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2)) + 
  theme(text = element_text(size = 10), legend.position = c(0.85, 0.65), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  ylim(0, 240) + 
  labs(fill='Sex') + 
  geom_blank(aes(fill = sex), data = dataset) + 
  scale_fill_manual(values = c(get_colors(2), get_colors(6)[2]))

# Display age distribution for unknown data
p2 = ggplot(dataset[which(dataset$sex == "unknown"), ], aes(x = age, fill = sex)) + 
  geom_bar(aes(y = ..count..), alpha = 0.5, position = 'identity') + 
  geom_text(stat='count', aes(label=..count..), vjust = -1, size = 2) + 
  theme_classic() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2)) + 
  theme(text = element_text(size = 10), legend.position = "none", axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
  ylim(0, 70) + 
  scale_fill_manual(values = get_colors(6)[2]) + 
  labs(fill='Sex')

# Arrange plot with all age distributions
grid.arrange(arrangeGrob(p1, p2, ncol = 1, nrow = 2, heights = c(2, 0.85)))






# Display length distribution for each sex
ggplot(dataset[which(dataset$sex != "unknown"), ], aes(x = length, fill = sex)) + 
  geom_histogram(aes(y = ..density..), color = 'white', binwidth = 2, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3, show.legend = FALSE, aes(color = sex)) + 
  theme_classic() + 
  xlab('Length (cm)') + 
  ylab('') + 
  scale_x_continuous() + 
  theme(text = element_text(size = 10), legend.position = c(0.85, 0.65)) + 
  labs(fill='Sex')





normalize = function(x) (x - mean(x))/sd(x)


# Discard data with incomplete features so we can fit a model 
# predicting the age using length and sex.
dataset = dataset[which(dataset$sex == 'female'), ]

# Fit Von Bertalenffy model to the data
# Define VBGF loss function
loss = function(params){
  labels = dataset$age
  predictions = params[1] - (1/params[2])*log(1 - dataset$length/params[3])
  return(mean((labels - predictions)**2))
}

# Find model parameters using numerical optimization
params = optimize(loss, c(100, 300))
params
loss(params$minimum)



# Function generator that returns an age prediction function
pred_func = function(params){
  return(function(x){
    return(params[1] - (1/params[2])*log(1 - x/params[3]))
    })
}

# Define an age prediction function for each sex
male_age = pred_func(test$par)
female_age = function(x){
  return(pred_func(params))
}

test = Rcgmin(c(-0.68, 0.05, 122.8), loss, lower = c(-10, 0, 102), upper = c(1, 1, 300))

curve(male_age, from = 0, to = 100)

for(i in seq(0, 100, 1)){
  lines(pred_func(i)(x))
}

curve(pred_func(120)(x), from = 0, to = 100)



anova(lm(age ~ length, dataset))

dataset$length[953]


sum((male_age(dataset$length) - dataset$age)**2)/2073

x = (dataset$age - mean(dataset$age)) / sd(dataset$age)
y = (dataset$length - mean(dataset$length)) / sd(dataset$length)
plot(x, y)

# Set colors for three groups: males, females and unknowns
my_colors = viridis(3, begin = 0.5, end = 0.9)

# Create a temporary dataframe adding a small term to the male length in order
# to make a plot with non-overlapping points.
df.temp = dataset
df.temp$length[df.temp$sex == 'male'] = df.temp$length[df.temp$sex == 'male'] + .5

# Plot age vs length for all three groups
ggplot(df.temp, aes(y = age, x = length)) + 
  geom_count(aes(color = sex), alpha = 0.6) + 
  #scale_color_manual(values = my_colors, labels=c('Females', 'Males', 'Unknowns')) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20)) + 
  labs(color='Sex') + 
  geom_function(fun = female_age)

# Plot age vs length for females
ggplot(subset(dataset, sex == 'female')) + 
  geom_count(aes(y = age, x = length), alpha = 0.5, color = my_colors[1]) + 
  geom_function(fun = male_age) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20)) + 
  xlim(0, 100)
  
ggplot(subset(dataset, sex == 'female')) + 
  geom_function(fun = female_age) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20)) + 
  ylim(0, 26)


# Plot age vs length for males
ggplot(subset(dataset, sex == 'male')) + 
  geom_count(aes(y = age, x = length), alpha = 0.5, color = my_colors[2]) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  ylab('Age') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20))





















