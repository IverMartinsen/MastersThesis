# Script for analysis of the Greenland halibut age distribution

library(ggplot2)
library(viridis)
library(dplyr)
library(gridExtra)
library(Rcgmin)

# Load dataframe with all available features arranged by filename
dataset = read.csv(r'(C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith images\Data\Blåkveiteotolitter\dataframe.csv)')

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
length_distribution = ggplot(dataset[which(dataset$sex != "unknown"), ], aes(x = length, fill = sex)) + 
    geom_histogram(aes(y = ..density..), color = 'white', binwidth = 2, alpha = 0.3, position = 'identity') + 
    geom_density(position = 'identity', alpha = 0.3, show.legend = FALSE, aes(color = sex)) + 
    theme_classic() + 
    xlab('Length (cm)') + 
    ylab('') + 
    scale_x_continuous() + 
    theme(text = element_text(size = 10), legend.position = c(0.85, 0.65), axis.text.y = element_blank(), axis.ticks.y = element_blank()) + 
    labs(fill='Sex')



# Define VBGF loss function in order to fit the Von Bertalanffy equation to the data.
loss = function(params){
    labels = subset(dataset, sex != 'unknown')$age
    predictions = (
        params[1] - 
            (1/params[2])*log(1 - subset(dataset, sex != 'unknown')$length/params[3])*(
                subset(dataset, sex != 'unknown')$sex == 'female'
            ) - 
            (1/params[4])*log(1 - subset(dataset, sex != 'unknown')$length/params[5])*(
                subset(dataset, sex != 'unknown')$sex == 'male'
                )
        )
    return(mean((labels - predictions)**2))
}

# Find model parameters using numerical optimization
params = optim(c(0, 0.05, 120, 0.05, 120), loss)$par
loss(params)

# Compute the adjusted R2 to compare with a linear model
n = dim(subset(dataset, sex != 'unknown'))[1]
k = 5
SSE = loss(params)*n
SST = sum((subset(dataset, sex != 'unknown')$age - mean(subset(dataset, sex != 'unknown')$age))**2)

R2.adjusted = 1 - (SSE/(n - k - 1))/(SST/(n - 1))
linear_model = lm(age ~ length*as.factor(sex) - as.factor(sex), subset(dataset, sex != 'unknown'))

# Function generator that returns an age prediction function
pred_func = function(params, sex){
  return(function(x){
    return((
        params[1] + params[2]*(sex == 'male') - 
            (1/params[3])*log(1 - x/params[4]) - 
            (1/params[5])*log(1 - x/params[6])*(sex == 'male')
        ))
    })
}


# Plot age vs length for both sexes
length_scatter = ggplot(subset(dataset, sex != 'unknown'), aes(y = age, x = length)) + 
    geom_count(aes(color = sex), alpha = 0.5, show.legend = FALSE) +
    theme_classic() + 
    scale_y_continuous(breaks = seq(0, 26, 2)) + 
    xlab('Length (cm)') + 
    ylab('Age') + 
    scale_size_area(max_size = 10) + 
    theme(text = element_text(size = 10), legend.position = c(1.00, 0.35)) + 
    labs(color='Sex') + 
    stat_smooth(method = 'lm', data = subset(dataset, sex == 'male'), color = 'black', linetype = 3) + 
    stat_smooth(method = 'lm', data = subset(dataset, sex == 'female'), color = 'black', linetype = 2)


# Combine length distribution and scatter plot in the same plot
grid.arrange(length_distribution, length_scatter, ncol = 2)


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





















