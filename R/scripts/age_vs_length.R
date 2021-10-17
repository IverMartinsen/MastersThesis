library(ggplot2)
library(viridis)

# Load dataframe
dataset = read.csv(r'(C:\Users\iverm\Desktop\UiT\Data\Grønlandskveiteotolitter\dataframe.csv)')

# Set colors for the two sexes
my_colors = viridis(2, begin = 0.5, end = 0.9)

# Plot age vs length for both groups
ggplot(dataset, aes(y = age, x = length)) + 
  geom_count(aes(color = sex), alpha = 0.4) + 
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

# Discard data with incomplete features
dataset = dataset[rowSums(is.na(dataset)) == 0, ]

# Fit Von Bertalenffy model to the data
loss = function(params){
  labels = dataset$age
  predictions = params[1] + params[2]*log(1 - dataset$length/params[3]) + 
    (params[4]*log(1 - dataset$length/params[5]))*(dataset$sex == 'male')
  return(sum((labels - predictions)**2) / (dim(dataset)[1] - 1 - 5))
}


params = optim(c(1, 1, 10000, 1, 10000), loss)$par

# Function generator
pred_func = function(params, sex){
  return(function(x){
    return(
      params[1] + params[2]*log(1 - x/params[3]) + (params[4]*log(1 - x/params[5]))*(sex == 'male')
    )})}

male_age = pred_func(params, 'male')
female_age = pred_func(params, 'female')
curve(female_age(x), xlim=c(0, 100))
curve(male_age(x), add=TRUE)

model = lm(age ~ length*factor(sex) - factor(sex), dataset)
anova(model)
