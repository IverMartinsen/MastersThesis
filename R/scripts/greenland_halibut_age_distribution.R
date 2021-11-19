library(ggplot2)
library(viridis)

# Load dataframe
dataset = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Data\Grønlandskveiteotolitter\dataframe.csv)')

# Set colors for the two sexes
my_colors = viridis(2, begin = 0.5, end = 0.9)

dataset$length[dataset$sex == 'male'] = dataset$length[dataset$sex == 'male'] + .5

# Plot age vs length for both groups
ggplot(dataset, aes(y = age, x = length)) + 
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
  theme(text = element_text(size = 20)) #+ 
#geom_function(fun = male_age) + 
#geom_line(aes(x = length, y = model$fitted.values[dataset$sex == 'male']))

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

plot(male_age(dataset$length), male_age(dataset$length) - dataset$age)


male_age = pred_func(params, 'male')
female_age = pred_func(params, 'female')
curve(female_age(x), xlim=c(0, 100))
curve(male_age(x), add=TRUE)

model = lm(age ~ length*factor(sex) - factor(sex), dataset)
anova(model)


plot(model$fitted.values, model$residuals)


df = data.frame(
  value = c(model$residuals[dataset$sex == 'male'], dataset$age[dataset$sex == 'male'] - male_age(dataset$length[dataset$sex == 'male'])),
  group = c(rep('linear', sum(dataset$sex == 'male')), rep('nonlinear', sum(dataset$sex == 'male'))),
  fitted = c(model$fitted.values[dataset$sex == 'male'], male_age(dataset$length[dataset$sex == 'male']))
)

ggplot(df, aes(x = fitted, y = value, colour = group)) + 
  geom_point()

curve(male_age, xlim = c(0, 100), ylim = c(0, 26))
l(dataset$length, model$fitted.values)


library(ggplot2)


df = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Data\Grønlandskveiteotolitter\dataframe.csv)')

ggplot(df[rowSums(is.na(df)) == 0, ], aes(x = age, fill = sex)) + 
  geom_bar(aes(y = ..count..), binwidth = 1, alpha = 0.6, position = 'identity') + 
  geom_text(stat='count', aes(label=..count..), vjust = -1) + 
  #geom_density(position = 'identity', alpha = 0.3, show.legend = FALSE, aes(color = sex)) + 
  theme_classic() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2)) + 
  theme(text = element_text(size = 20)) + 
  labs(fill='Sex')


ggplot(df[rowSums(is.na(df)) == 0, ], aes(x = length, fill = sex)) + 
  geom_histogram(aes(y = ..density..), binwidth = 2, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3, show.legend = FALSE, aes(color = sex)) + 
  theme_classic() + 
  xlab('Length (cm)') + 
  ylab('') + 
  scale_x_continuous() + 
  theme(text = element_text(size = 20)) + 
  labs(fill='Sex')

sd(df[df$sex == 'male', ]$age)

shapiro.test(df[df$sex == 'male', ]$age)

ks.test(df[df$sex == 'female', ]$age, df[df$sex == 'male', ]$age)

qqnorm(df[df$sex == 'female', ]$age)


plot(log(df[df$sex == 'male', ]$length), log(df[df$sex == 'male', ]$age))

model = lm(age ~ length*factor(sex), df)
plot(model$residuals)
qqnorm(model$residuals)
