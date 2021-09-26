library(ggplot2)
library(viridis)

age.m = read.csv(r'(C:\Users\iverm\Google Drive\Data\Grønlandskveiteotolitter\agem.csv)')
age.f = read.csv(r'(C:\Users\iverm\Google Drive\Data\Grønlandskveiteotolitter\agef.csv)')
len.m = read.csv(r'(C:\Users\iverm\Google Drive\Data\Grønlandskveiteotolitter\lm.csv)')
len.f = read.csv(r'(C:\Users\iverm\Google Drive\Data\Grønlandskveiteotolitter\lf.csv)')

data.m = setNames(cbind(age.m, len.m, Sex = 'Males'), c('Age', 'Length', 'Sex'))
data.f = setNames(cbind(age.f, len.f, sex = 'Females'), c('Age', 'Length', 'Sex'))

dataset = rbind(data.m, data.f)

my_colors = viridis(2, begin = 0.5, end = 0.9)


ggplot(dataset, aes(y = Age, x = Length)) + 
  geom_count(aes(color = Sex), alpha = 0.4) + 
  scale_color_manual(values = my_colors) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  scale_size_area(max_size = 16) + 
  theme(
    text = element_text(size = 20))

ggplot(subset(dataset, Sex == 'Females')) + 
  geom_count(aes(y = Age, x = Length, alpha = 0.2), color = my_colors[1]) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20))

ggplot(subset(dataset, Sex == 'Males')) + 
  geom_count(aes(y = Age, x = Length, alpha = 0.2), color = my_colors[2]) + 
  theme_classic() + 
  scale_y_continuous(breaks = seq(0, 26, 2)) + 
  xlab('Length (cm)') + 
  scale_size_area(max_size = 16) + 
  theme(text = element_text(size = 20))

p.male = rep(0, 26)
for(i in 1:26){
  num_males = sum((dataset['Age'] == i) * (dataset['Sex'] == 'Males'))
  num_females = sum((dataset['Age'] == i) * (dataset['Sex'] == 'Females'))
  p.male[i] = num_males / (num_males + num_females)
}


mse = function(y){
  inv.log = function(x){
    return(0.5 / (y[3]*exp(y[1]*(x - y[2]))))
  }  
  return(sum((p.male - inv.log(1:26))^2))
}

help(optimize)
optim(c(1, 1, 1), mse)

curve(dgamma(x, 18.89, 2.31), from = 0, to = 26)

inv.log = function(x){
  return(0.5 / (0.9*exp(0.06*(x - 1.07))))
}

curve(inv.log(x), from = 0, to = 26, ylim = c(0, 1))
