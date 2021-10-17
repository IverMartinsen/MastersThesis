library(ggplot2)


df = read.csv(r'(C:\Users\iverm\Desktop\UiT\Data\Grønlandskveiteotolitter\dataframe.csv)')

ggplot(df[rowSums(is.na(df)) == 0, ], aes(x = age, fill = sex)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3, show.legend = FALSE, aes(color = sex)) + 
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

var(df[df$sex == 'female', ]$age)

shapiro.test(df[df$sex == 'male', ]$age)

ks.test(df[df$sex == 'female', ]$age, df[df$sex == 'male', ]$age)

qqnorm(df[df$sex == 'female', ]$age)


plot(log(df[df$sex == 'male', ]$length), log(df[df$sex == 'male', ]$age))

model = lm(age ~ length*factor(sex), df)
plot(model$residuals)
qqnorm(model$residuals)
