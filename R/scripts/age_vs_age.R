library(ggplot2)
library(viridis)

y_hat4 = read.csv(r'(C:\Users\iverm\Desktop\UiT\Grønlandskveiteotolitter\Forberedende forsøk\figures\y_hat4.csv)')
y_te4 = read.csv(r'(C:\Users\iverm\Desktop\UiT\Grønlandskveiteotolitter\Forberedende forsøk\figures\y_te4.csv)')

trial4 = setNames(data.frame(y_te4, y_hat4), c('true_age', 'predicted_age'))

trial6 = read.csv(r'(C:\Users\iverm\Desktop\UiT\Grønlandskveiteotolitter\Forberedende forsøk\test_result_6.csv)')

z = read.csv(r'(C:\Users\iverm\Desktop\UiT\Grønlandskveiteotolitter\Forberedende forsøk\design_matrix.csv)')

ggplot(trial6, aes(x=true_age, y=dl_predictions)) + 
  geom_count(aes(color=abs(true_age-predicted_age))) + 
  scale_color_viridis(option='viridis', guide=FALSE) + 
  geom_abline(slope = 1, intercept = 0, color=viridis(1)) + 
  theme_classic() + 
  xlab('True age') + 
  ylab('Predicted age') + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
  scale_y_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
  scale_size_area(max_size = 16) +
  theme(text = element_text(size = 20))


ggplot(trial6, aes(x = dl_predictions, fill= sex, color = sex)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3) + 
  theme_minimal() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2), limits = c(0, 26)) + 
  scale_y_continuous(limits = c(0, 0.2))
  theme(text = element_text(size = 20))


model = lm(y ~ z1 - 1, data = z)
summary(model)  

cor(z$z1, z$z2)  

hist(z$z2)
