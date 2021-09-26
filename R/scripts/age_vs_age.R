library(ggplot2)
library(viridis)

y_te = read.csv(r'(C:\Users\iverm\Google Drive\Forberedende forsøk\y_te4.csv)')
y_hat = read.csv(r'(C:\Users\iverm\Google Drive\Forberedende forsøk\y_hat4.csv)')
         
ggplot(setNames(cbind(y_te, y_hat), c('y_te', 'y_hat')), aes(x=y_te, y=y_hat)) + 
  geom_count(aes(color=abs(y_te-y_hat))) + 
  scale_color_viridis(option='viridis', guide=FALSE) + 
  geom_abline(slope = 1, intercept = 0, color=viridis(1)) + 
  theme_minimal() + 
  xlab('True age') + 
  ylab('Predicted age') + 
  scale_x_continuous(breaks=seq(0, 26, by=2)) +
  scale_y_continuous(breaks=seq(0, 26, by=2)) + 
  scale_size_area(max_size = 16) +
  theme(text = element_text(size = 20))




