library(ggplot2)

# import ages
age.f = read.csv(r'(C:\Users\iverm\Google Drive\Data\Grønlandskveiteotolitter\agef.csv)')
age.m = read.csv(r'(C:\Users\iverm\Google Drive\Data\Grønlandskveiteotolitter\agem.csv)')

# stack ages by sex factor
df = rbind(
  setNames(cbind(age.f, 'Females'), c('age', 'Sex')),
  setNames(cbind(age.m, 'Males'), c('age', 'Sex'))
  )

ggplot(df, aes(x = age, fill= Sex, color = Sex)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3) + 
  theme_minimal() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2)) + 
  theme(text = element_text(size = 20))


