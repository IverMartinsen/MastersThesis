library(ggplot2)
library(reshape2)
library(viridis)

df = read.csv(r'(C:\Users\iverm\Desktop\UiT\Grønlandskveiteotolitter\Forberedende forsøk\test_result_8.csv)')
#df = melt(df[, c('filenames', 'true_age', 'pred_age_deep', 'sex')])

colours = c('black', "#35B779FF", "#FDE725FF", 'black')
         
ggplot(df, aes(x = value, fill = interaction(sex, variable))) + 
  geom_density(position = 'identity', alpha = 0.5, show.legend = TRUE) + 
  #stat_density(position = 'identity', geom = 'line', lwd = 0) +
  theme_classic() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2)) + 
  theme(text = element_text(size = 20)) + 
  labs(fill = '') + 
  scale_fill_manual(values = colours, labels = c(
    "True age - females", "True age - males", "Predicted age - females", "Predicted age - males"))


ggplot(df, aes(x=true_age, y=pred_age_deep)) + 
  geom_count(aes(color=abs(true_age-pred_age_deep))) + 
  scale_color_viridis(option='viridis', guide=FALSE) + 
  geom_abline(slope = 1, intercept = 0, color=viridis(1)) + 
  theme_classic() + 
  xlab('True age') + 
  ylab('Predicted age') + 
  scale_x_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) +
  scale_y_continuous(breaks=seq(0, 26, by=2), limits=c(0, 26)) + 
  scale_size_area(max_size = 16) +
  theme(text = element_text(size = 20))


ggplot(df, aes(x = pred_age_deep, fill= sex, color = sex)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, alpha = 0.3, position = 'identity') + 
  geom_density(position = 'identity', alpha = 0.3) + 
  theme_minimal() + 
  xlab('Age') + 
  ylab('') + 
  scale_x_continuous(breaks = seq(0, 26, 2), limits = c(0, 26)) + 
  scale_y_continuous(limits = c(0, 0.2))
theme(text = element_text(size = 20))



length_df = read.csv(r'(C:\Users\iverm\Desktop\UiT\Data\Grønlandskveiteotolitter\dataframe.csv)')

modell = lm(true_age ~ pred_age_deep + length:sex, df)

length = rep(0, length(df$filenames))
i = 1
for(filename in df$filenames){
  length[i] = length_df[which(length_df == filename, arr.ind = TRUE)[1], 3]
  i = i + 1
}
