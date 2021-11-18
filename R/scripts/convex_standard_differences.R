library(reshape2)
library(ggplot2)


y1 = read.csv(r'(C:\Users\iverm\OneDrive - UiT Office 365\UiT\Torskeotolitter\Forsøk\Forsøk 13.08.2021\individual_scores.csv)', sep = ';', dec = ',')
y2 = read.csv(r'(C:\Users\iverm\Desktop\Div\individual_scores.csv)', sep = ';', dec = ',')
colnames(y1)[1] = 'filename'
colnames(y2)[1] = 'filename'         

num_samples = length(y1$filename)

# Squeeze individual scores
y1s = data.frame(
  filename = y1$filename,
  'trial1' = rep(0, num_samples),
  'trial2' = rep(0, num_samples),
  'trial3' = rep(0, num_samples),
  'trial4' = rep(0, num_samples),
  ncc = grepl('cc', y1$filename)
)

# Squeeze individual scores
y2s = data.frame(
  filename = y2$filename,
  'trial1' = rep(0, num_samples),
  'trial2' = rep(0, num_samples),
  'trial3' = rep(0, num_samples),
  'trial4' = rep(0, num_samples),
  ncc = grepl('cc', y2$filename)
)

for(i in 1:num_samples){
  y1s[i, 2:5] = y1[i, seq(2, 21)][is.na(y1[i, seq(2, 21)]) == FALSE]
}

for(i in 1:num_samples){
  y2s[i, 2:5] = y2[i, seq(2, 21)][is.na(y2[i, seq(2, 21)]) == FALSE]
}

d = y1s[which(y1s$ncc == TRUE), 2:5] - y2s[which(y2s$ncc == TRUE), 2:5]
d = c(d$trial1, d$trial2, d$trial3, d$trial4)

t = mean(d)*sqrt(num_samples) / sd(d)

1 - pt(t, df = num_samples - 1)


ggplot() + 
  geom_density(aes(melt(y1s)$value - melt(y2s)$value, fill = melt(y1s)$ncc), alpha = 0.5) + 
  theme_classic() + 
  scale_fill_discrete(name = 'Stock', labels = c('NEAC', 'NCC')) + 
  labs(x = 'difference in score') + 
  theme(text = element_text(size = 20))

