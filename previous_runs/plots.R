setwd("C:/Users/kaspe/python-projects/intensiclassifier/previous_runs")
data = read.csv(choose.files())

reference <- data$correct[0:500]
clfs <- data
clf <- clfs$Colored.B..Multinomial[0:500]

# make the confusion matrix
confusion_matrix <- as.data.frame(table(clf, reference))

# plot it
ggplot(data = confusion_matrix,
       mapping = aes(x = reference,
                     y = clf)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white",
                      high = "darkblue",
                      limit = c(0,
                                (length(reference) /
                                   length(unique(reference))
                                 )
                                )
  ) +
  ylab("Prediction") +
  xlab('Correct class')
    
count = 0
for (i in 1:500){
  if (clf[i] == reference[i]){
    count = count + 1
  }
}
