library(caret)
library(e1071)

setwd("C:/Users/kaspe/python-projects/intensiclassifier/previous_runs")
data = read.csv('1kfeatures_times10_mutinfo_2kimdbmovies_predictions.csv')

reference <- data$correct
clfs <- data[,2:10]
for(clf in clfs){
  cm <- confusionMatrix(
    factor(clf), factor(reference)
    )
}
