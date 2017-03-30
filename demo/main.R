library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(caret)
library(fastknn)
library(ranger)

## Load data
adult <- read_csv("demo/adult.csv.zip", na = "?")
glimpse(adult)
table(adult$income)

## Select variables
var.names <- c("age", "workclass", "education", "marital.status", "occupation",
               "relationship", "race", "sex", "hours.per.week", 
               "native.country", "income")
dataset <- select(adult, one_of(var.names))

## Convert string to categorical
chr.vars <- which(map_lgl(dataset, is.character))
dataset <- mutate_each(dataset, funs(as.factor), chr.vars)

## Remove or impute NAs
# Impute by median or mode
na.fill <- map(dataset, function(column) {
   if (is.numeric(column)) {
      return(median(column))
   } else {
      column.tbl <- table(column)
      return(names(column.tbl)[which.max(column.tbl)])
   }
})
dataset <- replace_na(dataset, replace = na.fill)
# Remove NAs
dataset <- na.omit(dataset)

## Exploratory data analysis
# Caculate the classification performance of each hypothesis
# H1: People who are older earn more
# H2: Income bias based on working class
# H3: People with more education earn more
# H4: Maried people tend to earn more
# H5: There is a bias in income based on race
# H6: There is a bias in the income based on occupation
# H7: Men earn more
table(dataset$sex, dataset$income) %>% prop.table(2)
# H8: People who work in more hours earn more
# H9: There is a bias in income based on the country of origin

## Machine learning
# Split data: 70% for training
set.seed(1024)
tr.idx <- createDataPartition(dataset$income, p = 0.7, list = FALSE)
# Fit a decision tree
tree.model <- train(form = income ~ age + education, data = dataset[tr.idx,], 
                    method = "rpart")
# Predict test set
tree.pred <- predict(tree.model, newdata = dataset[-tr.idx,])
# Performance
classAccuracy <- function(y, yhat) {
   sum(yhat == y) / length(y)
}
classAccuracy(dataset$income[-tr.idx], tree.pred)

## Encode categorical features
x <- model.matrix(income ~ . -1, data = dataset)
dim(x)
y <- dataset$income

## Fit KNN
# Dist method
knn.model <- fastknn(x[tr.idx,], y[tr.idx], x[-tr.idx,], k = 25)
classAccuracy(dataset$income[-tr.idx], knn.model$class)
# Vote method
knn.model <- fastknn(x[tr.idx,], y[tr.idx], x[-tr.idx,], k = 25, 
                     method = "vote")
classAccuracy(dataset$income[-tr.idx], knn.model$class)
# Vote method with normalization
knn.model <- fastknn(x[tr.idx,], y[tr.idx], x[-tr.idx,], k = 25, 
                     method = "vote", normalize = "robust")
classAccuracy(dataset$income[-tr.idx], knn.model$class)
# Find the best k
set.seed(2048)
cv.out <- fastknnCV(x[tr.idx,], y[tr.idx], k = seq(from = 10, to = 30, by = 5), 
                    method = "vote", normalize = "robust", 
                    folds = 10, nthread = 8)
cv.out$cv_table
cv.out$best_k

## Fit Random Forest
rf.model <- ranger(income ~ ., dataset[tr.idx,], num.trees = 500, mtry = 2, 
                   num.threads = 8)
rf.pred <- predict(rf.model, dataset[-tr.idx,])
classAccuracy(dataset$income[-tr.idx], rf.pred$predictions)
