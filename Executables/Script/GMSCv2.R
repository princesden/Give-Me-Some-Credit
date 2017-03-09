# Autor: Princewill Eneh
# Date : October 24, 2015


#INSTALL NEEDED LIBRARIES
install.packages("boot") # Require for cross validation
install.packages("ggplot2") # Ggplot
install.packages("glmnet") # for regularization
install.packages("dplyr") # Require for Normalization
install.packages("tree") #fitting Decision trees
install.packages("caret") #for Confusion matrix
install.packages("rpart") # Support Vector Machine
install.packages("e1071") #SVM
install.packages("kernlab") #svm
install.packages("MASS") #LDA
install.packages("neuralnet") #ANN
install.packages("sampling") #for stratified sampling 
install.packages("ROCR") # ROC
install.packages("randomForest")


library(boot) # Require for cross validation
library(ggplot2) # Ggplot
library(glmnet) # for regularization
library(dplyr) # Require for Normalization
library(tree) #fitting Decision trees
library(caret) #for Confusion matrix
library(rpart) # Support Vector Machine
library("e1071") #SVM
library(kernlab) #svm
library(MASS) #LDA
library(neuralnet) #ANN
library(sampling) #for stratified sampling 
library(ROCR) # ROC
library(randomForest)

#----------------------------------------------------------------------------------------------------------- 
#Import Data
#----------------------------------------------------------------------------------------------------------- 

#train <- read.csv("/Dataset/cs-training.csv") #Point to the location on the dataset
test<-read.csv(file.choose()) #Choose the training file from your computer. The training file 
                              #for this project is in the Dataset folder (/Dataset/cs-training.csv)

#----------------------------------------------------------------------------------------------------------- 
#Explore Date
#----------------------------------------------------------------------------------------------------------- 
str(train)
head(train)
names(train)

#----------------------------------------------------------------------------------------------------------- 
#Data Cleansing Process
#----------------------------------------------------------------------------------------------------------- 

train[1] <- NULL # Remove first column with IDs. we  do not care about this column

#----------------------------------------------------------------------------------------------------------- 
#Imputation
#----------------------------------------------------------------------------------------------------------- 

train<-na.omit(train) # Just remove rows with missing values

#----------------------------------------------------------------------------------------------------------- 
#Stratified Sampling   //Balance our dependent variables
#----------------------------------------------------------------------------------------------------------- 
stratified01 <- subset(train, SeriousDlqin2yrs == "1") #find all 1s
stratified00 <- subset(train, SeriousDlqin2yrs == "0") #find all 0s
stratified02 <-stratified00[1:8357,]

stratified<-rbind(stratified01,stratified02)


#----------------------------------------------------------------------------------------------------------- 
#Shuffle
#----------------------------------------------------------------------------------------------------------- 
train <- stratified[sample(nrow(stratified)),]

rm(stratified01)
rm(stratified00)
rm(stratified02)
rm(stratified)

#----------------------------------------------------------------------------------------------------------- 
#Normalization 
#----------------------------------------------------------------------------------------------------------- 
train<-train %>% mutate_each_(funs(scale),vars=c("RevolvingUtilizationOfUnsecuredLines","age","NumberOfTime30.59DaysPastDueNotWorse",
                                                 "DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
                                                 "NumberRealEstateLoansOrLines","NumberOfTime60.89DaysPastDueNotWorse","NumberOfDependents")) # Here you chose what colums you want to normalize

#-----------------------------------------------------------------------------------------------------------       
#Split into Training and Test Sets.
#----------------------------------------------------------------------------------------------------------- 
set.seed(2)
Train <-sample(1:nrow(train), nrow(train)/2)
Test = -Train
training_data = train[Train,]
testing_data =  train[Test,]
testing_SeriousDlqin2yrs = train$SeriousDlqin2yrs[Test]

#----------------------------------------------------------------------------------------------------------- 
#Check for Missing Data
#----------------------------------------------------------------------------------------------------------- 
sapply(train,function(x) sum(is.na(x))) # See the number of missing values
sapply(train, function(x) length(unique(x)))


#----------------------------------------------------------------------------------------------------------- 
#Regularization
#----------------------------------------------------------------------------------------------------------- 
# 

#----------------------------------------------------------------------------------------------------------- 
#RIDGE Regularization
#----------------------------------------------------------------------------------------------------------- 
x.tr <- model.matrix(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines  + age + NumberOfTime30.59DaysPastDueNotWorse  
                     + DebtRatio  + MonthlyIncome +NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate +  
                       NumberRealEstateLoansOrLines +  NumberOfTime60.89DaysPastDueNotWorse
                     + NumberOfDependents, data = training_data)[, -1]
y.tr <- training_data$SeriousDlqin2yrs


x.val <- model.matrix(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines  + age + NumberOfTime30.59DaysPastDueNotWorse  
                      + DebtRatio  + MonthlyIncome +NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate +  
                        NumberRealEstateLoansOrLines +  NumberOfTime60.89DaysPastDueNotWorse
                      + NumberOfDependents, data = testing_data)[, -1]
y.val <- testing_data$SeriousDlqin2yrs



# CV to obtain best lambda
set.seed(10)
rr.cv <- cv.glmnet(x.tr, y.tr, alpha = 0)
plot(rr.cv)

rr.bestlam <- rr.cv$lambda.min
rr.goodlam <- rr.cv$lambda.1se

# predict validation set using best lambda and calculate RMSE
rr.fit <- glmnet(x.tr, y.tr, alpha = 0)
plot(rr.fit, xvar = "lambda", label = TRUE)


rr.pred <- predict(rr.fit, s = rr.bestlam, newx = x.val)

rr.pred  <- ifelse(rr.pred  > 0.5,1,0)
table(rr.pred ,testing_data[,1 ])

MSE <-mean(rr.pred != testing_SeriousDlqin2yrs)
MSE
print(paste('Accuracy',(1-MSE)*100,"%"))

sqrt(mean((rr.pred - y.val)^2))

#
#----------------------------------------------------------------------------------------------------------- 
#LASSO Regularization
#----------------------------------------------------------------------------------------------------------- 
# CV to obtain best lambda
set.seed(10)
las.cv <- cv.glmnet(x.tr, y.tr, alpha = 1)
plot(las.cv)


las.bestlam <- las.cv$lambda.min
las.goodlam <- las.cv$lambda.1se

# predict validation set using best lambda and calculate RMSE
las.fit <- glmnet(x.tr, y.tr, alpha = 1)
plot(las.fit, xvar = "lambda", label = TRUE)


las.pred <- predict(las.fit, s = las.bestlam, newx = x.val)

las.pred <- ifelse(las.pred > 0.5,1,0)
table(las.pred,testing_data[,1 ])


MSE <-mean(las.pred != testing_SeriousDlqin2yrs)
MSE
print(paste('Accuracy',(1-MSE)*100,"%"))

sqrt(mean((las.pred - y.val)^2))


#----------------------------------------------------------------------------------------------------------- 
#---------LOGISTIC REGRESSION---------#
#----------------------------------------------------------------------------------------------------------- 

glmdl <- glm(SeriousDlqin2yrs ~.,family=binomial(link='logit'),data=training_data)
summary(glmdl)
plot(glmdl)

#LOOCV

cv.glmdl<-cv.glm(glmdl, data = training_data)$delta[1]
cv.glmdl

#KFCV

cv.glmdl<- cv.glm(training_data, glmdl, K=6)$delta[1]
cv.glmdl


glmdlpredict <- predict(glmdl,testing_data,type='response')
glmdlpredict <- ifelse(glmdlpredict > 0.5,1,0)


#------ANALYSIS----
table(glmdlpredict,testing_data[,1 ])
MSE <-mean(glmdlpredict != testing_SeriousDlqin2yrs)
MSE
print(paste('Accuracy',(1-MSE)*100,"%"))

sqrt(mean((glmdlpredict - testing_data$SeriousDlqin2yrs)^2))

#           confMat AUC ROC
confusionMatrix(glmdlpredict, testing_SeriousDlqin2yrs)

glmroc<-prediction(glmdlpredict, testing_SeriousDlqin2yrs, label.ordering = NULL)

glmroc.perf <- performance(glmroc, measure = "tpr", x.measure = "fpr")
plot(glmroc.perf, col = "dark red")
abline(a=0, b= 1)

# AUC 
auc.perf <-performance(glmroc, measure = "auc")
auc.perf@y.values

#----------------------------------------------------------------------------------------------------------- 
#LDA
#----------------------------------------------------------------------------------------------------------- 

ldamdl<-lda(SeriousDlqin2yrs ~., data = training_data)
ldamdl
plot(ldamdl)

ldamdlpredict<-predict(ldamdl, newdata = testing_data[,c(2,3,4,5,6,7,8,9,10,11)])$class

table(ldamdlpredict,testing_data[,1 ])
MSE <-mean(ldamdlpredict != testing_SeriousDlqin2yrs)
MSE
print(paste('Accuracy',(1-MSE)*100,"%"))
#----------------------------------------------------------------------------------------------------------- 
#SVM
#----------------------------------------------------------------------------------------------------------- 
svmmdl <- ksvm(SeriousDlqin2yrs ~ ., data=training_data, type = "C-bsvc", kernel = "rbfdot",kpar = list(sigma = 0.1), C = 10, prob.model = TRUE)

svmmdl

#plot(svmmdl,data = train_dataframe)

svmmdlpredict <- predict(svmmdl,testing_data[,c(2,3,4,5,6,7,8,9,10,11)])
table(svmmdlpredict,testing_data[,1 ])

MSE <-mean(svmmdlpredict != testing_SeriousDlqin2yrs)
MSE
print(paste('Accuracy',(1-MSE)*100,"%"))
#----------------------------------------------------------------------------------------------------------- 
#ANN
#----------------------------------------------------------------------------------------------------------- 
ptm <- proc.time() #Timing Starts (We want to see how fast ANN in on R)


nnmdl <- neuralnet(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines  + age + NumberOfTime30.59DaysPastDueNotWorse  
                      + DebtRatio  + MonthlyIncome +NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate +  
                        NumberRealEstateLoansOrLines +  NumberOfTime60.89DaysPastDueNotWorse
                      + NumberOfDependents, data = training_data, hidden=2, threshold=0.01)

proc.time() - ptm  #Timing stops


plot(nnmdl)

nnmdlpredict <- compute(nnmdl, testing_data[,c(2,3,4,5,6,7,8,9,10,11)])

results <- data.frame(actual = testing_SeriousDlqin2yrs, prediction = nnmdlpredict$net.result)
nnmdlpredict <- round(results$prediction)

table(nnmdlpredict,testing_data[,1 ])

MSE <-mean(nnmdlpredict != testing_SeriousDlqin2yrs)
MSE

print(paste('Accuracy',(1-MSE)*100,"%"))

confusionMatrix(nnmdlpredict, testing_SeriousDlqin2yrs)

nnmdlpredict<- as.numeric(nnmdlpredict)

nnroc<-prediction(nnmdlpredict, testing_SeriousDlqin2yrs, label.ordering = NULL)

nnroc.perf <- performance(nnroc, measure = "tpr", x.measure = "fpr")
plot(nnroc.perf, col = "dark red")
abline(a=0, b= 1)

#AUC
auc.perf <-performance(nnroc, measure = "auc")
auc.perf@y.values

#----------------------------------------------------------------------------------------------------------- 
#RANDOM FOREST
#----------------------------------------------------------------------------------------------------------- 
set.seed(100)
rfmdl <- randomForest(training_data[,-c(1,2,7,12)], factor(training_data$SeriousDlqin2yrs),
                   sampsize=1000, do.trace=TRUE, importance=TRUE, ntree=500, forest=TRUE)
plot(rfmdl)

rfmdlpredict <- data.frame(SeriousDlqin2yrs=predict(rfmdl,testing_data[,-c(1,2,7,12)],type="prob")[,2])
rfmdlpredict <- ifelse(rfmdlpredict > 0.5,1,0)

table(rfmdlpredict,testing_data[,1 ])
MSE <-mean(rfmdlpredict != testing_SeriousDlqin2yrs)
MSE
print(paste('Accuracy',(1-MSE)*100,"%"))

confusionMatrix(rfmdlpredict, testing_SeriousDlqin2yrs)

rfmdlpredict<- as.numeric(rfmdlpredict)

rfroc<-prediction(rfmdlpredict, testing_SeriousDlqin2yrs, label.ordering = NULL)

rfroc.perf <- performance(rfroc, measure = "tpr", x.measure = "fpr")
plot(rfroc.perf, col = "dark red")
abline(a=0, b= 1)

# AUC 
auc.perf <-performance(rfroc, measure = "auc")
auc.perf@y.values




#Plot the AUC of all Algorithms in one graph
plot( glmroc.perf, col = "dark red")
plot(ldaroc.perf, add = TRUE, col = "green")
plot(svmroc.perf, add = TRUE, col = "blue")
plot(nnroc.perf, add = TRUE, col = "black")
grid()
abline(a=0, b= 1)
text(.8,.9,"ANN")
text(.28,.6,"GLM")
text(.15,.6,"SVM")
text(.2,.35,"LDA")
