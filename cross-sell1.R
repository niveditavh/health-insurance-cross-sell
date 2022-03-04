ins_data <- read.csv("E:/DMML_Datasets/cross-sell/train.csv")

head(ins_data)
str(ins_data)

nrow(ins_data)



######checking null######
sum(is.na(ins_data))

#install.packages("tidyverse")

library(tidyverse)
#install.packages("sqldf")

library(dplyr)


# mean of vehicle damage for both it is similar ######
mean_AP <- ins_data %>% group_by(Vehicle_Damage) %>% summarise(mean = mean(Annual_Premium))
mean_AP

library(ggthemes)


#### response with gender #######

#male has majority tend to buy insurance
ins_data_gender<-table(ins_data$Response,ins_data$Gender)
barplot(ins_data_gender, xlab='gender',ylab='Count',main="Response with gender",
        col=c("cyan4","coral")
        ,legend=rownames(ins_data_gender), args.legend = list(x = "topleft"),beside=T)


####### response with driving licence ######
##customers with driving license opt for insurance
ins_data_driving_licence<-table(ins_data$Response,ins_data$Driving_License)
barplot(ins_data_driving_licence, xlab='Driving Licence',ylab='Count',main="Response with driving licence",
        col=c("cyan4","coral")
        ,legend=rownames(ins_data_gender), args.legend = list(x = "topleft"),beside=T)


##### response with previously insured ##########

#this depicts that the customers want to have only an insurance policy if not previously insured. 
#It means those who already have insurance won't convert

ins_data_driving_prev_insured<-table(ins_data$Response,ins_data$Previously_Insured)
barplot(ins_data_driving_prev_insured, xlab='Previous Insurance',ylab='Count',main="Response with Previous insurance",
        col=c("cyan4","coral")
        ,legend=rownames(ins_data_driving_prev_insured), args.legend = list(x = "topleft"),beside=T)


######## response with vehicle age ######

#the customers with vehicle age lesser than the 2 years are more tend to buy insurance.

ins_data_vehicle_age<-table(ins_data$Response,ins_data$Vehicle_Age)
barplot(ins_data_vehicle_age, xlab='Vehicle Age',ylab='Count',main="Response with vehicle age",
        col=c("cyan4","coral")
        ,legend=rownames(ins_data_vehicle_age), args.legend = list(x = "topleft"),beside=T)



######### response with vehicle damage #########


#we can infer that if the vehicle has been damaged previously then the customer 
#will be more interested in buying the insurance as they know the cost.

ins_data_vehicle_damage<-table(ins_data$Response,ins_data$Vehicle_Damage)
barplot(ins_data_vehicle_damage, xlab='Vehicle Damage',ylab='Count',main="Response with vehicle Damage",
        col=c("cyan4","coral")
        ,legend=rownames(ins_data_vehicle_damage), args.legend = list(x = "topleft"),beside=T)


########### response with age #######

#People aged between 30-60 are more likely to be interested.
ins_data_age_response<-table(ins_data$Response,ins_data$Age)
barplot(ins_data_age_response, xlab='Age',ylab='Count',main="Response",
        col=c("cyan4","coral")
        ,legend=rownames(ins_data_age_response), args.legend = list(x = "topleft"),beside=T)



############################# target variable response with count ##########

# The given problem is an imbalance problem as the Response variable with the value 1 is significantly lower than the value zero.

counts <- table(ins_data$Response)
barplot(counts, main="Target Variable visualization",ylab="Count",
        xlab="Default payment",col=c("goldenrod4","firebrick"))

## 0->87% 1->15%
prop.table(table(ins_train$Response))


######### Label encoder used to convert to numeric ###########

install.packages('superml')

library(superml)


label <- LabelEncoder$new()


ins_data$Gender <- label$fit_transform(ins_data$Gender)
print(ins_data$Gender)

ins_data$Vehicle_Age <- label$fit_transform(ins_data$Vehicle_Age)
print(ins_data$Vehicle_Age)

ins_data$Vehicle_Damage <- label$fit_transform(ins_data$Vehicle_Damage)
print(ins_data$Vehicle_Damage)


#ins_data$Previously_Insured <- label$fit_transform(ins_data$Previously_Insured)
#print(ins_data$Previously_Insured)


#ins_data_copy <- sqldf("select * from ins_data")

#Control Structures
#ins_data_copy$Gender <- ifelse(ins_data_copy$Gender == "Male", 1, 0)
#ins_data_copy$Vehicle_Damage <- ifelse(ins_data_copy$Vehicle_Damage == "Yes", 1, 0)
#ins_data_copy$Vehicle_Age <- ifelse(ins_data_copy$Vehicle_Age == "> 2 Years", 2, ifelse(ins_data_copy$Vehicle_Age == "1-2 Year", 1, 0))
#str(ins_data_copy)


str(ins_data)

library(corrplot)

#corrplot(cor(ins_data), method="number", type="lower")

corrplot(cor(ins_data),method = "number")


##### splitting the data ######

set.seed(123)
train_idx <- sample(nrow(ins_data), .70*nrow(ins_data))

ins_train <- ins_data[train_idx,]
ins_test <- ins_data[-train_idx,]


##############################################################################################################################

############ before applying random sampling for unbalanced data #######


ins_logreg <- glm(Response ~.,family=binomial,data = ins_train)
summary(ins_logreg)


###### removing non significant features from above summary #######
#id,vintage,region code removed


ins_logreg_final <- glm(Response ~ Gender+Age+Driving_License+Previously_Insured
                        +Vehicle_Age+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel,family=binomial,data = ins_train,na.action=na.exclude)
summary(ins_logreg_final)

####predicting probability #######
testing_ins <- predict(ins_logreg_final, ins_test, type = "response")
hist(testing_ins)

#from histogram no predicted probability threshold is >0.5 
#For this case study, we will predict that any individual with a predicted response probability greater than 0.25 is predicted as a buyer.




########confusion matrix ######
Y_hat_mod2 <- as.numeric(testing_ins > 0.25)
library(caret)
confusionMatrix(as.factor(ins_test$Response),as.factor(Y_hat_mod2),dnn = c("Actual", "Predicted"))



#accuracy - 79%
#sensitivity - 93%
#specificity - 32%
#kappa - 30%
#balanced accuracy - 62%
#auc - 83.2%
###### ROC-AUC to set threshold previous we set as 0.25 better to have ROC-AUC to make decision #####
library(pROC)

##ROC-AUC for logistics###

par(pty ="s")
auc_lm <- roc(ins_test$Response,testing_ins, plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)





########### applying random forest for unbalanced data##########

######### it takes 20mins to run #################
library(stats)
library(randomForest)



RF_Model = randomForest(Response~Gender+Age+Driving_License+Previously_Insured
                        +Vehicle_Age+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel,family=binomial,
                        data=ins_train,ntree=100,na.action=na.exclude)
plot(RF_Model)
#predicted_response = 
predicted_response<-predict(RF_Model,ins_test)

predicted_response
### maximum value was found >0.25 so taking that ######
rf_mod<- as.numeric(predicted_response > 0.25)


##########confusion matric for random forest #########

library(caret)
confusionMatrix(as.factor(ins_test$Response),as.factor(rf_mod),dnn = c("Actual", "Predicted"))



#accuracy - 78%
#sensitivity - 95%
#specificity - 32%
#kappa - 34%
#balanced accuracy - 64%
#auc -85.5%



##ROC-AUC for RF####

library(pROC)
auc_rf <- roc(as.numeric(ins_test$Response),as.numeric(predicted_response), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#4daf4a", lwd=4, print.auc = TRUE)



##################  naive bayes #################




ins_train$Response<-as.factor(ins_train$Response)
class(ins_train$Response)


library(e1071)
library(caTools)
library(caret)
model_nb_before_sm<-naiveBayes(Response ~ Gender+Age+Driving_License+Previously_Insured
                               +Vehicle_Age+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel,family=binomial,
                                data=ins_train)

pred_test_naive_before_sm<-predict(model_nb_before_sm,ins_test)
library(caret)
confusionMatrix(as.factor(ins_test$Response),as.factor(pred_test_naive_before_sm))

#accuracy - 63%
#sensitivity - 99%
#specificity - 25%
#kappa - 25%
#blanced accuracy - 62%
#auc - 78.4%

library(pROC)
par(pty ="s")
auc_nb <- roc(as.numeric(ins_test$Response),as.numeric(pred_test_naive_before_sm), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)






##########applying smote for imbalance response data #####
####synthetic minority oversampling

library('smotefamily')
#library('DMwR')
set.seed(500)
trainSplit <- SMOTE(Response ~ .)
prop.table(table(trainSplit$Response))
(trainSplit)



set.seed(123)
train_idx <- sample(nrow(trainSplit), .70*nrow(trainSplit))
str(train_idx)

ins_train_sm <- trainSplit[train_idx,]
ins_test_sm <- trainSplit[-train_idx,]
ins_logreg_smote <- glm(Response ~ .,
                        family=binomial,data = ins_train_sm)
summary(ins_logreg_smote)



testing_smote <- predict(ins_logreg_smote, ins_test_sm, type = "response")
hist(testing_smote)

summary(ins_logreg_smote)
install.packages('caret')
library(caret)
library(mlbench)
library(caret)
importance<- varImp(ins_logreg_smote,scale=FALSE)
print(importance)
plot(importance)


ins_logreg_smote_feature <- glm(Response ~ Age+Previously_Insured+Vehicle_Age+Vehicle_Damage,
                        family=binomial,data = ins_train_sm)
summary(ins_logreg_smote)
testing_smote_feature <- predict(ins_logreg_smote_feature, ins_test_sm, type = "response")
hist(testing_smote_feature)

#To check the probability threshold using which we can classify our data, we will create a prediction using type = "response",
#which will create the probabilities instead of actual predicted values. Then we create a histogram and check which threshold we can take. 
#As observed from the below histogram, we have taken 0.5 as the probability threshold.



########confusion matrix ######
smote_mod2 <- as.numeric(testing_smote > 0.5)
library(caret)
confusionMatrix(as.factor(ins_test_sm$Response),as.factor(smote_mod2),dnn = c("Actual", "Predicted"))

smote_mod2_feature <- as.numeric(testing_smote_feature > 0.5)
library(caret)
confusionMatrix(as.factor(ins_test_sm$Response),as.factor(smote_mod2_feature),dnn = c("Actual", "Predicted"))

#accuracy - 78%
#sensitivity - 96%
#specificity - 70%
#kappa - 57%
#balanced accuracy - 83%
#roc-auc - 83.3%

##ROC-AUC after smote for logistics###

par(pty ="s")
auc_lm <- roc(ins_test_sm$Response,testing_smote, plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)

par(pty ="s")
auc_lm <- roc(ins_test_sm$Response,testing_smote_feature, plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)



#####  random forest after smote  ######



library(stats)
library(randomForest)
memory.limit() 
memory.limit(24000)

RF_Model_smote = randomForest(Response~Gender+Age+Driving_License+Previously_Insured
                        +Vehicle_Age+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel,family=binomial,
                        data=ins_train_sm,ntree=100,na.action=na.exclude)
#predicted_response = 
predicted_response_rf_sm<-predict(RF_Model_smote,ins_test_sm)


##########confusion matric for random forest #########

library(caret)
confusionMatrix(as.factor(ins_test_sm$Response),as.factor(predicted_response_rf_sm),dnn = c("Actual", "Predicted"))

#accuracy - 79%
#sensitivity - 90%
#specificity - 73%
#kappa - 59%
#balanced accuracy - 82%
#auc - 79.7%



##ROC-AUC for RF####


auc_rf <- roc(as.numeric(ins_test_sm$Response),as.numeric(predicted_response_rf_sm), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#4daf4a", lwd=4, print.auc = TRUE)






############### naive bayes smote #############


ins_train_sm$Response<-as.factor(ins_train_sm$Response)
class(ins_train_sm$Response)


library(e1071)
library(caTools)
library(caret)
model_nb<-naiveBayes(Response ~ Gender+Age+Driving_License+Previously_Insured
                     +Vehicle_Age+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel,family=binomial,
                     data=ins_train_sm)

pred_test_naive<-predict(model_nb,ins_test_sm)
library(caret)
confusionMatrix(as.factor(ins_test_sm$Response),as.factor(pred_test_naive))

#accuracy - 78%
#sensitivity - 96%
#specificity - 70%
#kappa - 57%
#blanced accuracy - 83%
#auc - 78.5%

library(pROC)
par(pty ="s")
auc_nb <- roc(as.numeric(ins_test_sm$Response),as.numeric(pred_test_naive), plot = TRUE, legacy.axes= TRUE, 
              percent = TRUE, xlab = "False Positive %", ylab = "True Positive %", 
              col="#377eb8", lwd=4, print.auc = TRUE)



################################# conclusion ####################################

#before smote
#accuracy roc-auc
#Logistic -79% 83.2%
#Random Forest- 78% 85.5%
#Naïve Bayes - 63% 78.4%

#after smote

#Logistic 78% 83.3%
#Random Forest 79% 79.7%
#Naïve Bayes 78% 78.5%
#We have considered the Accuracy an ROc-AUC because ROC_AUC has ability to differentiate between classes ,so by this 
#insurance company can target customers for advertisements to buy vehicle insurance
############ Future work ##############


#take different threshold values
#other sampling techniques can be used - under,over and compare







































################### not used k-fold validation ###########################

library(tidyverse)

library(caret)


# setting seed to generate a 
# reproducible random sampling
set.seed(123)

# define training control which
# generates parameters that further
# control how models are created
train_control <- trainControl(method = "cv",
                              number = 10)


# building the model and
# predicting the target variable
# as per the Naive Bayes classifier
model <- train(Response ~ Gender+Age+Driving_License+Previously_Insured
               +Vehicle_Age+Vehicle_Damage+Annual_Premium+Policy_Sales_Channel,family=binomial,
               data=ins_train_sm,
               trControl = train_control,
               method = "nb")
print(model)

