rm(list = ls())
library(simpleNeural)
library(randomForest)
library(cutpointr)
options(warn = -1)
load("TrainTestDataset.RData")
for(analysis in names(TrainTestDataset)){
  dataFrameALL<-TrainTestDataset[[analysis]]
  ###########Let us summarize Train dataset
  dataFrame<-dataFrameALL[dataFrameALL$statusOrigin=="Train",]
  dataGLM<-data.frame(subject=unique(dataFrame$sampleID))
  rownames(dataGLM)<-dataGLM$subject
  for(sub in rownames(dataGLM)){
    dataGLM[sub, "Bmean"]<-mean(dataFrame$BscoreScale[dataFrame$sampleID==sub])
    dataGLM[sub, "Dmean"]<-mean(dataFrame$DscoreScale[dataFrame$sampleID==sub])
    dataGLM[sub, "Nmean"]<-mean(dataFrame$NscoreScale[dataFrame$sampleID==sub])
    dataGLM[sub, "Pmean"]<-mean(dataFrame$PscoreScale[dataFrame$sampleID==sub])
    dataGLM[sub, "Vmean"]<-mean(dataFrame$VscoreScale[dataFrame$sampleID==sub])
    dataGLM[sub, "cellCount"]<-dim(dataFrame[dataFrame$sampleID==sub,])[1]
  }
  dataGLM$statusname<-"CTRL"
  dataGLM$statusname[grepl("vexas", dataGLM$subject)]<-"VEXAS"
  ###let us plot
  dataGLM$status<-0
  dataGLM$status[grepl("vexas", dataGLM$subject)]<-1
  ##########Let us summarize Test dataset
  dataFrameTest<-dataFrameALL[dataFrameALL$statusOrigin=="Test",]
  dataGLMTest<-data.frame(subject=unique(dataFrameTest$sampleID))
  rownames(dataGLMTest)<-dataGLMTest$subject
  for(sub in rownames(dataGLMTest)){
    dataGLMTest[sub, "Bmean"]<-mean(dataFrameTest$BscoreScale[dataFrameTest$sampleID==sub])
    dataGLMTest[sub, "Dmean"]<-mean(dataFrameTest$DscoreScale[dataFrameTest$sampleID==sub])
    dataGLMTest[sub, "Nmean"]<-mean(dataFrameTest$NscoreScale[dataFrameTest$sampleID==sub])
    dataGLMTest[sub, "Pmean"]<-mean(dataFrameTest$PscoreScale[dataFrameTest$sampleID==sub])
    dataGLMTest[sub, "Vmean"]<-mean(dataFrameTest$VscoreScale[dataFrameTest$sampleID==sub])
    dataGLMTest[sub, "cellCount"]<-dim(dataFrameTest[dataFrameTest$sampleID==sub,])[1]
  }
  dataGLMTest$statusname<-"CTRL"
  dataGLMTest$statusname[grepl("vexas", dataGLMTest$subject)]<-"VEXAS"
  ###let us plot
  dataGLMTest$status<-0
  dataGLMTest$status[grepl("vexas", dataGLMTest$subject)]<-1
  ###MLP data
  X=as.matrix(sN.normalizeDF(as.data.frame(dataGLM[,c("Bmean","Dmean","Pmean","Vmean")])));
  y=as.matrix(dataGLM[, "status"]);
  myMLP=sN.MLPtrain(X=X,y=y,hidden_layer_size=3,it=50,lambda=0.5,alpha=0.5);
  myPrediction=sN.MLPpredict(nnModel=myMLP,X=X,raw=TRUE);
  predictions_original <- myPrediction[,2]
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLM$status==1]<-"YES"
  cp <- cutpointr(dataROC, dataOut,labels, method = maximize_metric, metric = sum_sens_spec, silent = TRUE)
  #roc(dataROC$labels, dataROC$dataOut, plot=T)
  cat(analysis);cat("\t");cat(as.numeric(cp$AUC));cat("\t");cat(as.numeric(cp$acc));cat("\t");cat(as.numeric(cp$specificity));cat("\t")
  TP<-sum(dataROC$labels=="YES" & predictions_original>=cp$optimal_cutpoint)
  TN<-sum(dataROC$labels=="NO" & predictions_original<=cp$optimal_cutpoint)
  FP<-sum(dataROC$labels=="YES" & predictions_original<cp$optimal_cutpoint)
  FN<-sum(dataROC$labels=="NO" & predictions_original>cp$optimal_cutpoint)
  cat(precision(TP,FP,TN,FN));cat("\t")
  cat(F1_score(TP,FP,TN,FN));cat("\t")
  cat(recall(TP,FP,TN,FN));cat("\t")
  
  Xtest<-as.matrix(dataGLMTest[, c("Bmean","Dmean","Pmean","Vmean")])
  Xtest[,1]<-(Xtest[,1]-min(dataGLM[,"Bmean"]))/(max(dataGLM[,"Bmean"])-min(dataGLM[,"Bmean"]))
  Xtest[,2]<-(Xtest[,2]-min(dataGLM[,"Dmean"]))/(max(dataGLM[,"Dmean"])-min(dataGLM[,"Dmean"]))
  Xtest[,3]<-(Xtest[,3]-min(dataGLM[,"Pmean"]))/(max(dataGLM[,"Pmean"])-min(dataGLM[,"Pmean"]))
  Xtest[,4]<-(Xtest[,4]-min(dataGLM[,"Vmean"]))/(max(dataGLM[,"Vmean"])-min(dataGLM[,"Vmean"]))
  myPrediction=sN.MLPpredict(nnModel=myMLP,X=Xtest,raw=TRUE);
  predictions_original <- myPrediction[,2]
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLMTest$status==1]<-"YES"
  cp <- cutpointr(dataROC, dataOut,labels, method = maximize_metric, metric = sum_sens_spec, silent = TRUE)
  cat(as.numeric(cp$AUC));cat("\t");cat(as.numeric(cp$acc));cat("\t");cat(as.numeric(cp$specificity));cat("\t")
  TP<-sum(dataROC$labels=="YES" & predictions_original>=cp$optimal_cutpoint)
  TN<-sum(dataROC$labels=="NO" & predictions_original<=cp$optimal_cutpoint)
  FP<-sum(dataROC$labels=="YES" & predictions_original<cp$optimal_cutpoint)
  FN<-sum(dataROC$labels=="NO" & predictions_original>cp$optimal_cutpoint)
  cat(precision(TP,FP,TN,FN));cat("\t")
  cat(F1_score(TP,FP,TN,FN));cat("\t")
  cat(recall(TP,FP,TN,FN));cat("\t")
  
  # Train the Random Forest model
  model_rf <- randomForest(status ~ ., data = dataGLM[,c("status","Bmean","Dmean","Pmean","Vmean")], ntree = 50)
  # Make predictions on new data (test_data)
  predictions_original <- predict(model_rf, newdata = dataGLM[,c("status","Bmean","Dmean","Pmean","Vmean")], type = "response")
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLM$status==1]<-"YES"
  cp <- cutpointr(dataROC, dataOut,labels, method = maximize_metric, metric = sum_sens_spec, silent = TRUE)
  cat(as.numeric(cp$AUC));cat("\t");cat(as.numeric(cp$acc));cat("\t");cat(as.numeric(cp$specificity));cat("\t")
  TP<-sum(dataROC$labels=="YES" & predictions_original>=cp$optimal_cutpoint)
  TN<-sum(dataROC$labels=="NO" & predictions_original<=cp$optimal_cutpoint)
  FP<-sum(dataROC$labels=="YES" & predictions_original<cp$optimal_cutpoint)
  FN<-sum(dataROC$labels=="NO" & predictions_original>cp$optimal_cutpoint)
  cat(precision(TP,FP,TN,FN));cat("\t")
  cat(F1_score(TP,FP,TN,FN));cat("\t")
  cat(recall(TP,FP,TN,FN));cat("\t")
  predictions_original <- predict(model_rf, newdata = dataGLMTest[,c("status","Bmean","Dmean","Pmean","Vmean")], type = "response")
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLMTest$status==1]<-"YES"
  cp <- cutpointr(dataROC, dataOut,labels, method = maximize_metric, metric = sum_sens_spec, silent = TRUE)
  cat(as.numeric(cp$AUC));cat("\t");cat(as.numeric(cp$acc));cat("\t");cat(as.numeric(cp$specificity));cat("\t")
  TP<-sum(dataROC$labels=="YES" & predictions_original>=cp$optimal_cutpoint)
  TN<-sum(dataROC$labels=="NO" & predictions_original<=cp$optimal_cutpoint)
  FP<-sum(dataROC$labels=="YES" & predictions_original<cp$optimal_cutpoint)
  FN<-sum(dataROC$labels=="NO" & predictions_original>cp$optimal_cutpoint)
  cat(precision(TP,FP,TN,FN));cat("\t")
  cat(F1_score(TP,FP,TN,FN));cat("\t")
  cat(recall(TP,FP,TN,FN));cat("\n")
}

