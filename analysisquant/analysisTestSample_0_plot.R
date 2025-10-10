rm(list = ls())
library(e1071)
library(pROC)
options(warn = -1)
load("TrainTestDataset.RData")
for(analysis in names(TrainTestDataset)){
  dataFrameALL<-TrainTestDataset[[analysis]]
  ###########Let us summarize Train dataset
  dataFrame<-dataFrameALL[dataFrameALL$statusOrigin=="Train",]
  png(paste0("FigureSamples/ROC_origin_", analysis, ".png"), width = 500, height = 500, res = 100)
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
  ###########Let us summarize Test dataset
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
  ###svm data
  svm_model <- svm(status ~ Bmean + Dmean + Pmean + Vmean, data = dataGLM, kernel = "radial", cost = 1, gamma = 0.1)
  predictions_original <- predict(svm_model)
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLM$status==1]<-"YES"
  result <- roc(dataROC$labels, dataROC$dataOut, plot=F, quiet = TRUE);auc1<-round(as.numeric(result$auc),3)
  plot(1-result$specificities, result$sensitivities, col="black", xlab = "False Positive Rate", ylab = "True Positive Rate", pch=19)
  predictions_original <- predict(svm_model, dataGLMTest[, c("Bmean","Dmean","Pmean","Vmean")])
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLMTest$status==1]<-"YES"
  result <- roc(dataROC$labels, dataROC$dataOut, plot=F, quiet = TRUE);auc2<-round(as.numeric(result$auc),3)
  lines(1-result$specificities, result$sensitivities, col="blue", pch=19, lty='dotted', lwd=3)
  
  mylogit <- glm(status ~ Bmean + Dmean + Pmean + Vmean, family = binomial(link = "logit"), data = dataGLM)
  predictions_original <- predict(mylogit)
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLM$status==1]<-"YES"
  result <- roc(dataROC$labels, dataROC$dataOut, plot=F, quiet = TRUE);auc3<-round(as.numeric(result$auc),3)
  lines(1-result$specificities, result$sensitivities, col="red", pch=19, lty='dotted', lwd=3)
  predictions_original <- predict(mylogit, dataGLMTest[, c("Bmean","Dmean","Pmean","Vmean")])
  dataROC<-data.frame(labels="NO", dataOut=predictions_original)
  dataROC$labels[dataGLMTest$status==1]<-"YES"
  result <- roc(dataROC$labels, dataROC$dataOut, plot=F, quiet = TRUE);auc4<-round(as.numeric(result$auc),3)
  lines(1-result$specificities, result$sensitivities, col="green", pch=19, lty='dotted', lwd=3)
  legend(x ='bottomright', legend=c(paste0("SVMTrain_AUC", auc1),paste0("SVMTest_AUC", auc2),paste0("LogisticTrain_AUC", auc3),paste0("LogisticTest_AUC", auc4))
         , fill = c("black","blue","red","green"))
  dev.off()
}

