rm(list=ls())
source("C:/work/ecare/pre-processing-R/tools.R")

setwd("C:/work/ecare/") #set working directory 
#############################################################################
##  this is the main file to run
#############################################################################
#############################################################################
## create data that for predicting fueture 1 to 10 week home-care duration
#############################################################################
## fucntion:get_combined_data
## combine multiple files into one file
## parameter: 
##     i: week number, the response is the home-care duration in the future i week 
## function: pre_processing_data
## scaling, center and Yeo-Johnson transformation data
## function: extract_features: 
## feature selection based on entropy-based method
## parameter:
##     i: week number, the response is the home-care duration in the future i week 
##     random seed list: used for set the random seed for sample 50000 data from dataset,
##                       and these samples will be used for feature selection 
##                       remove the common non-important features 
for (i in c(8:8)){
  get_combined_data(i)
  pre_processing_data(i)
  extract_features(i,c(1234,8888,10,46))
}

#############################################################################
## plot data for data without any preprocessing, 
## that is, scaling, center, Yeo-Johnson transformation and feature selection
#############################################################################
## function savePlot:
## scatter plot for numerical data
## boxplot for cateorical data
## violin plot, boxplot and bar plot for the response 
## parameter:
##     i: week number, the response is the home-care duration in the future i week 
## fucntion savCorr:
## plot correlation
## parameter: 
##     i: week number, the resposne is teh home-care duration in the future i week
##     withHeatmap: yes:   plot correlation in heatmap
##                  no:    plot correlaion in matrix plot
##                  other: plot correlation in both heatmap and matrix plot
for(i in c(11:15)){
  # allCom <- feature_engineering(i)
  # savePlot(allCom,i)
  saveCorr(i,"other")
}


#############################################################################
## create categorical data for classfication
#############################################################################
allCom <- readRDS("C:/work/ecare/data/allCom_v9_final.rds")
summary(allCom)

create_categorized_file <- function(startPoint, intervals, dataset){
  # create predictor about care duration categorization 
  dataset$duration_next_week_cat <- ceiling(dataset$duration_next_week/60)
  
  df <- data.frame(table(dataset$duration_next_week_cat))
  colnames(df) <- c('categorize','Freq')
  df$Perc <- df$Freq / sum(df$Freq) * 100
  df$iterats <- as.numeric(as.factor(df$categorize))
  df

  iterations <- ceiling((length(df$categorize) - startPoint)/intervals)
  # merge df and dataset
  df <- df[,c("categorize","iterats")]
  dataset <- merge(dataset, df,by.x = "duration_next_week_cat", by.y="categorize")
  
  for(i in c(1:iterations)){
    if(i == iterations){
      dataset$duration_next_week_cat[dataset$iterats %in% c(((startPoint+1)+intervals*(i-1)):(max(dataset$iterats)))] <- startPoint+i
    }else{
      dataset$duration_next_week_cat[dataset$iterats %in% c(((startPoint+1)+intervals*(i-1)):((startPoint+1)+intervals*i))] <- startPoint+i
    }
  }
  return(dataset)
}
# start from categorize 23, and categorize each 19 classes into one class
allCom <- create_categorized_file(23,10,allCom)
# check
summary(allCom$duration_next_week_cat)
write.csv(allCom,"C:/work/ecare/data/allCom_v1_categorized.csv",row.names=FALSE)

#############################################################################
## remove correlation, test for classification 
#############################################################################
allCom <- readRDS("C:/work/ecare/data/allCom_v1_final.rds")
## remove highly correlated one, test for nueral network 
asNumeric <- function(x) as.numeric(as.character(x))
factorsNumeric <- function(d) modifyList(d, lapply(d[, sapply(d, is.factor)],   
                                                   asNumeric))
allCom <-factorsNumeric(allCom)

descrCor <- cor(allCom)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .9)
allCom <- allCom[,-highlyCorDescr]
write.csv(allCom,"C:/work/ecare/data/allCom_v1_final_removeCorr_0.75.csv",row.names=FALSE)
