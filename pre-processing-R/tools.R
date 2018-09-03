rm(list=ls())
library(ggplot2)
library(lubridate)
library(gridExtra)
library(scales)
library(GGally)
library(dplyr) 
library(feather)
library(readr)
library(caret)
library(ggcorrplot)

# update.packages(checkBuilt=TRUE, ask=FALSE)

setwd("C:/work/ecare/") #set working directory 
#####################################################################
## this file is all functions that will be called in main.R
#####################################################################

#####################################################################
### combined all files 
## weekNumDiff: int, this is you wanna predict how many weeks after assessment
#####################################################################
get_combined_data <- function(weekNumDiff){
  registration_by_week <- readRDS("C:/work/ecare/data/registration_by_week.rds")
  registrattion <- readRDS("C:/work/ecare/data/registrattion_with_timeDiff.rds")
  patient_problem_merged <- readRDS( "C:/work/ecare/data/patient_problem_merged.rds")
  
  ## merged registration_next with registration 
  registrattion <- merge(registration_by_week,registrattion,by=c("PatientID","WeekNR"))
  registration_next_week <- registration_by_week[,c("WeekNR","PatientID","total_week_time")]
  
  colnames(registration_next_week)[which(names(registration_next_week) == "WeekNR")] <- "NextWeekNR"
  colnames(registration_next_week)[which(names(registration_next_week) == "total_week_time")] <- "next_week_duration"
  
  registration_next_week$WeekNR <- registration_next_week$NextWeekNR -weekNumDiff
  registrattion <- merge(registration_next_week,registrattion, by=c("PatientID","WeekNR"))
  
  filnemas <- paste("C:/work/ecare/data/registrattion_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,".rds",sep = "")
  saveRDS(registrattion, filnemas)
  
  registration_next_week <- NULL
  ## combine registrattion and patient_problem_merged 
  colnames(patient_problem_merged) <- c("PatientID","Problem_EndTime","problem_Endtime_weekNR","SignAndSymptomName","ProblemName")
  colnames(registrattion) <- c("PatientID","register_WeekNR","register_NextWeekNR","duration_next_week","duration_mean_each_time","duration_median_each_time",
                               "care_times_weekly","duration_week_minus","Financer","JobTitle","TeamID","register_Starttime","register_Einddtime","duration_each_time" )
  
  patient_problem_merged$problem_end_weekNR <- ifelse(is.na(patient_problem_merged$Problem_EndTime),53,week(patient_problem_merged$Problem_EndTime)) 
  patient_problem_merged$problem_Endtime_weekNR <- NULL
  problem <- patient_problem_merged
  patient_problem_merged <- NULL
  
  problem_reg <- merge(registrattion,problem,by="PatientID")
  # rm(list=setdiff(ls(), "problem_reg","weekNumDiff"))
  problem_reg <- problem_reg[which(problem_reg$problem_end_weekNR > problem_reg$register_WeekNR),] 
  
  
  ##############################
  ## combined with assessment
  ###############################
  assesments_sub <- read_feather("C:/work/ecare/data/assesments_sub.feather")
  colnames(assesments_sub) <- c("PatientID","estimated_care_duration","estimated_care_request", "Estimated_CareMoments_Weekly",
                                "Estimated_Minutes_Weekly","Advice_Instructions_Travel","Treatments",
                                "Case_management","Monitoring_Bevaking","environment_num","psychosocial_num","physiological_num",
                                "healthRelated_num","assessment_date","ass_week_num","ass_year_day")
  assesments_sub$register_WeekNR <- assesments_sub$ass_week_num
  ass_problem_reg <- merge(assesments_sub,problem_reg,by=c("PatientID","register_WeekNR"))
  assesments_sub <- NULL 
  
  ############################
  ###combined with patient
  ############################
  # rm(list=setdiff(ls(), "ass_problem_reg","weekNumDiff"))
  Patients <- readRDS( "C:/work/ecare/data/Patients.rds")
  
  # filter pateint
  ass_patientID <- unique(ass_problem_reg$PatientID)
  Patients <- Patients[which(Patients$PatientID %in% ass_patientID),]
  # select useful predictor
  Patients <- select(Patients, -c("Date_StartofCare","Date_EndOfCare","Date_Deceased","Date_EndOfCare_year","Date_EndOfCare_copied"))
  allCom <- merge(ass_problem_reg,Patients, by = "PatientID")
  # rm(list=setdiff(ls(), "allCom","weekNumDiff"))
  
  allCom$TeamID <- factor(allCom$TeamID)
  allCom <- select(allCom, -c("Problem_EndTime","problem_end_weekNR"))
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_raw.rds",sep = "")
  saveRDS(allCom, filnemas)
}


############################################
## preprocessing data: 
## include:
## 1. exract information from date time data
## 2. one-hot-encoding for categorical data
## 3. scale, center and remove Zero- and Near Zero-Variance Predictors,Yeo-Johnson transformation
##########################################

pre_processing_data <- function(weekNumDiff){
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_raw.rds",sep = "")
  allCom <- readRDS(filnemas)
  # extract info from assessment_date
  allCom$assessment_date <- paste("2017-",allCom$assessment_date,sep = "")
  allCom$assessment_date <- as.Date(allCom$assessment_date)
  allCom$ass_date <- day(allCom$assessment_date)
  allCom$ass_weekday <- wday(allCom$assessment_date)
  allCom$ass_month <- month(allCom$assessment_date)
  allCom$ass_month_day <- mday(allCom$assessment_date)
  allCom$ass_quater_day <- qday(allCom$assessment_date)
  
  allCom$assessment_date  <- NULL
  
  #extract info from registration care end time
  head(allCom$register_Einddtime)
  allCom$register_Einddtime <- as.Date(allCom$register_Einddtime)
  allCom$reg_date <- day(allCom$register_Einddtime)
  allCom$reg_weekDay <- wday(allCom$register_Einddtime)
  allCom$reg_month  <- month(allCom$register_Einddtime)
  allCom$reg_month_day <- mday(allCom$register_Einddtime)
  allCom$reg_quater_day <- qday(allCom$register_Einddtime)
  allCom$register_Einddtime <- NULL
  allCom$register_Starttime <- NULL
  
  # change na into unknown
  # head(allCom[which(!allCom$Marital_status %in% c("Gehuwd","Alleenstaand","Samenwonend","Samenwonend met contract")),])
  allCom$Marital_status[is.na(allCom$Marital_status)] <- "unkwnon"
  
  cols <- c("estimated_care_duration", "estimated_care_request", "Advice_Instructions_Travel",
            "Treatments","Case_management","Monitoring_Bevaking","ass_week_num","ass_year_day",
            "register_NextWeekNR","Financer","JobTitle","TeamID","SignAndSymptomName","ProblemName",
           "Gender","Marital_status","Living_unit","ass_date","ass_weekday","ass_month","ass_month_day",
            "ass_quater_day","reg_date","reg_weekDay","reg_month","reg_month_day","reg_quater_day")
  
  colnames(allCom)
  allCom[cols] <- lapply(allCom[cols], factor)
  
  # rm(list=setdiff(ls(), "allCom","weekNumDiff"))
  allCom <- select(allCom, -c("PatientID"))
  
  ## replace jobTitle into english in case of encoding error
  levels(allCom$JobTitle) <- c("substitute power","Pupil","Subcontractor","Cleaner","trainee",
                               "nursing specialist","nurse in the neighborhood","nurturing level 2",
                               "nurturing level 3","district nurse","District nurse/nursing specialist",
                               "ward nurse","Sick-nurse","ZOZer")
  
  ## one-hot-encoding 
  dmy <- dummyVars(" ~Gender", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("Gender"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  dmy <- dummyVars(" ~Marital_status", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("Marital_status"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  dmy <- dummyVars(" ~Living_unit", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("Living_unit"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  
  dmy <- dummyVars(" ~Financer", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("Financer"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  
  dmy <- dummyVars(" ~JobTitle", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("JobTitle"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  
  dmy <- dummyVars(" ~estimated_care_duration", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("estimated_care_duration"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  
  dmy <- dummyVars(" ~estimated_care_request", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("estimated_care_request"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  summary(allCom)
  
  teamiDs <- allCom %>%
    group_by(TeamID) %>%
    dplyr::summarise( frequencies = n()/(dim(allCom)[1])) %>%
    ungroup()%>% 
    arrange(desc(frequencies))
  
  teamiDs_greater_than_005 <- teamiDs[which(teamiDs$frequencies > 0.005),]
  allCom$TeamID <-as.numeric(levels(allCom$TeamID))[allCom$TeamID]
  allCom_sub <- allCom[which(allCom$TeamID %in% teamiDs_greater_than_005$TeamID),]
  allCom_sub_remained <- allCom[which(!(allCom$TeamID %in% teamiDs_greater_than_005$TeamID)),]
  allCom_sub_remained$TeamID <- 0
  allCom <- bind_rows(allCom_sub, allCom_sub_remained)
  allCom$TeamID <- factor(allCom$TeamID)
  
  teamiDs<-NULL
  teamiDs_greater_than_005<-NULL
  allCom_sub<-NULL
  allCom_sub_remained <- NULL
  
  
  length(unique(allCom$TeamID))
  dmy <- dummyVars(" ~TeamID", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("TeamID"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  summary(allCom)
  
  
  dmy <- dummyVars(" ~ProblemName", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  # allCom <- select(allCom,-c("ProblemName"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  summary(allCom)
  
  synFreq <- allCom %>%
    group_by(SignAndSymptomName,ProblemName) %>%
    summarise( frequencies = n()/(dim(allCom)[1])) %>%
    ungroup() %>%
    arrange(desc(frequencies))
  
  
  synFreq_01 <- synFreq[which(synFreq$frequencies > 0.01),]
  allCom_sub <- allCom[which((allCom$SignAndSymptomName %in% synFreq_01$SignAndSymptomName) 
                             &(allCom$ProblemName %in% synFreq_01$ProblemName)),]
  allCom_sub$combindSymptonProblem <- paste(allCom_sub$ProblemName,allCom_sub$SignAndSymptomName,sep="_")
  allCom_sub_remained <- allCom[which(!((allCom$SignAndSymptomName %in% synFreq_01$SignAndSymptomName) 
                                        &(allCom$ProblemName %in% synFreq_01$ProblemName))),]
  allCom_sub_remained$SignAndSymptomName <- "omittedSymptons"
  allCom_sub_remained$combindSymptonProblem <- "omittedSymptons"
  allCom <- bind_rows(allCom_sub, allCom_sub_remained)
  allCom_sub_remained <- NULL
  allCom_sub <- NULL
  
  dmy <- dummyVars(" ~combindSymptonProblem", data = allCom)
  trsf <- data.frame(predict(dmy, newdata = allCom))
  allCom <- select(allCom,-c("combindSymptonProblem"))
  allCom <- cbind(allCom,trsf)
  trsf <- NULL
  summary(allCom)
  synFreq_01 <- NULL
  synFreq <- NULL
  allCom <- select(allCom,-c("SignAndSymptomName","ProblemName"))
  
  ## remove non-variance columns
  nzv <- nearZeroVar(allCom)
  remove_non_zero_data<- allCom[, -nzv]
  dim(remove_non_zero_data)
  
  setdiff(colnames(allCom), colnames(remove_non_zero_data))
  
  ###  scale and ceterin data, also use Yeo-Johnson transoform
  label <- select(remove_non_zero_data, c('duration_next_week')) 
  non_label <- select(remove_non_zero_data, -c("duration_next_week"))
  
  pp_hpc <- preProcess(non_label,  method = c("center", "scale", "YeoJohnson"))
  transformed_without_label  <- predict(pp_hpc, newdata = non_label)
  
  allCom <- cbind(transformed_without_label,label)
  
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_no_feature_selection.rds",sep = "")
  saveRDS(allCom, filnemas)
}  



###########################################################
## use entropy based methods to extract features, remove features with 0 attribut importance
## seedList is a list used in set.seed() to randomly select 50000 samples to test feature importance
## seedList: for example: c(1,293,190022) 
####################################################
extract_features <- function(weekNumDiff,seedList){
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_no_feature_selection.rds",sep = "")
  allCom <- readRDS(filnemas)
  allCom <- allCom[, !duplicated(colnames(allCom))]
  
  if (Sys.getenv("JAVA_HOME")!="")Sys.setenv(JAVA_HOME="")
  library(rJava)
  library(FSelector)
  # extend java heap memory into 8g
  options(java.parameters = "-Xmx8g") 
  
  for(i in seedList){
    set.seed(i)
    allCom_sample <- allCom[sample(nrow(allCom), 5000),]
    weights <- information.gain(duration_next_week~., allCom_sample)
    sorted_names1 <- rownames(weights)[which(weights$attr_importance == 0)]
    weights <- gain.ratio(duration_next_week~., allCom_sample)
    sorted_names2 = rownames(weights)[which(weights$attr_importance == 0)]
    weights <- symmetrical.uncertainty(duration_next_week~., allCom_sample)
    sorted_names3 = rownames(weights)[which(weights$attr_importance == 0)]
    if (exists("commonList")){
      commonList <- Reduce(intersect, list(sorted_names1,sorted_names2,sorted_names3,commonList))
    }else{
      commonList <- Reduce(intersect, list(sorted_names1,sorted_names2,sorted_names3))
    }
  }
  
  allCom <- select(allCom , -commonList)
  
  
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_final.rds",sep = "")
  saveRDS(allCom, filnemas)
  
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_final.csv",sep = "")
  write.csv(allCom,filnemas,row.names=FALSE)
  
}

###################################################
## scatter plot for each predictor with label
#################################################
## read raw data and creat feartures for date type data
feature_engineering <- function(weekNumDiff){
  filnemas <- paste("C:/work/ecare/data/allCom_v",weekNumDiff,sep = "")
  filnemas <- paste(filnemas,"_raw.rds",sep = "")
  allCom <- readRDS(filnemas)
  
  allCom$assessment_date <- paste("2017-",allCom$assessment_date,sep = "")
  allCom$assessment_date <- as.Date(allCom$assessment_date)
  allCom$ass_weekday <- wday(allCom$assessment_date)
  allCom$ass_month <- month(allCom$assessment_date)
  allCom$ass_month_day <- mday(allCom$assessment_date)
  allCom$ass_quater_day <- qday(allCom$assessment_date)
  
  allCom$assessment_date  <- NULL
  
  #extract info from registration care end time
  head(allCom$register_Einddtime)
  allCom$register_Einddtime <- as.Date(allCom$register_Einddtime)
  allCom$reg_weekDay <- wday(allCom$register_Einddtime)
  allCom$reg_month  <- month(allCom$register_Einddtime)
  allCom$reg_month_day <- mday(allCom$register_Einddtime)
  allCom$reg_quater_day <- qday(allCom$register_Einddtime)
  allCom$register_Einddtime <- NULL
  allCom$register_Starttime <- NULL
  
  # change na into unknown
  # head(allCom[which(!allCom$Marital_status %in% c("Gehuwd","Alleenstaand","Samenwonend","Samenwonend met contract")),])
  allCom$Marital_status[is.na(allCom$Marital_status)] <- "unkwnon"
  allCom <- select(allCom, -c("PatientID","ProblemName","SignAndSymptomName","register_NextWeekNR"))
  # allCom <- select(allCom, -c("PatientID","ProblemName","SignAndSymptomName","register_NextWeekNR","Problem_EndTime","problem_end_weekNR"))
  allCom$TeamID <- as.factor(as.numeric(allCom$TeamID))
  # allCom$register_WeekNR<-as.factor(allCom$register_WeekNR)
  

  cols <- c("register_WeekNR","estimated_care_duration","estimated_care_request","Estimated_CareMoments_Weekly",
        "Advice_Instructions_Travel","Treatments","Case_management",
        "Monitoring_Bevaking","environment_num","psychosocial_num","physiological_num",
        "healthRelated_num","ass_week_num","ass_year_day",
        "care_times_weekly",
        "Financer","JobTitle",
        "Age","Gender","Marital_status","Living_unit",
        "ass_weekday","ass_month","ass_month_day","ass_quater_day",
        "reg_weekDay","reg_month","reg_month_day","reg_quater_day" )

  # cols <- c("register_WeekNR")
            
  # colnames(allCom)
  allCom[cols] <- lapply(allCom[cols], factor)
  
  return(allCom)
}

breaksOnScales <- function(type){
  return (switch(type, 
                 quater_day_type= scale_x_discrete(breaks = c(1:92)), 
                 weekDay_type=  scale_x_discrete(breaks = c(1:7)),
                 moth_day_type=  scale_x_discrete(breaks = c(1:31)),
                 moth_type= scale_x_discrete(breaks = c(1:12)),
                 weekNR_type= scale_x_discrete(breaks = c(1:51)),
                 teamID_type = scale_x_discrete(breaks = seq(1,858,by=33)),
                 others = NULL))
}
## plot scatter
savePlot <- function(datasets,weekNumDiff){
  for (i in 1:length(colnames(datasets))){
    predictor_name <- colnames(datasets)[i]
    quater_day_type <- length(grep("(_quater_day)", predictor_name, perl = TRUE, value = TRUE))
    moth_type <- length(grep("(_month)", predictor_name, perl = TRUE, value = TRUE))
    moth_day_type <- length(grep("(_month_day)", predictor_name, perl = TRUE, value = TRUE))
    weekDay_type <- length(grep("(_weekDay)", predictor_name, perl = TRUE, value = TRUE))
    weekNR_type <-  length(grep("(_WeekNR)", predictor_name, perl = TRUE, value = TRUE))
    teamID_type <-  length(grep("(TeamID)", predictor_name, perl = TRUE, value = TRUE))
    type <- "others"
    myPlot <- NULL

    if (quater_day_type != 0){ type <- "quater_day_type"}
    if (moth_type != 0){ type <- "moth_type"}
    if (moth_day_type != 0){ type <- "moth_day_type"}
    if (weekDay_type != 0){ type <- "weekDay_type"}
    if (weekNR_type != 0){ type <- "weekNR_type"}
    if (teamID_type != 0){  type <- "teamID_type"}

    addBreaks <- breaksOnScales(type)

    myPlot <- ggplot(allCom, aes_string(predictor_name, "duration_next_week"))+
      # geom_point(alpha=.2,position = position_jitter(h=0))+
      geom_boxplot()+
      labs(x =predictor_name, y = "total care time weekly",title=paste(predictor_name,"VS duration_next_week",sep=""))+
      theme(axis.text.x=element_text(color = "black", size=8, angle=45, vjust=.8, hjust=0.8),
            axis.title=element_text(size=15))+
      addBreaks

    if(predictor_name %in% c("ass_year_day","duration_mean_each_time","duration_median_each_time",
                             "duration_week_minus","duration_each_time","Estimated_Minutes_Weekly",
                             "TeamID")){
      myPlot <- ggplot(allCom, aes_string(predictor_name, "duration_next_week"))+
        geom_point(alpha=.2,position = position_jitter(h=0))+
        labs(x =predictor_name, y = "total care time weekly",title=paste(predictor_name,"VS duration_next_week",sep=""))+
        theme(axis.text.x=element_text(color = "black", size=8, angle=45, vjust=.8, hjust=0.8),
              axis.title=element_text(size=15))+
        addBreaks
    }

    fileName = paste0("C:/work/ecare/plot/v",weekNumDiff, "/", predictor_name, "VScare.png")
    ggsave(filename=fileName, plot=myPlot,width=5,height=5,limitsize=FALSE)

  }
  ## save histgram for the response
  p1<- datasets %>% count(duration_next_week) %>% 
    ggplot(aes(n,duration_next_week)) + 
    geom_boxplot()+
    coord_flip()+
    labs(y="next week care duration", x="count")
  
  p2<- ggplot(data=datasets, aes(datasets$duration_next_week)) +
    geom_bar(color="black")+
    labs(x = "next week care duration", y="count")
  
  p3<- datasets %>% count(duration_next_week) %>% 
    ggplot(aes(n,duration_next_week))+ 
    geom_violin()+
    coord_flip()+
    labs(y="next week care duration", x= "count")
  
  myPlot <- arrangeGrob(p1,p2,p3, ncol =1)
  fileName = paste0("C:/work/ecare/plot/v",weekNumDiff, "/hist.png")
  ggsave(filename=fileName, plot=myPlot,width=5,height=5,limitsize=FALSE)
}

##########################################
## save correlation plot
#####################################
library(gplots)
saveCorr <- function(weekNumDiff,withDendrogram){
  fileName = paste0("C:/work/ecare/data/allCom_v",weekNumDiff, "_final.rds")
  allCom <- readRDS(fileName)
  # change all predictors into numeric
  asNumeric <- function(x) as.numeric(as.character(x))
  factorsNumeric <- function(d) modifyList(d, lapply(d[, sapply(d, is.factor)],   
                                                     asNumeric))
  allCom <-factorsNumeric(allCom)
  # replace dot with underscore incolu mn names
  names(allCom) <- gsub("\\.", "_", names(allCom))
  # plot correlation
  # Barring the no significant coefficient
  corr <- round(cor(allCom),1)
  # corr <-round(cor(allCom[sapply(allCom, function(x) !is.factor(x))]),1)
  par(oma=c(10,4,4,2))
  switch(
    withDendrogram,
    yes={col<- colorRampPalette(c("blue", "white", "red"))(20)
         fileName = paste0("C:/work/ecare/plot/v",weekNumDiff, "/cor_heat.png")
         png(file=fileName, units="in", width=8, height=8, res=300)
         # heatmap.2(x = corr, col = col, symm = TRUE,tracecol=NA)
         heatmap.2(x=corr,col =col, scale="row",
                   key=TRUE, symkey=FALSE, density.info="none",cexRow=0.6,cexCol=0.6,margins=c(10,10),trace="none",srtCol=45,srtRow = 45)
         dev.off()},
    no={
      p.mat <- cor_pmat(allCom)
      coorlot <- ggcorrplot(corr, hc.order = TRUE,
                            type = "lower", p.mat = p.mat)
      fileName = paste0("C:/work/ecare/plot/v",weekNumDiff, "/cor.png")
      ggsave(filename=fileName, plot=coorlot, width=10,height=10,limitsize=FALSE)
    },
    {
      col<- colorRampPalette(c("blue", "white", "red"))(20)
      fileName = paste0("C:/work/ecare/plot/v",weekNumDiff, "/cor_heat.png")
      png(file=fileName, units="in", width=8, height=8, res=300)
      # heatmap.2(x = corr, col = col, symm = TRUE,tracecol=NA)
      heatmap.2(x=corr,col =col, scale="row",
                key=TRUE, symkey=FALSE, density.info="none",cexRow=0.6,cexCol=0.6,margins=c(10,10),trace="none",srtCol=45,srtRow = 45)
      dev.off()
      
      p.mat <- cor_pmat(allCom)
      coorlot <- ggcorrplot(corr, hc.order = TRUE,
                            type = "lower", p.mat = p.mat)
      fileName = paste0("C:/work/ecare/plot/v",weekNumDiff, "/cor.png")
      ggsave(filename=fileName, plot=coorlot, width=10,height=10,limitsize=FALSE)
    })
 
}
