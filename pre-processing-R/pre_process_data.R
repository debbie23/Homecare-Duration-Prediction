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
# update.packages(checkBuilt=TRUE, ask=FALSE)

setwd("C:/work/ecare/") #set working directory 

################################################################################
### This file is about prepraing data, about
### 1. remove outliers 
### 2. combine some predictor into new predictor
###    combine the same categorical problem in assessment table
### 3. create new features:
###    meadian and mean care duration for care duration 
###    month, date, quarter date, week number etc information for assessment date and care happend time
### 4. merge two lookup tables with the patient_problem table
### all files created will be used in the tools files 
#################################################################################
#################################################################################
## load data
#################################################################################
# read origianl csv files and change them into rds files
symptons <- read_csv("C:/work/ecare/data/OMAHA_SignsAndSymptoms.csv")
problem <- read_csv("C:/work/ecare/data/OMAHA_Problems.csv")
Patients <- read_csv("C:/work/ecare/data/Patients.csv")
patient_problem <- read_csv("C:/ecare/test/data/Patients_ProblemsWithSigns.csv")
assesments <- read_csv("C:/work/ecare/data/Assessment_CareEstimation.csv")

write_feather(assesments,"C:/work/ecare/data/assesments.feather")
saveRDS(symptons, "C:\\work\\ecare\\data\\symptons.rds")
saveRDS(problem, "C:\\work\\ecare\\data\\problem.rds")
saveRDS(Patients, "C:\\work\\ecare\\data\\Patients.rds")
saveRDS(patient_problem, "C:\\work\\ecare\\data\\patient_problem.rds")

#################################################################################
## read assessement data and combine the same categorical data into one predictor
#################################################################################
## combine multiple predictros 
## environment domain: income, sanitation, residence, workplaceSafety
assesments$environment <- assesments[,12:15]
## psychosocial Domain: 
## Communication with community resources, Social contact, Role change, Interpersonal relationship, 
## Spirituality,Grief,Mental health,Sexuality,Caretaking/parenting,Neglect,Abuse,Growth and development
assesments$psychosocial <- assesments[,c(16:25,51:53)]
## physiological domain: 
## Hearing,Vision,Speech and language,Oral health,Cognition,Pain,
## Consciousness,Skin,Neuro-musculo-skeletal function,Respiration,Circulation,
## Digestion-hydration,Bowel function,Urinary function,Reproductive function,Pregnancy,
## Postpartum,Communicable/infectious condition
assesments$physiological <- assesments[,26:42]
## health-related behavior domain:
## Nutrition,Sleep and rest patterns,Physical activity,Personal care,Substance use,
## Family planning,Health care supervision,Medication regimen
assesments$healthRelated <- assesments[,43:50]
# sum all the amount for each categorical illness
assesments$environment_num <- apply(assesments$environment,1,sum)
assesments$psychosocial_num  <- apply(assesments$psychosocial ,1,sum)
assesments$physiological_num  <- apply(assesments$physiological,1,sum)
assesments$healthRelated_num <- apply(assesments$healthRelated,1,sum)
# select useful predictor
assesments_sub <- assesments[,c(1:11,58:61)] # remove all symptons
# create nuew features by assessment_date
assesments_sub$assessment_date <- format(as.Date(assesments_sub$Assessment_Date), "%m-%d")
assesments_sub$week_num <- week(as.Date(assesments_sub$Assessment_Date)) ## week number 
assesments_sub$year_day <- yday(as.Date(assesments_sub$Assessment_Date)) ## day in the year
assesments_sub$Assessment_Date <- NULL
# save into fearther file since it contains dutch letter, cannot be encoding with utf8 with rds file format 
write_feather(assesments_sub, "C:/work/ecare/data/assesments_sub.feather")

#### read data 
registrattion <- read_csv("C:/work/ecare/data/registration.csv")
patient_problem <- readRDS( "C:/work/ecare/data/patient_problem.rds")
problem <- readRDS( "C:/work/ecare/data/problem.rds")
symptons <- readRDS( "C:/work/ecare/data/symptons.rds")
assesments_sub <-  read_feather("C:/work/ecare/data/assesments_sub.feather")
Patients <- readRDS( "C:/work/ecare/data/Patients.rds")

#################################################################################
###remove outliers
#################################################################################
# clean patient table
levels(Patients$Marital_status)[match("Alleenstaand",levels(patient$Marital_status))] <- "single"
levels(Patients$Marital_status)[match("Gehuwd",levels(patient$Marital_status))] <- "married"
levels(Patients$Marital_status)[match("Samenwonend",levels(patient$Marital_status))] <- "living together"
levels(Patients$Marital_status)[match("Samenwonend met contract",levels(patient$Marital_status))] <- "living together with contract"

# abnormality, select age from 0 to 110 
Patients <-Patients[which(Patients$Age %in% c(0:110)),]
# save
saveRDS(Patients,"C:/work/ecare/data/Patients.rds")

# find the common patient in assessment, patient and registration tables
registrattionPatient <- unique(registrattion$PatientID)
assessmentPatient <- unique(assesments_sub$PatientID)
patientPatient <- unique(Patients$PatientID)
problemPatient<- unique(patient_problem$PatientID)
commonPatientIDs <- Reduce(intersect, list(registrattionPatient,assessmentPatient,patientPatient,problemPatient))
# save RDS
saveRDS(commonPatientIDs,"C:/work/ecare/data/commonPatientIDs.rds")
#################################################################################
#filter all files out in these common patient
registrattion <- registrattion[which(registrattion$PatientID %in% commonPatientIDs), ]
assesments_sub <- assesments_sub[which(assesments_sub$PatientID %in% commonPatientIDs), ]
Patients <- Patients[which(Patients$PatientID %in% commonPatientIDs), ]
patient_problem <- patient_problem[which(patient_problem$PatientID %in% commonPatientIDs), ]
## save all these files
saveRDS(registrattion,"C:/work/ecare/data/registrattion.rds")
write_feather(assesments_sub,"C:/work/ecare/data/assesments_sub.feather")
saveRDS(Patients,"C:/work/ecare/data/Patients.rds")
saveRDS(patient_problem,"C:/work/ecare/data/patient_problem.rds")

#################################################################################
# clean patient_problem data
# select useful predictors
patient_problem = patient_problem[,c("PatientID","problemID","SignAndSymptomID","Endtime")]
# remove end time is below than 2017
patient_problem$Endtime_year <- ifelse(is.na(patient_problem$Endtime),NA,year(as.Date(patient_problem$Endtime,format = "%yyyy-%mm-%dd")))
patient_problem <- patient_problem[which((patient_problem$Endtime_year > 2016) | is.na(patient_problem$Endtime_year)),]
## change endtime 2018 into NA
patient_problem_2018 <- patient_problem[which(patient_problem$Endtime_year == 2018),]
patient_problem_2018$endTime <- NA

patient_problme_not_2018 <-  patient_problem[which(patient_problem$Endtime_year != 2018),]
patient_problme_not_2018$endTime <- patient_problme_not_2018$Endtime

patient_problem <- bind_rows(patient_problme_not_2018,patient_problem_2018)
#get patient problem endtime week number
patient_problem$Endtime_weekNR <- ifelse(is.na(patient_problem$endTime),NA,week(patient_problem$endTime))
# filter useful predictor
problem <- problem[,c("ProblemID","ProblemName_US")]
symptons <- symptons[,c("SignAndSymptomID","SignAndSymptomName_US","ProblemID")]
illnessName <- merge(symptons,problem,by="ProblemID")

colnames(patient_problem)[names(patient_problem) == "problemID"] <- "ProblemID"
patient_problem_merged <-merge(patient_problem,illnessName, by=c("ProblemID","SignAndSymptomID"))

patient_problem_merged <- select(patient_problem_merged,-c("ProblemID","SignAndSymptomID"))

#################################################################################
### calculate mean and meadian care duration for each week 
### and save the file into register_by_week
#################################################################################
# select useful predictors
registrattion <- registrattion[,c("Financer","PatientID","WeekNR","JobTitle","TeamID","Starttime","Einddtime")]

registrattion$timeDiff <- as.numeric(difftime(strptime(registrattion$Einddtime, "%Y-%m-%d %H:%M:%S"),
                                              strptime(registrattion$Starttime, "%Y-%m-%d %H:%M:%S"))
                                     , units = "mins")

## cant handle 2017.3.26, difffTimeg get NA for all data on this date
test <-registrattion[which(is.na(registrattion$timeDiff)),]
test$EndwithOutDate <- format(as.POSIXct(test$Einddtime) ,format = "%H:%M:%S")
test$StartwithOutDate <- format(as.POSIXct(test$Starttime) ,format = "%H:%M:%S")

test$timeDiff <-as.numeric(difftime(strptime(test$EndwithOutDate, "%H:%M:%S"),
                                    strptime(test$StartwithOutDate, "%H:%M:%S"))
                           , units = "mins")

test_not_NA <-registrattion[which(!is.na(registrattion$timeDiff)),]
registrattion <- bind_rows(test, test_not_NA)

saveRDS(patient_problem_merged, "C:/work/ecare/data/patient_problem_merged.rds")
saveRDS(registrattion, "C:/work/ecare/data/registrattion_with_timeDiff.rds")


rm(list=ls())
patient_problem_merged <- readRDS( "C:/work/ecare/data/patient_problem_merged.rds")
registrattion <- readRDS( "C:/work/ecare/data/registrattion_with_timeDiff.rds")

commonPatientIDs <- readRDS("C:/work/ecare/data/commonPatientIDs.rds")
registrattion <- registrattion[which(registrattion$PatientID %in% commonPatientIDs), ] ## 11,842,740 => 1,859,727
saveRDS(registrattion, "C:/work/ecare/data/registrattion_with_timeDiff.rds")
#################################################################################
# feature engineering: create mean and meadian home-care duration for each week
registration_by_week <- registrattion %>%
  group_by(WeekNR,PatientID) %>%
  summarise(mean_week_time = mean(as.numeric(timeDiff)),
            median_week_time = median(as.numeric(timeDiff)),
            care_times_weekly = n(),
            total_week_time = sum(as.numeric(timeDiff)))%>%
  ungroup()%>%
  arrange(WeekNR)

saveRDS(registration_by_week,"C:/work/ecare/data/registration_by_week.rds")
summary(registration_by_week)




