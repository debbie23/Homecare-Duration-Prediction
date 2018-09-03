pre_process_data.R: 
preprocessing each files, such as
* removing outliers
* change file format from csv into rds or feather
* combine some predictors 
* roughly select some useful predictors.

tools.R: all tool functions

main.R: 
* create files which are used for predicting home-care for next 1 to 15 week
* plot data and save into plot directory
* create categorical data for classfication

first run pre_process_data.R, then run main.R
    

