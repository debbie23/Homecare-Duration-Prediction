from pyspark import SparkContext
from pyspark.sql import SQLContext
import socket, struct
from pyspark.sql.functions import udf
from pyspark.sql.functions import explode
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import json



sc = SparkContext(appName="patientTime")
sqlc = SQLContext(sc)


#df = sqlc.read.format('csv').option("header", 'true').load("/user/s1988476/collections/test.csv")
df = (sqlc.read.format("com.databricks.spark.csv").option("header", "true")
	.option("inferschema", "true")
	.option("mode", "DROPMALFORMED")
	.load("/home/rachel/Downloads/allCom_v1_final.csv")
	.drop("PatientID")
	.withColumnRenamed("duration_next_week","label")
	.drop("register_NextWeekNR")
	.drop("assessment_date")
	.drop("PatientID")
	.drop("register_NextWeekNR")
	.drop("problem_end_weekNR")
	.drop("register_Starttime")
	.drop("register_Einddtime"))

df.cache()
for col in df.columns:
  df = df.withColumnRenamed(col,col.replace(".", "_"))
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.sql.functions import col  # for indicating a column using a string in the line below
df = df.select([col(c).cast("double").alias(c) for c in df.columns])
# test = test.select([col(c).cast("double").alias(c) for c in test.columns])
df.printSchema()



def load_data_kfold(k,X,Y):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X,Y))
    return folds


k=5
predicts = df.select('label').toPandas()
features =df.drop('label').toPandas()

import numpy as np
rownums = predicts.values.shape[0]
folds = load_data_kfold(k,features.values,np.reshape(predicts.values,[rownums,]))
for j,(train_index,val_index) in enumerate(folds):
  trainXDF = features.loc[train_index]
  trainYDF = predicts.loc[train_index]
  
  testXDF = features.loc[val_index]
  testYDF = predicts.loc[val_index]
  
  trainDF = pd.concat([trainXDF, trainYDF], axis=1)
  testDF = pd.concat([testXDF, testYDF], axis=1)
  train = sqlc.createDataFrame(trainDF)
  test = sqlc.createDataFrame(testDF)
  
  featuresCols = train.columns
  featuresCols.remove('label')

  vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
  vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=2)

  from pyspark.ml.regression import GBTRegressor,GBTRegressionModel
  gbt = GBTRegressor(labelCol="label")

  from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
  from pyspark.ml.evaluation import RegressionEvaluator
  # Define a grid of hyperparameters to test:
  #  - maxDepth: max depth of each decision tree in the GBT ensemble
  #  - maxIter: iterations, i.e., number of trees in each GBT ensemble
  paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [10,20])\
  .addGrid(gbt.maxIter, [20,30])\
  .build()
  # We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
  evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
  # Declare the CrossValidator, which runs model tuning for us.
  cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid,numFolds = 2)

  from pyspark.ml import Pipeline,PipelineModel
  pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])


  pipelineModel = pipeline.fit(train)
  predictions = pipelineModel.transform(test)
  rmse = evaluator.evaluate(predictions)
  r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
  mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

  # print(pipelineModel.bestModel.getMaxIter())
  # maxIter
  #getBestModel = pipelineModel.stages[-1].bestModel
  #maxIter = getBestModel._java_obj.getMaxIter() ##maxDepth: 5 , maxIter:100
  #maxDepth = getBestModel._java_obj.getMaxDepth() ##maxDepth: 5 , maxIter:100
  #print("maxIter is ",maxIter)
  #print("maxDepth is",maxDepth)
  print("PipeLine stages ",pipelineModel.stages)
  
  exDict = {'RMSE': rmse,'r2':r2,'mae':mae}

  with open("/home/rachel/Downloads/ecare/regression/GBTs/result/result1_"+str(j)+".txt", 'w') as file:
    file.write(json.dumps(exDict)) # use `json.loads` to do the reverse

  rfPath = "/home/rachel/Downloads/ecare/regression/GBTs/model/testModel1_"+str(j)
  pipelineModel.save(rfPath)
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  y_valid_cv = predictions.select("prediction").toPandas().values
  y_pred = predictions.select("label").toPandas().values
  
  regressor.fit(y_valid_cv.reshape(-1, 1), y_pred)
  y_fit = regressor.predict(y_pred)

  reg_intercept = round(regressor.intercept_[0], 4)
  reg_coef = round(regressor.coef_.flatten()[0], 4)
  reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

  plt.scatter(y_valid_cv, y_pred, color='blue', label='data')
  plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
  plt.title('Linear Regression')
  plt.legend()
  plt.xlabel('observed')
  plt.ylabel('predicted')
  plt.savefig("/home/rachel/Downloads/ecare/regression/GBTs/plot/v1/linear_regression_" + str(j) + "_fold.png", dpi=100)
  plt.clf()

  

