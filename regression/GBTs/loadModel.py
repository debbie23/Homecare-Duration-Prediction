from pyspark import SparkContext
from pyspark.sql import SQLContext
import socket, struct
from pyspark.sql.functions import udf
from pyspark.sql.functions import explode
from pyspark.ml.evaluation import RegressionEvaluator
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

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.sql.functions import col  # for indicating a column using a string in the line below
df = df.select([col(c).cast("double").alias(c) for c in df.columns])
# test = test.select([col(c).cast("double").alias(c) for c in test.columns])
df.printSchema()

train, test = df.randomSplit([0.7, 0.3])
from pyspark.ml import Pipeline,PipelineModel
################
#load model
rfPath = "/home/rachel/Downloads/ecare/regression/GBTs/model/testModel1_0"
cv = PipelineModel.load(rfPath)

predictions = cv.transform(test)
#print(predictions.select("longMonths", "prediction", *featuresCols))

evaluator = RegressionEvaluator(metricName="rmse", labelCol = "label",predictionCol = "prediction")

rmse = evaluator.evaluate(predictions)
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("RMSE on our test set: %g" % rmse)
print("r2 on our test set: %g" % r2)
print("mae on our test set: %g" % mae)
