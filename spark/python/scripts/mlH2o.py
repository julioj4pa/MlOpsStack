import pyspark
import os
#import json
#import tempfile
#from pathlib import Path
import sys
sys.path.append('/mnt/python/lib/')
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import *

spark = SparkSession \
        .builder \
        .appName("SparklingWaterApp") \
        .config("spark.jars.package","ai.h2o:sparkling-water-package_2.12:3.46.0.1-1-3.5") \
        .config("spark.scheduler.minRegisteredResourcesRatio","1") \
        .config("spark.locality.wait","10s") \
        .config("spark.driver.extraClassPath","-Dhdp.version=current") \
        .config("spark.executor.extraClassPath","-Dhdp.version=current") \
        .config("spark.driver.extraLibraryPath","/mnt/jars") \
        .config("spark.executor.extraLibraryPath","/mnt/jars") \
        .config("spark.driver.extraClassPath","/mnt/jars/iceberg-spark-runtime-3.5_2.12-1.5.0.jar:/mnt/jars/url-connection-client-2.25.18.jar:/mnt/jars/hadoop-aws-3.3.4.jar:/mnt/jars/aws-java-sdk-1.12.688.jar:/mnt/jars/aws-java-sdk-core-1.12.688.jar:/mnt/jars/aws-java-sdk-dynamodb-1.12.688.jar:/mnt/jars/aws-java-sdk-kms-1.12.688.jar:/mnt/jars/aws-java-sdk-s3-1.12.688.jar:/mnt/jars/bundle-2.25.18.jar:/mnt/jars/spark-extension_2.12-2.11.0-3.5.jar") \
        .config("spark.executor.extraClassPath","/mnt/jars/iceberg-spark-runtime-3.5_2.12-1.5.0.jar:/mnt/jars/url-connection-client-2.25.18.jar:/mnt/jars/hadoop-aws-3.3.4.jar:/mnt/jars/aws-java-sdk-1.12.688.jar:/mnt/jars/aws-java-sdk-core-1.12.688.jar:/mnt/jars/aws-java-sdk-dynamodb-1.12.688.jar:/mnt/jars/aws-java-sdk-kms-1.12.688.jar:/mnt/jars/aws-java-sdk-s3-1.12.688.jar:/mnt/jars/bundle-2.25.18.jar:/mnt/jars/spark-extension_2.12-2.11.0-3.5.jar") \
        .config("spark.driver.extraJavaOptions","-Divy.cache.dir=/mnt/jars -Divy.home=/tmp -verbose:class") \
        .config("spark.executor.extraJavaOptions","-verbose:class") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio-service.default.svc.cluster.local:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minio") \
        .config("spark.hadoop.fs.s3a.secret.key", "minio123") \
        .config("spark.hadoop.fs.s3a.path.style.access", True) \
        .config("spark.hadoop.fs.s3a.fast.upload", True) \
        .config("spark.hadoop.fs.s3a.multipart.size", 104857600) \
        .config("fs.s3a.connection.maximum", 100) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.example", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.example.type", "hadoop") \
        .config("spark.sql.catalog.example.s3.endpoint", "http://minio-service.default.svc.cluster.local:9000") \
        .config("spark.sql.catalog.example.warehouse", "s3a://lakehouse/development/iceberg/") \
        .getOrCreate()
os.environ['AWS_ACCESS_KEY_ID']="minio"
os.environ['AWS_SECRET_ACCESS_KEY']="minio123"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service.default.svc.cluster.local:9000"
os.environ['MLFLOW_GATEWAY_URI'] = "http://my-mlflow-tracking.mlflow.svc.cluster.local:80"
os.environ['MLFLOW_ENABLE_ASYNC_LOGGING'] = "True"


from gresearch.spark import *

spark.install_pip_package("mlflow==2.11.3","joblib==1.3.0", "--cache-dir", "/mnt/python/lib/")
import mlflow
from mlflow.exceptions import MlflowException
mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = True
mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
experiment_name = "Powe Plant"
experiment_id=None
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
except AttributeError:
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")

mlflow.set_experiment(experiment_name)


from pysparkling import *
import h2o
hc = H2OContext.getOrCreate()


powerplant_df = spark.read.option("inferSchema", "true").csv("s3a://h2o/powerplant_output.csv", header=True)
#powerplant_df=powerplant_df.withColumnRenamed("HourlyEnergyOutputMW", "label")
splits = powerplant_df.randomSplit([0.8, 0.2], seed=1)
train = splits[0]
for_predictions = splits[1]


from pyspark.ml.feature import SQLTransformer
from pysparkling.ml import H2OAutoML
from pyspark.ml import Pipeline

temperatureTransformer = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE TemperatureCelcius > 10")

automlEstimator = H2OAutoML(maxModels=2, labelCol="HourlyEnergyOutputMW", seed=1)

pipeline = Pipeline(stages=[temperatureTransformer, automlEstimator])

# Fit AutoML model
model = pipeline.fit(train)

# Generate predictions using fitted model
predicted = model.transform(for_predictions)

predicted.show()

leaderboard =automlEstimator.getLeaderboard("ALL")
leaderboard.show(truncate = False)


models = automlEstimator.getAllModels()
for model in models:
    #print(model.getCrossValidationScoringHistory())
    #print(model.getModelDetails())
    model.transform(for_predictions).show(truncate = False)
    print(model.getModelCategory())
    print(model.getFeatureTypes())
    if(model.getFeatureImportances() is not None):
       print(model.getFeatureImportances().show())
    if(model.getScoringHistory() is not None):
       print(model.getScoringHistory().show())
    print(model.getDomainValues())
    print(model.getTrainingParams())
    print(model.getTrainingMetrics())
    print(model.getCrossValidationModels())
    print(model.getCrossValidationMetrics())
    print(model.getCurrentMetrics())
    if(model.getCoefficients() is not None):
       print(model.getCoefficients().show)
    print(model.getValidationMetrics())


mlflow.flush_async_logging()
with mlflow.last_active_run() as parent_run:
    for i in range(0, len(models)):
        with mlflow.start_run(
            run_name=str(models[i].getTrainingParams()['model_id']),
            experiment_id=experiment_id,
            description="child",
            nested=True,
        ) as child_run:
            #mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
            #mlflow.set_tag(df_metrics.iloc[i,0])
            for x, y in models[i].getTrainingParams().items():
                mlflow.log_param(key=str(x), value=str(y))
            for x, y in models[i].getCurrentMetrics().items():
                mlflow.log_metric(key=str(x), value=float(y))
            mlflow.h2o.log_model(sk_model=models[i],artifact_path=str(experiment_id)+"/"+str(models[i].getTrainingParams()['model_id']))
            mlflow.h2o.save_model(sk_model=models[i],path="s3://mlflow/"+str(experiment_id)+"/"+str(models[i].getTrainingParams()['model_id']))