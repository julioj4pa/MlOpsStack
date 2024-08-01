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
        .appName("pycaret-MlFlow") \
        .config("spark.driver.extraLibraryPath","/mnt/jars") \
        .config("spark.executor.extraLibraryPath","/mnt/jars") \
        .config("spark.driver.extraClassPath","/mnt/jars/spark-extension_2.12-2.11.0-3.5.jar") \
        .config("spark.executor.extraClassPath","/mnt/jars/spark-extension_2.12-2.11.0-3.5.jar") \
        .config("spark.driver.extraJavaOptions","-verbose:class") \
        .config("spark.executor.extraJavaOptions","-verbose:class") \
        .getOrCreate()
os.environ['AWS_ACCESS_KEY_ID']="minio"
os.environ['AWS_SECRET_ACCESS_KEY']="minio123"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service.default.svc.cluster.local:9000"
os.environ['MLFLOW_GATEWAY_URI'] = "http://my-mlflow-tracking.mlflow.svc.cluster.local:80"
os.environ['MLFLOW_ENABLE_ASYNC_LOGGING'] = "True"
#https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html
from gresearch.spark import *

spark.install_pip_package("fugue==0.9.0.dev3","pycaret==3.3.1","xgboost","catboost","mlflow==2.11.3","joblib==1.3.0", "--cache-dir", "/mnt/python/lib/")
import mlflow
from mlflow.exceptions import MlflowException
mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = True
mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
experiment_name = "Class Diabets"
experiment_id=None
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
except AttributeError:
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")

mlflow.set_experiment(experiment_name)
#mlflow.autolog(log_input_examples=False, log_model_signatures=True, log_models=True, disable=False, exclusive=False, disable_for_unsupported_versions=False, silent=False)
from pycaret.datasets import get_data
diabetes = get_data('diabetes')

# init setup
from pycaret.classification import *
setup(experiment_name=experiment_name, log_experiment='mlflow',log_plots=True, data = diabetes, target = 'Class variable', n_jobs =1)
from pycaret.parallel import FugueBackend
best_models=compare_models(n_select=5,experiment_custom_tags=experiment_name, parallel=FugueBackend(spark),cross_validation=True)
mlflow.flush_async_logging()
print("Models \n")
print(best_models)
print("Display \n")
print(pull())
print("Metrics \n")
print(get_metrics())
df_metrics=pull()
df_metrics=df_metrics.rename(columns={"TT (Sec)": "TT_Sec"})
#for i in range(0, len(best_models)):
#        mlflow.set_experiment_tag(key=str(df_metrics.iloc[i,0]),value=str(df_metrics.iloc[i,0]))
#        for x, y in best_models[i].get_params().items():
#                mlflow.log_param(key=str(x),value=str(y))
#        for f in range(1, len(df_metrics.columns)):
#                mlflow.log_metric(key=str(df_metrics.columns[f]),value=float(df_metrics.iloc[i,f]))
#        mlflow.sklearn.save_model(best_models[i])
with mlflow.last_active_run() as parent_run:
        for i in range(0, len(best_models)):
                with mlflow.start_run(
                        run_name=str(df_metrics.iloc[i,0]),
                        experiment_id=experiment_id,
                        description="child",
                        nested=True,
                ) as child_run:
                        #mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
                        #mlflow.set_tag(df_metrics.iloc[i,0])
                        for x, y in best_models[i].get_params().items():
                                mlflow.log_param(key=str(x), value=str(y))
                        for f in range(1, len(df_metrics.columns)):
                                mlflow.log_metric(key=str(df_metrics.columns[f]),value=float(df_metrics.iloc[i,f]))
                        mlflow.sklearn.log_model(sk_model=best_models[i],artifact_path=str(experiment_id)+"/"+str(df_metrics.iloc[i,0]))
                        mlflow.sklearn.save_model(sk_model=best_models[i],path="s3://mlflow/"+str(experiment_id)+"/"+str(df_metrics.iloc[i,0]))

#save_model(best_model,'/tmp/best_model')
#print(mlflow.get_artifact_uri())
#print(best_model)
#mlflow.log_metric('Accuracy', float(best_model['Accuracy']))
#mlflow.sklearn.log_model(best_model)
#mlflow.log_param('Model_Param', str(best_model))
#mlflow.log_artifacts('/tmp/best_model', artifact_path="s3://mlflow/model")
#final_model = finalize_model(best_model)
#mlflow.sklearn.log_model(final_model, 'model')
#mlflow.client.create_registered_model('cc')
#final_model = finalize_model(best_model)
#mlflow.log_model(final_model, 'model')