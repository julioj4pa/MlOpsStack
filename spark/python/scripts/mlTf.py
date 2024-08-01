import pyspark
import os
import json
import tempfile
from pathlib import Path
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
#os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service.default.svc.cluster.local:9000"
#https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html
import mlflow
mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = True
mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
experiment_name = "tf"
mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog()
from gresearch.spark import *


spark.install_pip_package("mlflow","termcolor", "tensorflow==2.16.1", "spark_tensorflow_distributor", "joblib==1.3.0", "--cache-dir", "/mnt/python/lib/")

from spark_tensorflow_distributor import MirroredStrategyRunner

# Adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
def train():
    import tensorflow as tf
    import uuid

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def make_datasets():
        print(str(uuid.uuid4())+'mnist.npz')
        (mnist_images, mnist_labels), _ = \
            tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64))
        )
        dataset = dataset.repeat().batch(BATCH_SIZE)
        #dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return dataset

    def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'],
        )
        return model

    train_datasets = make_datasets()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=1, steps_per_epoch=5)
    model_dir = '/tmp/best_model'
    tf.saved_model.save(multi_worker_model, model_dir)
    mlflow.tensorflow.log_model(tf_saved_model_dir=model_dir, tf_signature_def_key="serving_default", tf_meta_graph_tags="serve", artifact_path=model_dir)

MirroredStrategyRunner(num_slots=1, use_gpu=False).run(train)




import pyspark
import os
import json
import tempfile
from pathlib import Path
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
#os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service.default.svc.cluster.local:9000"
#https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html

from gresearch.spark import *

spark.install_pip_package("tensorflow==2.16.1", "--cache-dir", "/mnt/python/lib/")
spark.install_pip_package("termcolor", "mlflow", "spark_tensorflow_distributor", "joblib==1.3.0", "--cache-dir", "/mnt/python/lib/")

from spark_tensorflow_distributor import MirroredStrategyRunner

# Adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
def train():
    import tensorflow as tf
    import uuid
    import mlflow

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def make_datasets():
        print(str(uuid.uuid4())+'mnist.npz')
        (mnist_images, mnist_labels), _ = \
            tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64))
        )
        dataset = dataset.repeat().batch(BATCH_SIZE)
        #dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return dataset

    def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'],
        )
        return model
    mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = True
    mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
    experiment_name = "testtaaa"
    mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
    mlflow.set_experiment(experiment_name)
    mlflow.tensorflow.autolog()
    train_datasets = make_datasets()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=1, steps_per_epoch=5)
    
MirroredStrategyRunner(num_slots=1, use_gpu=False).run(train)





import pyspark
import os
import json
import tempfile
from pathlib import Path
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
#os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service.default.svc.cluster.local:9000"
#https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html
import mlflow

mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = True
mlflow.set_tracking_uri('http://my-mlflow-tracking.mlflow.svc.cluster.local:80')
experiment_name = "testt"
mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
mlflow.set_experiment(experiment_name)
from gresearch.spark import *

spark.install_pip_package("tensorflow==2.16.1")
spark.install_pip_package("termcolor", "mlflow", "spark_tensorflow_distributor", "joblib==1.3.0", "--cache-dir", "/mnt/python/lib/")

from spark_tensorflow_distributor import MirroredStrategyRunner

# Adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
def train():
    import tensorflow as tf
    import uuid

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    def make_datasets():
        print(str(uuid.uuid4())+'mnist.npz')
        (mnist_images, mnist_labels), _ = \
            tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64))
        )
        dataset = dataset.repeat().batch(BATCH_SIZE)
        #dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return dataset

    def build_and_compile_cnn_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'],
        )
        return model

    train_datasets = make_datasets()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets, epochs=1, steps_per_epoch=5)
    model_dir = '/tmp/best_model'
    tf.saved_model.save(multi_worker_model, model_dir)
    print(mlflow.get_artifact_uri())
    mlflow.log_artifacts('/tmp/best_model', artifact_path="s3://mlflow/model")
MirroredStrategyRunner(num_slots=1, use_gpu=False).run(train)