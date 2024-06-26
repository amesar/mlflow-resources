# MLflow FAQ

Last updated: 2024-06-06.

Other pages:
* [README.md](README.md)
* [MLflow_Java_FAQ.md](MLflow_Java_FAQ.md)

## General MLflow

### How do I copy an MLflow object from one MLflow tracking or registry server to another?

I would like to:
* Back up my registered models, model versions, experiments or runs.
* Copy them to another MLflow tracking server (i.e. Databricks workspace).

The [mlflow-export-import](https://github.com/mlflow/mlflow-export-import) tool can handle these use case with caveats due to limitations in the MLflow and Databricks APIs.

Resource links: 
* https://github.com/mlflow/mlflow-export-import
* [Copy MLflow objects between workspaces](https://docs.databricks.com/en/mlflow/migrate-mlflow-objects.html) - Databricks documentation
  * To import or export MLflow objects to or from your Databricks workspace, you can use the community-driven open source project MLflow Export-Import to migrate MLflow experiments, models, and runs between workspaces.

### How do I backup up my MLflow objects?

See above.

### How do I create an MLflow run from a model I have trained elsewhere?

Steps:
* Create a new MLflow run.
* Load the native model.
* Log the the native model as an MLflow model in the MLflow run.

Also see: [MLflow Model Registry — Registering a Saved Model](https://mlflow.org/docs/latest/model-registry.html#registering-a-saved-model) - MLflow documentation.

#### Scikit-learn example

```
import mlflow
import cloudpickle

with open("scikit-model.pkl", "rb") as f:
    model = cloudpickle.load(f)
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")
```

#### TensorFlow Keras example

See [tensorflow.keras.models.load_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model).

```
import mlflow
import tensorflow as tf

model = tf.keras.models.load_model("tf_keras_model")
with mlflow.start_run() as run:
    mlflow.keras.log_model(model, "model")
```


### How to get the run_id of your run

Use `run_id` instead of `run_uuid` which is deprecated. No need to use `active_run` either.

Use this idiom:
```
with mlflow.start_run() as run:
   run_id = run.info.run_id
```
Instead of this:
```
with mlflow.start_run():
   run_id = mlflow.active_run().info.run_uuid
```

### How do I find the best run of an experiment?
Use the [MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_runs) method. A simple example is shown below where we look for the run with the lowest `RMSE` value.

```
import mlflow
client = mlflow.MlflowClient()

def get_best_run(experiment_id)
    runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
    return runs[0]

best_run = get_best_run("3532228")
print(best_run.info.run_id, best_run.data.metrics["rmse"])

fc9337b500054dc7869f7611a74e3c62', 0.7367947360663162
```

For a full-fledged version that accounts for nested runs see [Find best run for experiment](https://github.com/amesar/mlflow-tools/blob/master/mlflow_tools/tools/README.md#find-best-run-for-experiment) and [best_run.py](https://github.com/amesar/mlflow-tools/blob/master/mlflow_tools/tools/best_run.py).

### How do I find the last run of an experiment?

Use the [MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_runs) method.
```
import mlflow
client = mlflow.MlflowClient()

def get_last_run(experiment_id):
    runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
    return runs[0]
```
### How do I change a run's attributes?

As a system of record, for the most part an MLflow is immutable. The only run fields you can change are:
 * [update_run](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.update_run) - update the run.info.status
 * [set_tag()](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.set_tag) - change a run tag
 * [log_artifact()](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.log_artifact) - add an artifact to an existing run

### How do I dump run, experiment or registered model information?

I would like to see all the information of an experiment or run.

#### MLflow CLI

The mlflow CLI can dump run information but not experiment or registered model information.

```
mlflow runs describe --run-id
```

```
{
  "info": {
    "artifact_uri": "/Users/andre/work/mlflow_server/local_mlrun/mlruns/6/b8ad03e448834d95b10cd2bb4d93a2cb/artifacts",
    "end_time": 1596573698775,
    "experiment_id": "6",
    "lifecycle_stage": "active",
    "run_id": "b8ad03e448834d95b10cd2bb4d93a2cb",
    "run_uuid": "b8ad03e448834d95b10cd2bb4d93a2cb",
    "start_time": 1596573697304,
    "status": "FINISHED",
    "user_id": "andre"
  },
  "data": {
    "metrics": {
      "rmse": 0.7643663416167976,
      "r2": 0.25411980734506334,
      "mae": 0.5866955768114198
    },
    "params": {
      "max_depth": "4",
      "max_leaf_nodes": "32"
    },
    "tags": {
      "mlflow.user": "andre",
      "mlflow.source.name": "main.py",
      "mlflow.source.type": "LOCAL",
      "mlflow.source.git.commit": "180c807a1e0f283d14befcac586f41d899a5bae4",
      "mlflow.runName": "train.sh",
      "data_path": "../../data/train/wine-quality-white.csv",
      "run_origin": "train.sh",
      "version.mlflow": "1.10.0",
      "version.sklearn": "0.20.2",
      "version.platform": "Darwin-19.4.0-x86_64-i386-64bit",
      "version.python": "3.7.6",
      "mlflow.log-model.history": "[{\"run_id\": \"b8ad03e448834d95b10cd2bb4d93a2cb\", \"artifact_path\": \"sklearn-model\", \"utc_time_created\": \"2020-08-04 20:41:37.711168\", \"flavors\": {\"python_function\": {\"model_path\": \"model.pkl\", \"loader_module\": \"mlflow.sklearn\", \"python_version\": \"3.7.6\", \"env\": \"conda.yaml\"}, \"sklearn\": {\"pickled_model\": \"model.pkl\", \"sklearn_version\": \"0.20.2\", \"serialization_format\": \"cloudpickle\"}}}, {\"run_id\": \"b8ad03e448834d95b10cd2bb4d93a2cb\", \"artifact_path\": \"onnx-model\", \"utc_time_created\": \"2020-08-04 20:41:38.339746\", \"flavors\": {\"python_function\": {\"loader_module\": \"mlflow.onnx\", \"python_version\": \"3.7.6\", \"data\": \"model.onnx\", \"env\": \"conda.yaml\"}, \"onnx\": {\"onnx_version\": \"1.7.0\", \"data\": \"model.onnx\"}}}]",
      "version.onnx": "1.7.0",
      "output_path": "out_run_id.txt"
    }
  }
}
```

#### mlflow-tools

The dump commands of mlflow-tools provides you information in JSON, YAML and text formats for runs. For experiments and models only JSON is supported.
The value-add for dumping runs with the tool is that it can dump all the of run's artifact information.
See [Dump experiment or run as text](https://github.com/amesar/mlflow-tools/blob/master/mlflow_tools/tools/README.md#dump-experiment-or-run-as-text).

```
RunInfo:
  experiment_name: sklearn
  artifact_uri: /opt/mlflow/server/mlruns/6/b8ad03e448834d95b10cd2bb4d93a2cb/artifacts
  experiment_id: 6
  lifecycle_stage: active
  run_id: b8ad03e448834d95b10cd2bb4d93a2cb
  run_uuid: b8ad03e448834d95b10cd2bb4d93a2cb
  status: FINISHED
  user_id: andre
  start_time: 2020-08-04_20:41:37   1596573697304
  end_time:   2020-08-04_20:41:38   1596573698775
  _duration:  1.471 seconds
Params:
  max_depth: 4
  max_leaf_nodes: 32
Metrics:
  mae: 0.5866955768114198
  r2: 0.25411980734506334
  rmse: 0.7643663416167976
Tags:
  data_path: ../../data/train/wine-quality-white.csv
  mlflow.log-model.history: [{"run_id": "b8ad03e448834d95b10cd2bb4d93a2cb", "artifact_path": "sklearn-model", "utc_time_created": "2020-08-04 20:41:37.711168", "flavors": {"python_function": {"model_path": "model.pkl", "loader_module": "mlflow.sklearn", "python_version": "3.7.6", "env": "conda.yaml"}, "sklearn": {"pickled_model": "model.pkl", "sklearn_version": "0.20.2", "serialization_format": "cloudpickle"}}}, {"run_id": "b8ad03e448834d95b10cd2bb4d93a2cb", "artifact_path": "onnx-model", "utc_time_created": "2020-08-04 20:41:38.339746", "flavors": {"python_function": {"loader_module": "mlflow.onnx", "python_version": "3.7.6", "data": "model.onnx", "env": "conda.yaml"}, "onnx": {"onnx_version": "1.7.0", "data": "model.onnx"}}}]
  mlflow.runName: train.sh
  mlflow.source.git.commit: 180c807a1e0f283d14befcac586f41d899a5bae4
  mlflow.source.name: main.py
  mlflow.source.type: LOCAL
  mlflow.user: andre
  output_path: out_run_id.txt
  run_origin: train.sh
  version.mlflow: 1.10.0
  version.onnx: 1.7.0
  version.platform: Darwin-19.4.0-x86_64-i386-64bit
  version.python: 3.7.6
  version.sklearn: 0.20.2
Artifacts:
  Artifact 1/3 - level 0:
    path: onnx-model
    Artifact 1/3 - level 1:
      path: onnx-model/MLmodel
      bytes: 293
    Artifact 2/3 - level 1:
      path: onnx-model/conda.yaml
      bytes: 144
    Artifact 3/3 - level 1:
      path: onnx-model/model.onnx
      bytes: 2796
  Artifact 2/3 - level 0:
    path: plot.png
    bytes: 32649
  Artifact 3/3 - level 0:
    path: sklearn-model
    Artifact 1/3 - level 1:
      path: sklearn-model/MLmodel
      bytes: 357
    Artifact 2/3 - level 1:
      path: sklearn-model/conda.yaml
      bytes: 150
    Artifact 3/3 - level 1:
      path: sklearn-model/model.pkl
      bytes: 4893
Total: bytes: 41282 artifacts: 7
```

### What are the MLflow system run tags?

Overview
* MLflow runs can be created in a various number of ways - OSS (as project or no project) or Databricks (job, notebook UI, Repo).
* MLflow runs can be created in a various number of ways:
  * Open source (OSS) - as project or no project.
  * Databricks - notebook UI,  notebook job or Databricks Repo.
* A run has two sets of system tags:
  * Shared general tags.
  * Tags specific to the run type.
* System tag keys start with `mlflow.` and are reserved for internal use. Databricks system tags start with `mlflow.databricks`.
* See the MLflow System Tags documentation page and https://github.com/mlflow/mlflow/blob/master/mlflow/utils/mlflow_tags.py.

Column legend for the different kinds of runs:
* OSS MLflow
  * `Python` - without MLproject. Run is created using plain python command without an MLproject file.
  * `Project` -  with MLproject. Run is created with an MLproject file using the `mlflow run` command.
* Databricks
  * `NB UI` - Notebook UI. Run is created from a Databricks notebook (either notebook or workspace experiment) in the UI.
  * `NB job` - Notebook job. Run is created from a Databricks notebook job (must be a workspace experiment).
  * `Repo` - Run is created from a Databricks Repo notebook in the UI.

As of MLflow 1.30.0:
```
+----------------------------------------+--------+---------+-------+--------+------+
| Tag                                    | Python | Project | NB UI | NB job | Repo |
+----------------------------------------+--------+---------+-------+--------+------+
| mlflow.databricks.cluster.id           | -      | -       | y     | y      | y    |
| mlflow.databricks.cluster.info         | -      | -       | y     | y      | y    |
| mlflow.databricks.cluster.libraries    | -      | -       | y     | y      | y    |
| mlflow.databricks.jobID                | -      | -       | -     | y      | -    |
| mlflow.databricks.jobRunID             | -      | -       | -     | y      | -    |
| mlflow.databricks.jobType              | -      | -       | -     | y      | -    |
| mlflow.databricks.notebook.commandID   | -      | -       | y     | y      | y    |
| mlflow.databricks.notebookID           | -      | -       | y     | -      | y    |
| mlflow.databricks.notebookPath         | -      | -       | y     | -      | y    |
| mlflow.databricks.notebookRevisionID   | -      | -       | y     | -      | -    |
| mlflow.databricks.webappURL            | -      | -       | y     | y      | -    |
| mlflow.databricks.workspaceID          | -      | -       | y     | y      | y    |
| mlflow.databricks.workspaceURL         | -      | -       | y     | y      | y    |
| mlflow.databricks.gitRepoCommit        | -      | -       | -     | -      | y    |
| mlflow.databricks.gitRepoProvider      | -      | -       | -     | -      | y    |
| mlflow.databricks.gitRepoReference     | -      | -       | -     | -      | y    |
| mlflow.databricks.gitRepoReferenceType | -      | -       | -     | -      | y    |
| mlflow.databricks.gitRepoRelativePath  | -      | -       | -     | -      | y    |
| mlflow.databricks.gitRepoStatus        | -      | -       | -     | -      | y    |
| mlflow.databricks.gitRepoUrl           | -      | -       | -     | -      | y    |
| mlflow.gitRepoURL                      | -      | y       | -     | -      | -    |
| mlflow.log-model.history               | y      | y       | -     | -      | y    |
| mlflow.project.backend                 | -      | y       | -     | -      | -    |
| mlflow.project.entryPoint              | -      | y       | -     | -      | -    |
| mlflow.project.env                     | -      | y       | -     | -      | -    |
| mlflow.runName                         | y      | y       | y     | y      | y    |
| mlflow.source.git.commit               | y      | y       | -     | -      | -    |
| mlflow.source.git.repoURL              | -      | y       | -     | -      | -    |
| mlflow.source.name                     | y      | y       | y     | y      | y    |
| mlflow.source.type                     | y      | y       | y     | y      | y    |
| mlflow.user                            | y      | y       | -     | -      | y    |
+----------------------------------------+--------+---------+-------+--------+------+
```

### What’s the difference between log_model and save_model?

`log_model` saves a model in a location relative to the experiment’s [artifact location](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment). With `save_model` you specify an absolute path. Both methods are flavor-specific. Definitely prefer `log_model`. In order to use the standard `load_model` method you have to use `log_model`.

See the [sklearn](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html) flavor sample methods:
  * [log_model](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model)
  * [save_model](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.save_model)
  * [load_model](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model)

### Examples of MLflow Custom Python Models

* [Custom Python Models](https://mlflow.org/docs/latest/models.html#custom-python-models) - MLflow documentation
* Example: https://github.com/mlflow/mlflow/blob/master/examples/prophet/train.py

### How do I run a docker container with the MLflow scoring server on my laptop?

**Launch the MLflow tracking server in window 1**
```
mlflow server --host 0.0.0.0 --port 5000  \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root $artifact_store $PWD/mlruns
```

**Create an experiment run in window 2**
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sklearn \
  -P max_depth=2 -P max_leaf_nodes=32 \
  -P model_name=sklearn_wine \
  --experiment-name=sklearn_wine
```

**Launch the MLflow scoring server in window 2**

Plain docker
```
mlflow models build-docker --model-uri models:/sklearn_wine/1 --name dk-wine-sklearn
docker run --p 5001:8080 dk-wine-sklearn
```

SageMaker docker container in local mode
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-sklearn
mlflow sagemaker run-local -m models:/sklearn_wine/1  -p 5001 --image sm-wine-sklearn
```


**Send a prediction to MLflow scoring server in window 3**
```
curl  http://localhost:5001/invocations  \
  -H "Content-Type:application/json" \
  -d '{ "columns":   [ "alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity" ],
         "data": [
            [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ],
            [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ] ] }'
```
Response
```
[5.46875, 5.1716417910447765]
```

### MLflow Database Schema

|Version | Database |
|-----|-------|
| [schema_mlflow_2.13.0.sql](database_schemas/sqlite/schema_mlflow_2.13.0.sql) | Sqlite |
| [schema_mlflow_2.9.2.sql](database_schemas/postgres/schema_mlflow_2.9.2.sql) | PostgreSQL |
| [schema_mlflow_2.9.2.sql](database_schemas/sqlite/schema_mlflow_2.9.2.sql) | Sqlite |
| [schema_mlflow_2.4.1.sql](database_schemas/mysql/schema_mlflow_2.4.1.sql) | MySQL |
| [schema_mlflow_2.3.1.sql](database_schemas/mysql/schema_mlflow_2.3.1.sql) | MySQL |
| [schema_mlflow_2.2.2.sql](database_schemas/mysql/schema_mlflow_2.2.2.sql) | MySQL |
| [schema_mlflow_2.1.1.sql](database_schemas/mysql/schema_mlflow_2.1.1.sql) | MySQL |
| [schema_mlflow_1.30.0.sql](database_schemas/mysql/schema_mlflow_1.30.0.sql) | MySQL |
| [schema_mlflow_1.28.0.sql](database_schemas/mysql/schema_mlflow_1.28.0.sql) | MySQL |
| [schema_mlflow_1.22.0.sql](database_schemas/mysql/schema_mlflow_1.22.0.sql) | MySQL |



## Databricks MLflow

### How do I externally access Databricks MLflow?

There are several ways to access a Databricks MLflow tracking server from outside Databricks.

See the Databricks documentation page `Access the MLflow tracking server from outside Databricks` - [AWS](https://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html) or [Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/access-hosted-tracking-server).

1. With `.~/databrickscfg` and no profile specified. The host and token are picked up from the DEFAULT profile.
```
export MLFLOW_TRACKING_URI=databricks
```
2. By specifying a profile in  `~/.databrickscfg`.
```
export MLFLOW_TRACKING_URI=databricks://MY_PROFILE
```
3. Without using `~/.databrickscfg` file or to override `~/.databrickscfg` profiles.
```
export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=https://myshard.cloud.databricks.com
export DATABRICKS_TOKEN=MY_TOKEN
```

### Databricks MLflow operation modes

There are several ways you can train and track your model with Databricks MLflow tracking server.


![](images/databricks/MLflow_Operation_Modes.png)

Three main components are:
* Coding - where you write your code - non-Databricks (IDE or Jupyter notebook) or Databricks notebook.
* Training - where the training is executed.
* Tracking - on which MLflow Tracking Server the run is logged.

Options:
* Option 0 - Coding, training and tracking are all done in a non-Databricks open source environment (e.g laptop or on-prem).
* Option 1 - Coding, training and tracking are all done in a Databricks environment.
* Option 2 - Coding and training are done on laptop and tracking is done on Databricks.
* Option 3 - Coding is done on laptop but training and tracking are done on Databricks.

### DBFS vs Fusemount file paths with MLflow

MLflow API methods do not understand DBFS file paths. You need to use the Fusemount version starting with `/dbfs` instead of `dbfs:`.

Use:
```
tf.keras.models.save_model(model, "/dbfs/mymodel.keras")
```
instead of:
```
tf.keras.models.save_model(model, "dbfs:/mymodel.keras")
```

### Specify non-DBFS artifact location for an experiment

In Databricks MLflow, the default location for artifacts is DBFS. This location is specified at the experiment level. You can provide an alternate artifact location assuming there is a corresponding artifact plugin.

There are two major limitations to custom non-DBFS locations:
* The artifact will not appear in the UI.
* You cannot promote the model artifact to the model registry.


Open source MLflow documentation

* [Artifact Stores](https://mlflow.org/docs/latest/client.html#artifact-stores) - Supported plugins: Amazon S3 and S3-compatible storage, Azure Blob Storage, Google Cloud Storage, FTP server, SFTP Server, NFS and HDFS.

* [mlflow.create_experiment](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment) - Python API.

Create workspace experiment - Databricks MLflow documentation

* [AWS](https://docs.databricks.com/applications/mlflow.client.html#create-workspace-experiment) - Databricks supports DBFS, S3, and Azure Blob storage artifact locations.

* [Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow.client#create-workspace-experiment) - Azure Databricks supports DBFS and Azure Blob storage artifact locations. Note there is no ADLS support.

### What are the Databricks MLflow API rate limits?

#### MLflow Tracking Server rate limits

* [AWS](https://docs.databricks.com/dev-tools/api/latest/mlflow.html#rate-limits) - [Azure](https://docs.microsoft.com/en-us/azure/databricks/dev-tools/api/latest/mlflow#rate-limits) - rate limits - Databricks documentation
* Low throughput experiment management (list, update, delete, restore): 7 qps
* Search runs: 7 qps
* Log batch: 47 qps
* All other APIs: 127 qps

#### MLflow Serverless Model Serving on Databricks (v2)

Limitations ([AWS](https://docs.databricks.com/mlflow/serverless-real-time-inference.html#limitations) - [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow/serverless-real-time-inference#limitations) - GCP (not yet)):
* Payload size limit of 16 MB per request.
* Default limit of 200 QPS of scoring requests per workspace enrolled. You can increase this limit to up to 3000 QPS per workspace by reaching out to your Databricks support contact.
* Best effort support on less than 100 millisecond latency overhead and availability.

#### MLflow Classic Model Serving on Databricks (v1) - deprecated

Rate limits [AWS](https://docs.databricks.com/applications/mlflow/model-serving.html#mlflow-model-serving-on-databricks) - [Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-serving):
* Target throughput is 20 QPS.
* Target availability is 99.5%, although no guarantee is made as to either.
* Payload size limit of 16 MB per request.

#### MLflow and Databricks Repos Limitations

See the [CI/CD and MLOps](https://docs.databricks.com/repos/limits.html#cicd-and-mlops) section of the [Limits & FAQ for Git integration with Databricks Repos](https://docs.databricks.com/repos/limits.html) documentation page.

### How do I log a Delta table and version in MLflow?

You can log the Delta tables you use for training in the `sparkDatasourceInfo` tag.

Code:
```
import mlflow.spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.config("spark.jars.packages", "org.mlflow.mlflow-spark").getOrCreate()
mlflow.spark.autolog()
```

This will automatically log all Delta tables and their versions in the `sparkDatasourceInfo` tag.  An example of the tag value:

```
path=dbfs:REDACTED_LOCAL_PART@databricks.com/mlflow_demo/wine-quality.csv,format=text
```

Sample Databricks notebook: [Tracking ML Model Training with MLflow and Delta Lake
](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-delta-training.html).

## Where do I find more of Andre's MLflow stuff?

**Github code**
* https://github.com/mlflow/mlflow-export-import - Tools to export and import MLflow runs, experiments or registered models from one tracking server to another.
* https://github.com/amesar/mlflow-examples - Examples of many different Python ML frameworks (sklearn, SparkML, Keras/TensorFlow, etc.) and some Scala examples.
* https://github.com/amesar/mlflow-tools - Tools and utilities such as export/import runs.
* https://github.com/amesar/mlflow-tensorflow-serving - creates a TensorFlow Serving Docker container with an embedded MLflow Keras TensorFlow model.
* https://github.com/amesar/mlflow-pluggable-scoring-server - Exploratory POC to plug in custom request or response payloads for MLflow serving.
* https://github.com/amesar/mlflow-spark-summit-2019 - Code for Spark Summit 2019 tutorial session - obviously dated.
* https://github.com/amesar/mleap-sampler - All things MLeap (with MLflow too)- Scala and Python.

**Slides**
* [MLflow Model Serving](https://databricks.com/session_na21/mlflow-model-serving) - DAIS 2021 - 2021-05-27 - [slideshare](https://www.slideshare.net/amesar0/mlflow-model-serving-dais-2021)
* [ONNX and MLflow](https://www.slideshare.net/amesar0/onnx-overview-and-mlflow) - Spark Meetup - 2020-02-26

