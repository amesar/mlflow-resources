# MLflow FAQ


## How do I copy an experiment or run from one MLflow tracking server to another?

I would like to:
* Back up my experiments, runs or registered models.
* Copy them into another MLflow tracking server (i.e. Databricks workspace).

There is no official MLflow support for this. 

However, there is an unofficial tool that can export/import an experiment/run with caveats for Databricks MLflow using the [public MLflow API](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html).

See https://github.com/amesar/mlflow-export-import.

TLDR:
* It works well for OSS MLflow. 
* Unfortunately for Databricks MLflow, there is currently no API call to export notebook revisions (each run has a pointer to a notebook revision). However, the model artifacts and metadata are correctly exported.

## How do I find the best run of an experiment?
Use the [MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs) method. A simple example is shown below where we look for the run with the lowest `RMSE` value.

```
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best_run = runs[0]
print(best_run.info.run_id, best_run.data.metrics["rmse"])
fc9337b500054dc7869f7611a74e3c62', 0.7367947360663162
```

For a full-fledged version that accounts for nested runs see [Find best run for experiment](https://github.com/amesar/mlflow-tools/blob/master/mlflow_tools/tools/README.md#find-best-run-for-experiment) and [best_run.py](https://github.com/amesar/mlflow-tools/blob/master/mlflow_tools/tools/best_run.py).

## How do I dump all experiment or run information?

I would like to see all the information of an experiment or run.

You can use the mlflow CLI command to get basic run details.
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
If you want to get a consolidated experiment report with artifacts details use the custom tools below. 


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




## What are the MLflow system run tags?

Tag keys that start with mlflow. are reserved for internal use. See [System Tags](https://mlflow.org/docs/latest/tracking.html#system-tags) documentation page.

Column legend:
* python - if you run with normal Python 
* mlflow run - If you run with mlflow runcommand
* notebook - If run as a Databricks notebook

As of MLflow 1.9.1.
```
+--------------------------------------+----------+--------------+------------+
| tag                                  | python   | mlflow run   | notebook   |
|--------------------------------------+----------+--------------+------------|
| mlflow.log-model.history             | Y        | Y            | Y          |
| mlflow.runName                       | Y        | Y            | Y          |
| mlflow.source.name                   | Y        | Y            | Y          |
| mlflow.source.type                   | Y        | Y            | Y          |
| mlflow.user                          | Y        | Y            | Y          |
| mlflow.source.git.commit             | Y        | Y            | _          |
| mlflow.gitRepoURL                    | _        | Y            | _          |
| mlflow.project.backend               | _        | Y            | _          |
| mlflow.project.entryPoint            | _        | Y            | _          |
| mlflow.project.env                   | _        | Y            | _          |
| mlflow.databricks.cluster.id         | _        | _            | Y          |
| mlflow.databricks.notebookID         | _        | _            | Y          |
| mlflow.databricks.notebookPath       | _        | _            | Y          |
| mlflow.databricks.notebookRevisionID | _        | _            | Y          |
| mlflow.databricks.webappURL          | _        | _            | Y          |
+--------------------------------------+----------+--------------+------------+
```

## How do I create an MLflow run from a model I have trained elsewhere?

```
import mlflow
import cloudpickle

with open("data/model.pkl", "rb") as f:
    model = cloudpickle.load(f)
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "sklearn-model")
model_uri = f"runs:/{run.info.run_id}/sklearn-model"
model = mlflow.sklearn.load_model(model_uri)
```


## How do I run a docker container with the MLflow scoring server on my laptop?

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

## MLflow Database Schema (MySQL)

See [schema_mlflow_1.15.0.sql](schema_mlflow_1.15.0.sql).

## Where do I find more of Andre's MLflow stuff?

See:
* https://github.com/amesar/mlflow-examples - examples of many different ML frameworks (sklearn, SparkML, Keras/TensorFlow, etc.) and Scala examples.
* https://github.com/amesar/mlflow-export-import - Tools to export and import MLflow runs, experiments or registered models from one tracking server to another.
* https://github.com/amesar/mlflow-tools - tools and utilities such as export/import runs.
* https://github.com/amesar/mlflow-spark-summit-2019 - code for Spark Summit 2019 tutorial session. Dated.
