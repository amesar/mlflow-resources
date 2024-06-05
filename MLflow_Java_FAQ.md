# MLflow FAQ - Java client

Note: this content has not been updated since MLflow 1.28.0 (Aug 11, 2022).

## MLflow Java Client

### MLflow Java Client Documentation

Unfortunately there is only the [Javadoc](https://mlflow.org/docs/latest/java_api/index.html).
There are no examples in [MLflow examples github](https://github.com/mlflow/mlflow/tree/master/examples).

You can find some Scala examples at [mlflow-examples - scala-spark](https://github.com/amesar/mlflow-examples/tree/master/scala/sparkml).

### MLflow Scala Client

MLflow has a Java client that can be accessed from Scala.

Sample Scala code using the Java client: [github.com/amesar/mlflow-examples/tree/master/scala/sparkml](github.com/amesar/mlflow-examples/tree/master/scala/sparkml).

### MLflow Java Feature Gap

* Since much of MLflow functionality is client-based and is written in Python, there is a feature gap for other languages.
* Standard MLflow features such as MLflow projects, models and  flavors are not supported for Java/Scala.
* This is principally due less demand for JVM-based ML training vs Python.
* You can save your native model as a raw artifact but cannot log it as a managed MLflow model.
* See item below.

### Does the Java client support MLflow projects, moels  and flavors?

No. With the Java client you have to save your models as un-managed artifacts using [logArtifact](https://mlflow.org/docs/latest/java_api/org/mlflow.client/MlflowClient.html#logArtifact-java.lang.String-java.io.File-). There is no concept of MLflow Python’s log_model (e.g. [mlflow.sklearn.log_model](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model) which implies flavors.
See example in [TrainWine.scala](https://github.com/amesar/mlflow-examples/blob/master/scala/sparkml/src/main/scala/org/andre/mlflow/examples/wine/sparkml/TrainWine.scala).

### Python set_experiment() equivalent

See [MLflowUtils.getOrCreateExperimentId](https://github.com/amesar/mlflow-examples/blob/master/scala/sparkml/src/main/scala/org/andre/mlflow/util/MLflowUtils.scala#L24).

```
// Return the ID of an experiment - create it if it doesn't exist
def getOrCreateExperimentId(client: MlflowClient, experimentName: String) = {
  try {
    client.createExperiment(experimentName)
  } catch {
    case e: org.mlflow.client.MlflowHttpException => { // statusCode 400
      client.getExperimentByName(experimentName).get.getExperimentId
    }
  }
}
```

### How do I score a model in Scala that was saved in Python?

Works only for SparkML (MLlib) models.

Read the model artifact with the `downloadArtifacts` method.
```
import org.apache.spark.ml.PipelineModel
import org.mlflow.client.MlflowClient

val client = new MlflowClient()
val modelPath = client.downloadArtifacts(runId, "spark-model/sparkml").getAbsolutePath
val model = PipelineModel.load(modelPath.replace("/dbfs","dbfs:"))
val predictions = model.transform(data)
```

### How do I score a model in Python that was saved in Scala?

Works only for SparkML (MLlib) models.

Do the same as above using the Python `MlflowClient.download_artifacts` method.

### Searching for MLflow objects

MLflow allows you to search for a subset of MLflow objects. The MLflow search filter is a simplified version of the SQL WHERE clause.

You can search for the following MLflow objects:
* Runs
* Registered Models
* Versions of a Registered Model

#### Search runs

General
* [Search — MLflow 1.26.1 documentation](https://mlflow.org/docs/latest/search-syntax.html)  - Detailed description of the filter syntax for run search.

[mlflow package](https://mlflow.org/docs/latest/python_api/mlflow.html)
* search_runs - returns a list of Pandas DataFrames.
* Note: no description or link of/to filter syntax. Just some examples.

[mlflow.client package](https://mlflow.org/docs/latest/python_api/mlflow.client.html)
* [search_runs](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_runs) - returns a paged list of [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) objects.
* Note: no description or link of/to filter syntax. Just some examples.

#### Search registered models and versions

[mlflow.client package](https://mlflow.org/docs/latest/python_api/mlflow.client.html)
* [search_registered_models](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models) - returns a paged list of RegisteredModel objects.
  * filter_string – Filter query string, defaults to searching all registered models. Currently, it supports only a single filter condition as the name of the model, for example, name = 'model_name' or a search expression to match a pattern in the registered model name. For example, name LIKE 'Boston%' (case sensitive) or name ILIKE '%boston%'.
* [search_model_versions](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_model_versions) - returns a paged list of ModelVersion objects.
  * filter_string – A filter string expression. Currently, it supports a single filter condition either a name of model like name = 'model_name' or run_id = '...'.

#### Search experiments
Available now in MLflow 1.28.0.
