import mlflow

mlflow.set_experiment('i3d-tuning')
with mlflow.start_run():
    mlflow.log_param('Epoch', 5)

    # mlflow.log_params() log more than 1 param at same time

    mlflow.log_metric('Acc', 78.8)
    # mlflow.log_metric()
    mlflow.log_metric('Loss', 0.01)

    mlflow.log_artifacts(
        'har/logs'
    )  # Log all the contents of a local directory as artifacts of the run
