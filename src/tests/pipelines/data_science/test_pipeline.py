from pathlib import Path
import numpy as np
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import json


def test_data_shape():

    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        output = session.run(pipeline_name="__default__")

    data_catalog = session.load_context().config_loader["catalog"]
    metrics_fig = data_catalog["data_science.metrics_fig"]["data_set"]
    filepath = metrics_fig["filepath"]

    with open(filepath, "r") as file:
        loss_data = json.load(file)

    train_loss = loss_data["data"][0]["y"]

    start = train_loss[0]
    for i, p in enumerate(train_loss):
        assert p < start / (0.5 * (i + 1)), "loss does not decrease efficiently"

    assert 1 == 1
