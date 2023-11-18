from typing import Literal

import bentoml

from pydantic import BaseModel
from bentoml.io import JSON


iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

class Request(BaseModel):
    sepal_width: float
    sepal_length: float
    petal_width: float
    petal_length: float

class Response(BaseModel):
    label: Literal["setosa", "versicolor", "virginica"]


@svc.api(input=JSON(pydantic_model=Request), output=JSON(pydantic_model=Response))
def classify(request: Request) -> Response:
    input_ = [
        request.sepal_width,
        request.sepal_length,
        request.petal_width,
        request.petal_length,
    ]

    label_index = iris_clf_runner.predict.run([input_])[0]
    label = ["setosa", "versicolor", "virginica"][label_index]

    return Response(label=label)




