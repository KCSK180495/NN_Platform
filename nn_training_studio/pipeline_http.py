"""Local HTTP entry point copied into an exported pipeline as serve_pipeline.py."""

import argparse
from pathlib import Path

import pandas as pd

from nn_training_studio.pipeline_package import predict
from nn_training_studio.processing_pipeline import Pipeline


def create_app(bundle, *, trust_code=False, trust_artifacts=False, healthy=False):
    from fastapi import Body, FastAPI, HTTPException

    pipeline = Pipeline.read(bundle)
    app = FastAPI(title="NN Studio Processing Pipeline")

    def run(body, inference=False):
        if set(body) - {"rows", "record_column"} or not isinstance(body.get("rows"), list):
            raise HTTPException(422, "Send rows as a list of objects and optionally a record_column.")
        column = body.get("record_column")
        if column is not None and not isinstance(column, str):
            raise HTTPException(422, "record_column must be a column name.")
        if any(not isinstance(row, dict) for row in body["rows"]):
            raise HTTPException(422, "Each data row must be an object.")
        try:
            output = pipeline.transform(pd.DataFrame(body["rows"]), column,
                                        trust_code=trust_code, healthy=healthy)
            if inference:
                output = predict(pipeline, output, bundle, trust_artifacts=trust_artifacts)
            return {"rows": output.to_dict(orient="records")}
        except (ValueError, TypeError, OSError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "sample_rate_hz": pipeline.rate, "outputs": pipeline.outputs}

    @app.post("/transform")
    def transform(body: dict = Body(...)):
        return run(body)

    @app.post("/predict")
    def predict_rows(body: dict = Body(...)):
        return run(body, inference=True)

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust-code", action="store_true")
    parser.add_argument("--trust-artifacts", action="store_true")
    parser.add_argument("--healthy", action="store_true", help="Confirm known healthy startup for Scope v1")
    args = parser.parse_args()
    uvicorn.run(create_app(Path(__file__).resolve().parent, trust_code=args.trust_code,
                           trust_artifacts=args.trust_artifacts, healthy=args.healthy),
                host="127.0.0.1", port=args.port)
