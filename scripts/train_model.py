import json

from _bootstrap_path import bootstrap_src_path

bootstrap_src_path()

from wris.config.settings import load_settings
from wris.data.ingestion import load_rows
from wris.models.training import save_artifacts, train_and_evaluate


if __name__ == "__main__":
    settings = load_settings()
    rows = load_rows(settings.data.raw_path)

    model, metrics = train_and_evaluate(
        rows=rows,
        target=settings.model.target,
        test_size=settings.model.test_size,
        random_state=settings.model.random_state,
        k_neighbors=settings.model.k_neighbors,
    )

    save_artifacts(
        model=model,
        metrics=metrics,
        model_path=settings.artifacts.model_path,
        metrics_path=settings.artifacts.metrics_path,
    )

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))
