from wris.models.training import train_and_evaluate


def test_train_and_evaluate_returns_metrics_and_model() -> None:
    rows = [
        {"x": "1", "y": "2", "month": "jan", "day": "mon", "ffmc": "80", "dmc": "20", "dc": "100", "isi": "3", "temp": "10", "rh": "70", "wind": "2", "rain": "0", "area": "0"},
        {"x": "2", "y": "3", "month": "feb", "day": "tue", "ffmc": "82", "dmc": "25", "dc": "120", "isi": "4", "temp": "12", "rh": "65", "wind": "2.1", "rain": "0", "area": "0"},
        {"x": "3", "y": "4", "month": "mar", "day": "wed", "ffmc": "84", "dmc": "30", "dc": "140", "isi": "5", "temp": "14", "rh": "60", "wind": "2.2", "rain": "0.2", "area": "0.1"},
        {"x": "4", "y": "5", "month": "apr", "day": "thu", "ffmc": "85", "dmc": "35", "dc": "160", "isi": "6", "temp": "16", "rh": "55", "wind": "2.3", "rain": "0", "area": "0.5"},
        {"x": "5", "y": "6", "month": "may", "day": "fri", "ffmc": "86", "dmc": "40", "dc": "180", "isi": "7", "temp": "18", "rh": "50", "wind": "2.4", "rain": "0", "area": "1.0"},
        {"x": "6", "y": "7", "month": "jun", "day": "sat", "ffmc": "88", "dmc": "45", "dc": "200", "isi": "8", "temp": "20", "rh": "45", "wind": "2.5", "rain": "0.1", "area": "2.0"},
        {"x": "7", "y": "8", "month": "jul", "day": "sun", "ffmc": "89", "dmc": "50", "dc": "220", "isi": "9", "temp": "22", "rh": "40", "wind": "2.6", "rain": "0", "area": "3.0"},
        {"x": "8", "y": "9", "month": "aug", "day": "mon", "ffmc": "90", "dmc": "55", "dc": "240", "isi": "10", "temp": "24", "rh": "35", "wind": "2.7", "rain": "0", "area": "5.0"},
    ]

    model, metrics = train_and_evaluate(
        rows=rows,
        target="area",
        test_size=0.25,
        random_state=42,
        k_neighbors=3,
    )

    assert model is not None
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
