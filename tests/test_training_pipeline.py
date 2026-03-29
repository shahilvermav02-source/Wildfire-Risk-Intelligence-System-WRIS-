from wris.features.preprocessing import row_to_features
from wris.models.training import train_and_evaluate


def test_train_and_evaluate_returns_metrics_and_model() -> None:
    rows = [
        {"X": "1", "Y": "2", "month": "jan", "day": "mon", "FFMC": "80", "DMC": "20", "DC": "100", "ISI": "3", "temp": "10", "RH": "70", "wind": "2", "rain": "0", "AREA": "0"},
        {"X": "2", "Y": "3", "month": "feb", "day": "tue", "FFMC": "82", "DMC": "25", "DC": "120", "ISI": "4", "temp": "12", "RH": "65", "wind": "2.1", "rain": "0", "AREA": "0"},
        {"X": "3", "Y": "4", "month": "mar", "day": "wed", "FFMC": "84", "DMC": "30", "DC": "140", "ISI": "5", "temp": "14", "RH": "60", "wind": "2.2", "rain": "0.2", "AREA": "0.1"},
        {"X": "4", "Y": "5", "month": "apr", "day": "thu", "FFMC": "85", "DMC": "35", "DC": "160", "ISI": "6", "temp": "16", "RH": "55", "wind": "2.3", "rain": "0", "AREA": "0.5"},
        {"X": "5", "Y": "6", "month": "may", "day": "fri", "FFMC": "86", "DMC": "40", "DC": "180", "ISI": "7", "temp": "18", "RH": "50", "wind": "2.4", "rain": "0", "AREA": "1.0"},
        {"X": "6", "Y": "7", "month": "jun", "day": "sat", "FFMC": "88", "DMC": "45", "DC": "200", "ISI": "8", "temp": "20", "RH": "45", "wind": "2.5", "rain": "0.1", "AREA": "2.0"},
        {"X": "7", "Y": "8", "month": "jul", "day": "sun", "FFMC": "89", "DMC": "50", "DC": "220", "ISI": "9", "temp": "22", "RH": "40", "wind": "2.6", "rain": "0", "AREA": "3.0"},
        {"X": "8", "Y": "9", "month": "aug", "day": "mon", "FFMC": "90", "DMC": "55", "DC": "240", "ISI": "10", "temp": "24", "RH": "35", "wind": "2.7", "rain": "0", "AREA": "5.0"},
    ]

    model, metrics = train_and_evaluate(
        rows=rows,
        target="area",
        test_size=0.25,
        random_state=42,
        k_neighbors=3,
    )

    assert model is not None
    assert {"mae", "rmse", "r2", "k_selected", "k_config"}.issubset(set(metrics.keys()))


def test_row_to_features_supports_mixed_case_headers() -> None:
    row = {
        "X": "7.4",
        "Y": "4.1",
        "MONTH": "aug",
        "Day": "fri",
        "FFMC": "91.5",
        "dmc": "145.4",
        "DC": "678.2",
        "ISI": "8.3",
        "Temp": "29.1",
        "RH": "35",
        "Wind": "3.6",
        "Rain": "0",
    }
    features = row_to_features(row)
    assert len(features) > 10
