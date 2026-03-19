from wris.config.settings import load_settings
from wris.data.ingestion import download_csv


if __name__ == "__main__":
    settings = load_settings()
    out = download_csv(settings.data.url, settings.data.raw_path)
    print(f"Downloaded dataset to: {out}")
