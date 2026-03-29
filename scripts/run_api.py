import os

from _bootstrap_path import bootstrap_src_path

bootstrap_src_path()

from wris.api.app import run_server


if __name__ == "__main__":
    host = os.getenv("WRIS_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    run_server(host=host, port=port)
