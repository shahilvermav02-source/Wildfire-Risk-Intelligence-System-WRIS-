from _bootstrap_path import bootstrap_src_path

bootstrap_src_path()

from wris.api.app import run_server


if __name__ == "__main__":
    run_server(host="127.0.0.1", port=8000)
