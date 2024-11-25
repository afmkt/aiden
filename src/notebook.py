import threading
from jupyter_server.serverapp import ServerApp

def start_jupyter_server():
    server_app = ServerApp.instance()
    server_app.initialize([
        "--no-browser",
        "--port=8888",
        "--NotebookApp.base_url=/jupyter",
        "--NotebookApp.token=",
        "--NotebookApp.password=",
        "--NotebookApp.allow_origin=*",
    ])
    threading.Thread(target=server_app.start, daemon=True).start()
    return server_app

jupyter_app = start_jupyter_server()
