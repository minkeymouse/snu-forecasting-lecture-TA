from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = os.getenv("APP_RELOAD", "false").lower() == "true"
    uvicorn.run("app.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
