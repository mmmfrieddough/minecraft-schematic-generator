import asyncio
import logging
import os

import colorlog
from hypercorn.asyncio import serve
from hypercorn.config import Config

from minecraft_schematic_generator.server import app, get_config

if __name__ == "__main__":
    app.state = get_config()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s [%(process)d] [%(log_color)s%(levelname)s%(reset)s] %(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S %z]",
        )
    )
    logger = colorlog.getLogger()
    logger.setLevel(app.state.log_level)
    logger.addHandler(handler)

    config = Config()
    config.bind = [f"{app.state.host}:{app.state.port}"]
    config.accesslog = logging.getLogger("hypercorn.access")
    config.errorlog = logging.getLogger("hypercorn.error")

    # Configure TLS if certificate and key files are provided
    if app.state.certfile and app.state.keyfile:
        if not os.path.exists(app.state.certfile):
            raise FileNotFoundError(f"Certificate file not found: {app.state.certfile}")
        if not os.path.exists(app.state.keyfile):
            raise FileNotFoundError(f"Key file not found: {app.state.keyfile}")

        config.certfile = app.state.certfile
        config.keyfile = app.state.keyfile
        logging.info("TLS enabled")
    else:
        logging.info("TLS is disabled - running in HTTP mode")

    logging.info("Starting server")
    asyncio.run(serve(app, config))
