import argparse
import logging
import os

import uvicorn

from minecraft_schematic_generator.server.app import app


def parse_args():
    parser = argparse.ArgumentParser(description="Minecraft Structure Generator Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MSG_PORT", "8000")),
        help="Port to run the server on",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MSG_HOST", "0.0.0.0"),
        help="Host to run the server on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    args = parse_args()
    app.state.mode = "production"
    app.state.checkpoint_path = ""

    logger.info("Starting server")
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5)
