import asyncio
import logging

from hypercorn.asyncio import serve
from hypercorn.config import Config

from minecraft_schematic_generator.server import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S %z]",
)

if __name__ == "__main__":
    app.state.mode = "local"
    app.state.checkpoint_path = (
        # "lightning_logs/mini_model/higher_max_lr/checkpoints/last.ckpt"
        "data/center_data/version_12/checkpoints/last.ckpt"
    )

    config = Config()
    config.bind = ["127.0.0.1:8000"]
    config.accesslog = logging.getLogger("hypercorn.access")
    config.errorlog = logging.getLogger("hypercorn.error")

    asyncio.run(serve(app, config))
