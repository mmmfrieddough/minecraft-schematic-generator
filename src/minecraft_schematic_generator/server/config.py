import argparse
import logging
import os
from typing import Optional

from pydantic import BaseModel

from minecraft_schematic_generator.converter import BlockTokenMapperInterface
from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator

# Set PyMCTranslate logging level before importing it
logging.getLogger("PyMCTranslate").setLevel(logging.WARNING)
from PyMCTranslate import TranslationManager  # noqa: E402


class AppState(BaseModel):
    model: TransformerMinecraftStructureGenerator = None
    block_token_mapper: BlockTokenMapperInterface = None
    translation_manager: TranslationManager = None
    checkpoint_path: Optional[str]
    model_path: Optional[str]
    model_id: str
    model_revision: Optional[str]
    host: str
    port: int
    log_level: str
    device: str
    certfile: Optional[str] = None
    keyfile: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def get_config() -> AppState:
    """Parse command-line arguments or environment variables and return configuration."""
    parser = argparse.ArgumentParser(description="Minecraft Schematic Generator Config")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.environ.get("SCHEM_GEN_CHECKPOINT_PATH", ""),
        help="File path for local checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("SCHEM_GEN_MODEL_PATH", ""),
        help="File path for local model.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get(
            "SCHEM_GEN_MODEL_ID", "mmmfrieddough/minecraft-schematic-generator"
        ),
        help="The Hugging Face model ID.",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=os.environ.get("SCHEM_GEN_MODEL_REVISION", None),
        help="The Hugging Face model revision.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("SCHEM_GEN_HOST", "127.0.0.1"),
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SCHEM_GEN_PORT", "8000")),
        help="Port to run the server on",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("SCHEM_GEN_LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps", "auto"],
        default=os.environ.get("SCHEM_GEN_DEVICE", "auto"),
        help="Device to run the model on (auto will pick the best available)",
    )
    parser.add_argument(
        "--certfile",
        type=str,
        default=os.environ.get("SCHEM_GEN_CERTFILE", None),
        help="Path to the SSL certificate file",
    )
    parser.add_argument(
        "--keyfile",
        type=str,
        default=os.environ.get("SCHEM_GEN_KEYFILE", None),
        help="Path to the SSL private key file",
    )

    args = parser.parse_args()

    return AppState(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        model_id=args.model_id,
        model_revision=args.model_revision,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        device=args.device,
        certfile=args.certfile,
        keyfile=args.keyfile,
    )
