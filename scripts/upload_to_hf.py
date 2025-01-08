from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

if __name__ == "__main__":
    checkpoint_path = "lightning_logs/center_data/version_12/checkpoints/last.ckpt"
    checkpoint = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
        checkpoint_path
    )
    checkpoint.model.push_to_hub("mmmfrieddough/minecraft-schematic-generator")
