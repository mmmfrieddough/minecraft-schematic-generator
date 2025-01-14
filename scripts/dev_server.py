import uvicorn

from minecraft_schematic_generator.server import app

if __name__ == "__main__":
    app.state.mode = "local"
    app.state.checkpoint_path = (
        "lightning_logs/center_data/version_13/checkpoints/last.ckpt"
    )

    print("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5)
