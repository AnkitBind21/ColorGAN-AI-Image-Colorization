import os
import gdown

MODEL_PATH = "checkpoints/generator_latest.pt"

if not os.path.exists(MODEL_PATH):
    os.makedirs("checkpoints", exist_ok=True)

    gdown.download(
        "https://drive.google.com/uc?id=1JcB0xyADclJHKDPaavFGJyQoolobVXF_",
        MODEL_PATH,
        quiet=False
    )

    print("Model downloaded.")