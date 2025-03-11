import os

model_dir = "src/models"
if os.path.exists(model_dir):
    print("Files in models directory:", os.listdir(model_dir))
else:
    print(f"❌ Directory '{model_dir}' not found.")
