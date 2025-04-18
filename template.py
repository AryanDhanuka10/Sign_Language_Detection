import os

# Define directory structure
folders = [
    "artifacts", 
    "artifacts/Data", 
    "src"
]

# Create folders if not exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

# Create empty __init__.py to mark `src` as a package
open("src/__init__.py", "w").close()
print("Initialized src as a Python package.")
