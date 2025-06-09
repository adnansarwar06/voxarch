"""
init_voxarch.py

Creates the necessary project directories and minimal files for Voxarch.
All directory names and project settings are loaded from a YAML config for flexibility.
No files are created unless required for imports or documentation.
"""

import os
import yaml

# Default config location
CONFIG_PATH = "project_structure.yaml"

def load_config(config_path):
    if not os.path.exists(config_path):
        # Default structure if config does not exist
        return {
            "project_name": "voxarch",
            "folders": [
                "api",
                "rag",
                "config",
                "utils"
            ],
            "files": [
                "README.md",
                ".gitignore",
                "requirements.txt",
                ".env.example"
            ]
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_dirs_and_files(structure):
    for folder in structure.get("folders", []):
        os.makedirs(folder, exist_ok=True)
        # Only add __init__.py if it's inside voxarch
        if folder.startswith("voxarch/") and folder != "voxarch/config":
            init_path = os.path.join(folder, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write('"""\nAuto-generated for Python package structure.\n"""\n')
    for file in structure.get("files", []):
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                if file.endswith("README.md"):
                    f.write("# Voxarch\n\nMultimodal RAG pipeline.\n")
                elif file.endswith(".gitignore"):
                    f.write("*.pyc\n__pycache__/\n.env\n")
                elif file.endswith("requirements.txt"):
                    f.write("# Add project dependencies here\n")
                elif file.endswith(".env.example"):
                    f.write("# Copy this file to .env and fill in values\n")
                else:
                    pass

if __name__ == "__main__":
    structure = load_config(CONFIG_PATH)
    create_dirs_and_files(structure)
    print("Voxarch project structure created successfully.")
