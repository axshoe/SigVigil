"""Create .gitignore in the project root."""
import os
content = """data/
output/
.venv/
*.egg-info/
__pycache__/
*.pyc
.pytest_cache/
*.parquet
"""
path = ".gitignore"
with open(path, "w") as f:
    f.write(content)
print(f".gitignore created at {os.path.abspath(path)}")
