import sys
from pathlib import Path

# Ensure mlops-project (package root) is on sys.path so `import app...` works during pytest collection.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
