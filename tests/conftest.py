# tests/conftest.py
import sys
from pathlib import Path

# Inserta el directorio padre (project_root) en la primera posición de sys.path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))