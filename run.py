import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from backend.app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
