import os
import sys

# Get absolute path of the directory containing this script (api/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Append backend/ folder absolutely to resolve imports correctly in Vercel
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", "backend")))

from server import app
