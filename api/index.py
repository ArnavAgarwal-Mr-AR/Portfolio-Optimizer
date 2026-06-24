import os
import sys

# Add project root to python path to resolve imports correctly in Vercel environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend")))

from server import app
