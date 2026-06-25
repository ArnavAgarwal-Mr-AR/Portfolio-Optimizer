import os
import sys
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create a fallback FastAPI app in case of import errors
app = FastAPI()

try:
    # Get absolute path of the directory containing this script (api/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Append backend/ folder absolutely to resolve imports correctly in Vercel
    backend_dir = os.path.abspath(os.path.join(current_dir, "..", "backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # Import the actual server app
    from server import app as real_app
    app = real_app
except Exception as e:
    tb = traceback.format_exc()
    print(f"FAILED TO LOAD API SERVER: {e}\n{tb}")
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
    def fallback(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Backend initialization failed",
                "details": str(e),
                "traceback": tb.splitlines(),
                "sys_path": sys.path,
                "current_dir": current_dir if 'current_dir' in locals() else None,
                "exists_backend": os.path.exists(backend_dir) if 'backend_dir' in locals() else False,
                "files_in_backend": os.listdir(backend_dir) if ('backend_dir' in locals() and os.path.exists(backend_dir)) else []
            }
        )
