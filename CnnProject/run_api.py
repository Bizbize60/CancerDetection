"""
Convenience launcher for the MammoFusion API.

Usage:
    python run_api.py

Equivalent to:
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
