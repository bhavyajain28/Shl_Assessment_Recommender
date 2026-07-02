"""Local dev entrypoint: python run.py. Deployment platforms should instead
invoke uvicorn directly (see Procfile)."""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=False)
