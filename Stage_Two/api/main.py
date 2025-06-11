from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from .endpoints import terrain, wells, optimization

app = FastAPI(
    title="Terraforming AI API",
    description="API for terrain optimization and well placement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(terrain.router, prefix="/api/terrain", tags=["terrain"])
app.include_router(wells.router, prefix="/api/wells", tags=["wells"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])

@app.get("/")
async def root():
    """Root endpoint that returns available API endpoints"""
    return {
        "message": "Welcome to the Terraforming AI API",
        "endpoints": {
            "terrain": {
                "GET /api/terrain/initial": "Get initial terrain data",
                "GET /api/terrain/goal": "Get goal terrain data",
                "GET /api/terrain/current": "Get current terrain state",
                "GET /api/terrain/discrepancy": "Get terrain discrepancy"
            },
            "wells": {
                "GET /api/wells": "Get all placed wells",
                "POST /api/wells": "Add a new well",
                "GET /api/wells/{well_id}": "Get specific well details",
                "GET /api/wells/preview": "Preview well placement"
            },
            "optimization": {
                "POST /api/optimization/single": "Optimize single well",
                "POST /api/optimization/terrain": "Run full terrain optimization",
                "GET /api/optimization/history": "Get optimization history"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 