"""Script to run Sagar Drishti backend server."""
import uvicorn

if __name__ == "__main__":
    print("🌊 Starting Sagar Drishti Ocean Ingestion & Slicing Backend...")
    print("📡 Documentation available at http://localhost:8000/docs")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
