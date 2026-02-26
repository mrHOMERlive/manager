"""SPRUT 3.0 — FastAPI entry point."""
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="SPRUT 3.0", version="3.0.0")
app.include_router(router)
