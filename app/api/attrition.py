"""
Enterprise HR AI — Attrition API Routes
Separate route module for attrition prediction endpoints.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["attrition"])

# Import moved to avoid circular imports at module level
# Routes are registered in main.py via app.include_router()
