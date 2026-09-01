"""
Enterprise HR AI — Dashboard API Routes
Separate route module for dashboard/summary endpoints.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
