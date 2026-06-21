"""
Pytest configuration - ensures 'app' module is importable.
"""
import sys
import os

# Add backend/ to Python path so 'from app.xxx import yyy' works
sys.path.insert(0, os.path.dirname(__file__))