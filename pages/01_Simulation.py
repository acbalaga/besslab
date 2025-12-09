"""Thin wrapper to reuse the main simulation UI defined in ``app.py``.

Streamlit executes each page as a standalone script. Instead of duplicating
``app.py`` here, we import and run the shared ``run_app`` entry point so that
updates only need to be made in one place.
"""

from app import run_app

run_app()
