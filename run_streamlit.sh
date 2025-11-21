#!/bin/bash
# Script to run the Streamlit app

echo "Starting Streamlit app..."
echo "Make sure the FastAPI server is running at http://localhost:8000"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

