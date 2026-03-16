@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
start "" http://localhost:8501
streamlit run src\dashboard.py --server.headless true
