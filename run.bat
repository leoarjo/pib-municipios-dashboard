@echo off
REM Ativa o ambiente virtual e sobe o Streamlit
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run app.py
