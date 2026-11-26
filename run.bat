@echo off
echo ============================================================
echo  Culturally Sensitive GPT Hallucination Evaluation System
echo ============================================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit application...
streamlit run app.py
pause
