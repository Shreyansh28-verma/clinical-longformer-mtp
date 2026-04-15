@echo off
set PYTHONIOENCODING=utf-8
echo Running Training...
.\venv\Scripts\python.exe train_teacher_longformer.py
if %errorlevel% neq 0 (
    echo Training failed!
    exit /b %errorlevel%
)
echo Training complete. Running Inference...
.\venv\Scripts\python.exe inference_teacher_longformer.py
echo Pipeline Complete!
