@echo off
chcp 65001 >nul
title Lotto AI - Multi-Model Prediction System
setlocal enabledelayedexpansion

set "SCRIPT=%~dp0main_new.py"
set "REQ=%~dp0requirements.txt"

:: ===== 1. conda 찾기 (원본 패턴) =====
set "CONDA_BAT="
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"

if not defined CONDA_BAT goto no_conda

:: CONDA_ROOT 추출
set "CONDA_ROOT=!CONDA_BAT:\condabin\conda.bat=!"
echo [OK] conda: !CONDA_ROOT!

:: ===== 2. lotto 환경 확인 =====
if exist "!CONDA_ROOT!\envs\lotto" goto env_ready

echo.
echo ==================================================
echo   [SETUP] 'lotto' environment not found.
echo           Creating with Python 3.10...
echo ==================================================
echo.
call "%CONDA_BAT%" create -n lotto python=3.10 -y
if errorlevel 1 goto env_fail
echo.
echo [OK] 'lotto' environment created.

:env_ready
:: ===== 3. 환경 활성화 =====
call "%CONDA_BAT%" activate lotto >nul 2>&1

where python >nul 2>&1
if errorlevel 1 goto no_python

echo [OK] lotto environment activated.

:: ===== 4. 패키지 확인 =====
python -c "import torch" >nul 2>&1
if errorlevel 1 goto install_packages
python -c "import xgboost" >nul 2>&1
if errorlevel 1 goto install_packages
python -c "import sklearn" >nul 2>&1
if errorlevel 1 goto install_packages
goto packages_ok

:install_packages
echo.
echo ==================================================
echo   [SETUP] Missing packages detected. Installing...
echo ==================================================
echo.
pip install -r "%REQ%"
echo.
echo [OK] Package installation complete.
echo.
pause

:packages_ok
set "PYTHON=python"
echo.
timeout /t 1 >nul

:: ===== 메뉴 =====
:menu
cls
echo ==================================================
echo       Lotto AI - Multi-Model Prediction System
echo ==================================================
echo.
echo   [1] Predict - Single Model
echo   [2] Predict - Ensemble ^(Auto-Weight^)
echo   [3] Train - Single Model
echo   [4] Train - All Models
echo   [5] Compare Models
echo   [6] Backtest
echo   [7] Check Data Status
echo   [8] Statistics
echo   [9] CNN Grid Visual (Predict + Dashboard)
echo   [0] Exit
echo.
echo ==================================================
set /p "choice=Select: "

if "%choice%"=="1" goto predict_single
if "%choice%"=="2" goto predict_ensemble
if "%choice%"=="3" goto train_single
if "%choice%"=="4" goto train_all
if "%choice%"=="5" goto compare
if "%choice%"=="6" goto evaluate
if "%choice%"=="7" goto crawl
if "%choice%"=="8" goto analyze
if "%choice%"=="9" goto cnn_visual
if "%choice%"=="0" goto end
echo.
echo [!] Invalid input. Enter 0-9.
timeout /t 2 >nul
goto menu

:select_model
echo.
echo Available Models:
echo   [1] gru           ^(Sequence^)
echo   [2] transformer   ^(Attention^)
echo   [3] random_forest ^(Ensemble^)
echo   [4] markov        ^(Statistical^)
echo   [5] lstm          ^(Sequence^)
echo   [6] xgboost       ^(Boosting^)
echo   [7] cnn_grid      ^(Spatial CNN^)
echo.
set /p "model_choice=Select model [1-7]: "
set "MODEL=gru"
if "%model_choice%"=="1" set "MODEL=gru"
if "%model_choice%"=="2" set "MODEL=transformer"
if "%model_choice%"=="3" set "MODEL=random_forest"
if "%model_choice%"=="4" set "MODEL=markov"
if "%model_choice%"=="5" set "MODEL=lstm"
if "%model_choice%"=="6" set "MODEL=xgboost"
if "%model_choice%"=="7" set "MODEL=cnn_grid"
goto :eof

:select_mode
echo Selection Mode ^(MMR diversity^):
echo   [1] safe        - Conservative, high diversity
echo   [2] balanced    - Balanced quality/diversity
echo   [3] aggressive  - Score-focused, bold picks
echo   [0] skip        - Legacy mode ^(no MMR^)
echo.
set /p "mode_choice=Select mode [0-3, default 2]: "
set "MODE_ARG="
if "%mode_choice%"=="" set "MODE_ARG=--mode=balanced"
if "%mode_choice%"=="1" set "MODE_ARG=--mode=safe"
if "%mode_choice%"=="2" set "MODE_ARG=--mode=balanced"
if "%mode_choice%"=="3" set "MODE_ARG=--mode=aggressive"
if "%mode_choice%"=="0" set "MODE_ARG="
goto :eof

:predict_single
cls
echo ==================================================
echo        Predict - Single Model
echo ==================================================
call :select_model
echo.
set /p "sets=Number of sets ^(default 5^): "
if "%sets%"=="" set sets=5
echo.
call :select_mode
echo.
echo [Predicting with %MODEL%...]
echo.
if "%MODE_ARG%"=="" (
    "%PYTHON%" "%SCRIPT%" predict --model=%MODEL% --sets=%sets%
) else (
    "%PYTHON%" "%SCRIPT%" predict --model=%MODEL% --sets=%sets% %MODE_ARG%
)
echo.
pause
goto menu

:predict_ensemble
cls
echo ==================================================
echo        Predict - Ensemble ^(Auto-Weight^)
echo ==================================================
echo.
echo Uses models ^(XGBoost excluded^) with Extended features.
echo.
set /p "sets=Number of sets ^(default 5^): "
if "%sets%"=="" set sets=5
echo.
call :select_mode
echo.
if "%MODE_ARG%"=="" (
    echo Apply filters? ^(Pattern, Statistical, Frequency^)
    set /p "filter_choice=[Y/N, default Y]: "
    set "filter=--filter"
    if /i "!filter_choice!"=="N" set "filter="
) else (
    set "filter="
)
echo.
echo [Ensemble Prediction with Auto-Weight...]
echo.
"%PYTHON%" "%SCRIPT%" predict --ensemble --auto-weight --sets=%sets% %filter% %MODE_ARG%
echo.
pause
goto menu

:train_single
cls
echo ==================================================
echo        Train - Single Model
echo ==================================================
call :select_model
echo.
set /p "epochs=Epochs ^(default 100^): "
if "%epochs%"=="" set epochs=100
echo.
echo [Training %MODEL% for %epochs% epochs...]
echo.
"%PYTHON%" "%SCRIPT%" train --model=%MODEL% --epochs=%epochs%
echo.
pause
goto menu

:train_all
cls
echo ==================================================
echo        Train - All Models ^(XGBoost excluded^)
echo ==================================================
echo.
echo This will train 6 models with Extended features:
echo   gru, transformer, random_forest, markov, lstm, cnn_grid
echo.
set /p "epochs=Epochs ^(default 100^): "
if "%epochs%"=="" set epochs=100
echo.
set /p "confirm=Continue? [Y/N]: "
if /i not "%confirm%"=="Y" goto menu
echo.
echo [Training 6 models for %epochs% epochs...]
echo.
"%PYTHON%" "%SCRIPT%" train --model=all --epochs=%epochs%
echo.
pause
goto menu

:compare
cls
echo ==================================================
echo        Compare Models
echo ==================================================
echo.
set /p "rounds=Number of rounds ^(default 100^): "
if "%rounds%"=="" set rounds=100
echo.
echo [Comparing all models on last %rounds% rounds...]
echo.
"%PYTHON%" "%SCRIPT%" compare --rounds=%rounds%
echo.
pause
goto menu

:evaluate
cls
echo ==================================================
echo        Backtest
echo ==================================================
echo.
echo Evaluate which model?
echo   [1] Single model
echo   [2] All models
echo.
set /p "eval_choice=Select [1-2]: "
echo.
set /p "rounds=Number of rounds ^(default 100^): "
if "%rounds%"=="" set rounds=100

if "%eval_choice%"=="1" (
    call :select_model
    echo.
    echo [Backtesting !MODEL! on last %rounds% rounds...]
    echo.
    "!PYTHON!" "%SCRIPT%" evaluate --model=!MODEL! --rounds=%rounds%
) else (
    echo.
    echo [Backtesting all models on last %rounds% rounds...]
    echo.
    "!PYTHON!" "%SCRIPT%" evaluate --model=all --rounds=%rounds%
)
echo.
pause
goto menu

:crawl
cls
echo ==================================================
echo        Check Data Status
echo ==================================================
echo.
echo [Checking DB/API round gap and updating the blog repo...]
echo.
"%PYTHON%" "%SCRIPT%" crawl
echo.
pause
goto menu

:analyze
cls
echo ==================================================
echo        Statistics Analysis
echo ==================================================
echo.
"%PYTHON%" "%SCRIPT%" analyze
echo.
pause
goto menu

:cnn_visual
cls
echo ==================================================
echo        CNN Grid Visual (Predict + Dashboard)
echo ==================================================
echo.
echo Generates 4 images in analysis\output\:
echo   - Recent rounds grid
echo   - Frequency heatmap
echo   - Prediction probability heatmap
echo   - Dashboard (probability + 5 sets)
echo.
"%PYTHON%" "%~dp0analysis\visualize_grid.py"
echo.
pause
goto menu

:: ===== 에러 핸들러 =====
:no_conda
echo.
echo [ERROR] conda not found.
echo         Install Anaconda or Miniconda:
echo         https://www.anaconda.com/download
echo         https://docs.conda.io/en/latest/miniconda.html
echo.
pause
exit /b 1

:env_fail
echo.
echo [ERROR] Failed to create conda environment.
pause
exit /b 1

:no_python
echo.
echo [ERROR] Python not found after conda activate.
pause
exit /b 1

:end
echo.
echo Exiting...
timeout /t 1 >nul
exit /b 0
