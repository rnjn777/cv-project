@echo off
setlocal EnableDelayedExpansion
title CV Lab -- Alzheimer's Multi-Agent Pipeline

:: ============================================================
::  CV Lab — Alzheimer's Detection Pipeline
::  One-stop launcher for all project operations
:: ============================================================

:MENU
cls
echo.
echo  ============================================================
echo   CV Lab ^| Multi-Agent Alzheimer^'s Detector
echo  ============================================================
echo.
echo   [1]  Install dependencies
echo   [2]  Run smoke test          (verify pipeline, no data needed)
echo   [3]  Generate synthetic dataset
echo   [4]  Train   (synthetic data, quick — no files needed)
echo   [5]  Train   (generated dataset, full file-based)
echo   [6]  Inspect generated dataset
echo   [7]  Quick-train demo        (5 epochs, synthetic, fast)
echo   [8]  Exit
echo.
set /p CHOICE="  Select option [1-8]: "

if "%CHOICE%"=="1" goto INSTALL
if "%CHOICE%"=="2" goto SMOKETEST
if "%CHOICE%"=="3" goto GENERATE
if "%CHOICE%"=="4" goto TRAIN_SYNTHETIC
if "%CHOICE%"=="5" goto TRAIN_REAL
if "%CHOICE%"=="6" goto INSPECT
if "%CHOICE%"=="7" goto TRAIN_DEMO
if "%CHOICE%"=="8" goto END

echo   Invalid choice. Try again.
timeout /t 2 >nul
goto MENU


:: ============================================================
:INSTALL
:: ============================================================
cls
echo.
echo  [Installing dependencies]
echo  ----------------------------------------
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found in PATH.
    echo  Install Python 3.9+ from https://python.org
    pause
    goto MENU
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  Found: %%v

echo.
echo  Installing from requirements.txt...
echo  (torch-geometric is installed separately due to CUDA variants)
echo.

:: Core requirements (excluding torch-geometric which needs special install)
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn xgboost nibabel SimpleITK captum wandb
pip install hydra-core omegaconf numpy pandas matplotlib seaborn tqdm scipy

echo.
echo  NOTE: torch-geometric requires matching your CUDA version.
echo  For CPU-only:
echo    pip install torch-geometric
echo  For CUDA: visit https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
echo.
echo  All core dependencies installed!
pause
goto MENU


:: ============================================================
:SMOKETEST
:: ============================================================
cls
echo.
echo  [Smoke Test] Verifying full pipeline...
echo  ----------------------------------------
echo  No data or GPU required. Tests all tensor shapes and invariants.
echo.
python smoke_test.py
echo.
if errorlevel 1 (
    echo  SMOKE TEST FAILED -- see error above.
) else (
    echo  SMOKE TEST PASSED!
)
pause
goto MENU


:: ============================================================
:GENERATE
:: ============================================================
cls
echo.
echo  [Generate Dataset]
echo  ----------------------------------------
echo.
echo  This generates a realistic synthetic ADNI-style dataset:
echo    - 300 subjects (CN / MCI / AD)
echo    - NIfTI MRI files (91x109x91) with class-specific brain signals
echo    - clinical.csv, cognitive.csv, metadata.csv
echo    - train/val/test splits
echo.
echo  Output: data\raw\
echo.
set /p N_SUBJECTS="  Number of subjects [default=300]: "
if "!N_SUBJECTS!"=="" set N_SUBJECTS=300

echo.
echo  Generating !N_SUBJECTS! subjects...
echo  (MRI generation takes ~1-2 min per 100 subjects)
echo.
python data\generate_dataset.py --n !N_SUBJECTS! --seed 42
echo.
if errorlevel 1 (
    echo  Generation failed. Check errors above.
) else (
    echo  Dataset ready at data\raw\
    echo.
    echo  To use it in training, set in configs\config.yaml:
    echo    data:
    echo      synthetic: false
    echo      root: ./data/raw
)
pause
goto MENU


:: ============================================================
:TRAIN_SYNTHETIC
:: ============================================================
cls
echo.
echo  [Train -- Synthetic Mode]
echo  ----------------------------------------
echo  Uses in-memory synthetic data. No files needed.
echo  Config: data.synthetic=true (default)
echo.
echo  Training phases:
echo    Phase 1: sMRI branch (CE-AH + GNN)
echo    Phase 2: Features branch (Clinical + Cognitive)
echo.
python train.py data.synthetic=true
echo.
if errorlevel 1 (
    echo  Training failed. Check errors above.
) else (
    echo  Training complete! Checkpoints saved to .\checkpoints\
)
pause
goto MENU


:: ============================================================
:TRAIN_REAL
:: ============================================================
cls
echo.
echo  [Train -- Generated Dataset]
echo  ----------------------------------------
echo  Uses the file-based dataset in data\raw\
echo  Make sure you have run Option 3 first to generate the data.
echo.

if not exist "data\raw\metadata.csv" (
    echo  ERROR: data\raw\metadata.csv not found.
    echo  Please run Option 3 to generate the dataset first.
    pause
    goto MENU
)

echo  Found dataset at data\raw\
echo.
python train.py data.synthetic=false data.root=./data/raw data.num_workers=0
echo.
if errorlevel 1 (
    echo  Training failed. Check errors above.
) else (
    echo  Training complete! Checkpoints saved to .\checkpoints\
)
pause
goto MENU


:: ============================================================
:INSPECT
:: ============================================================
cls
echo.
echo  [Inspect Dataset]
echo  ----------------------------------------
echo.
if not exist "data\raw\metadata.csv" (
    echo  No dataset found at data\raw\
    echo  Run Option 3 to generate it first.
    pause
    goto MENU
)
echo.
echo  metadata.csv (first 10 rows):
python -c "import pandas as pd; df=pd.read_csv('data/raw/metadata.csv'); print(df.head(10).to_string()); print(f'\nClass counts:\n{df.label.map({0:\"CN\",1:\"MCI\",2:\"AD\"}).value_counts().to_string()}')"
echo.
echo  clinical.csv columns:
python -c "import pandas as pd; df=pd.read_csv('data/raw/clinical.csv',index_col='subject_id'); print(df.head(3).T.to_string())"
echo.
echo  cognitive.csv columns:
python -c "import pandas as pd; df=pd.read_csv('data/raw/cognitive.csv',index_col='subject_id'); print(df.head(3).T.to_string())"
echo.
if exist "data\raw\mri" (
    for /f %%c in ('dir /b "data\raw\mri\*.nii.gz" 2^>nul ^| find /c /v ""') do set MRI_COUNT=%%c
    echo  MRI files: !MRI_COUNT! NIfTI files in data\raw\mri\
) else (
    echo  MRI folder: not found (tabular-only dataset)
)
pause
goto MENU


:: ============================================================
:TRAIN_DEMO
:: ============================================================
cls
echo.
echo  [Quick Demo Train]
echo  ----------------------------------------
echo  5 epochs, synthetic data, small batch.
echo  Good for verifying training runs end-to-end.
echo.
python train.py ^
    data.synthetic=true ^
    data.num_synthetic_samples=80 ^
    data.batch_size=4 ^
    data.num_workers=0 ^
    training.epochs=5 ^
    training.patience=5 ^
    training.warmup_epochs=1 ^
    training.joint_finetune=false ^
    training.use_wandb=false
echo.
if errorlevel 1 (
    echo  Demo training failed. See error above.
) else (
    echo  Demo training complete!
)
pause
goto MENU


:: ============================================================
:END
:: ============================================================
echo.
echo  Goodbye!
endlocal
exit /b 0
