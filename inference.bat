@echo off
setlocal enabledelayedexpansion

REM Simple local inference runner (Windows).
REM
REM Usage:
REM   inference.bat v1.0
REM   inference.bat v1.5
REM   inference.bat v1.5 configs\inference\test.yaml
REM
REM Notes:
REM - This runs the pipeline from app\ by setting PYTHONPATH to include .\app
REM - Edit the YAML config to point at your video/audio paths.

if "%~1"=="" (
  echo Usage: %~nx0 ^<version^> [inference_config]
  echo   ^<version^>: v1.0 ^| v1.5
  echo   [inference_config] optional: path to YAML
  exit /b 1
)

set "VERSION=%~1"

REM Prefer configs moved under app\, but allow old location.
set "DEFAULT_CONFIG_APP=app\configs\inference\test.yaml"
set "DEFAULT_CONFIG_ROOT=configs\inference\test.yaml"
set "CONFIG_PATH="

if not "%~2"=="" (
  set "CONFIG_PATH=%~2"
) else (
  if exist "%DEFAULT_CONFIG_APP%" (
    set "CONFIG_PATH=%DEFAULT_CONFIG_APP%"
  ) else (
    set "CONFIG_PATH=%DEFAULT_CONFIG_ROOT%"
  )
)

set "RESULT_DIR=.\results"

if /i "%VERSION%"=="v1.0" (
  set "MODEL_DIR=.\models\musetalk"
  set "UNET_MODEL_PATH=!MODEL_DIR!\pytorch_model.bin"
  set "UNET_CONFIG=!MODEL_DIR!\musetalk.json"
  set "VERSION_ARG=v1"
) else if /i "%VERSION%"=="v1.5" (
  set "MODEL_DIR=.\models\musetalkV15"
  set "UNET_MODEL_PATH=!MODEL_DIR!\unet.pth"
  set "UNET_CONFIG=!MODEL_DIR!\musetalk.json"
  set "VERSION_ARG=v15"
) else (
  echo Invalid version "%VERSION%" ^(use v1.0 or v1.5^)
  exit /b 1
)

REM Ensure app\ is on the import path so `scripts.*` and `braivtalk` resolve.
set "PYTHONPATH=%CD%\app;%PYTHONPATH%"

python app\scripts\inference.py ^
  --inference_config "%CONFIG_PATH%" ^
  --result_dir "%RESULT_DIR%" ^
  --unet_model_path "%UNET_MODEL_PATH%" ^
  --unet_config "%UNET_CONFIG%" ^
  --version "%VERSION_ARG%"

endlocal
