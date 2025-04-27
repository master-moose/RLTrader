@echo off
REM run_training_pipeline.bat - Complete pipeline for data collection and model training

REM Create directory structure if it doesn't exist
mkdir data\lob_data 2>nul
mkdir models\dqn_lstm_scalper 2>nul
mkdir logs\dqn_lstm_scalper 2>nul
mkdir logs\dqn_lstm_scalper_continued 2>nul

REM Default values
set COLLECT_DATA=false
set COLLECT_HOURS=1
set CONTINUE_TRAINING=false
set TRAIN_FROM_SCRATCH=false
set CHECKPOINT=
set TIMESTEPS=1000000
set N_ENVS=4
set OUTPUT_DIR=data\lob_data

REM Parse arguments
:parse_args
if "%~1"=="" goto end_parse_args
if "%~1"=="--collect" (
    set COLLECT_DATA=true
    shift
    goto parse_args
)
if "%~1"=="--hours" (
    set COLLECT_HOURS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--continue" (
    set CONTINUE_TRAINING=true
    shift
    goto parse_args
)
if "%~1"=="--train" (
    set TRAIN_FROM_SCRATCH=true
    shift
    goto parse_args
)
if "%~1"=="--checkpoint" (
    set CHECKPOINT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--timesteps" (
    set TIMESTEPS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--envs" (
    set N_ENVS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--output" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" (
    call :display_help
    exit /b 0
)

echo Unknown argument: %1
exit /b 1

:end_parse_args

REM Show help if no arguments provided
if %COLLECT_DATA%==false if %CONTINUE_TRAINING%==false if %TRAIN_FROM_SCRATCH%==false (
    call :display_help
    exit /b 0
)

echo ===== Deep Scalper Training Pipeline =====
echo Configuration:
if %COLLECT_DATA%==true (
    echo - Collecting data for %COLLECT_HOURS% hours
    echo - Output directory: %OUTPUT_DIR%
)

if %CONTINUE_TRAINING%==true (
    if "%CHECKPOINT%"=="" (
        echo Error: Must specify a checkpoint path when using --continue
        exit /b 1
    )
    echo - Continuing training from checkpoint: %CHECKPOINT%
    echo - Training for %TIMESTEPS% timesteps
    echo - Using %N_ENVS% parallel environments
)

if %TRAIN_FROM_SCRATCH%==true (
    echo - Training new model from scratch
    echo - Training for %TIMESTEPS% timesteps
    echo - Using %N_ENVS% parallel environments
)

echo.
echo Starting pipeline...
echo.

REM 1. Data Collection
if %COLLECT_DATA%==true (
    echo ===== Step 1: Data Collection =====
    echo Collecting limit order book data for %COLLECT_HOURS% hours...
    
    REM Calculate collection time in seconds
    set /a COLLECTION_SECONDS=%COLLECT_HOURS% * 3600
    
    REM Run data collector with timeout
    timeout /t %COLLECTION_SECONDS% /nobreak | python deepscalper_agent\lob_collector_v2.py
    
    echo Data collection completed.
    echo Data saved to: %OUTPUT_DIR%
    echo.
)

REM 2. Training
if %TRAIN_FROM_SCRATCH%==true (
    echo ===== Step 2: Training New Model =====
    echo Training new model from scratch for %TIMESTEPS% timesteps...
    
    REM Set environment variables to override defaults
    set TOTAL_TIMESTEPS=%TIMESTEPS%
    set N_ENVS=%N_ENVS%
    set DATA_DIRECTORY=%OUTPUT_DIR%
    
    REM Run training
    python train_scalper.py
    
    echo Training completed.
    echo Model saved to: models\dqn_lstm_scalper
    echo.
)

REM 3. Continued Training
if %CONTINUE_TRAINING%==true (
    echo ===== Step 3: Continued Training =====
    echo Continuing training from checkpoint: %CHECKPOINT%
    echo Training for %TIMESTEPS% additional timesteps...
    
    REM Run continued training
    python train_scalper_continue.py --load_model "%CHECKPOINT%" --total_timesteps %TIMESTEPS% --n_envs %N_ENVS% --data_dir %OUTPUT_DIR%
    
    echo Continued training completed.
    echo Updated model saved to: models\dqn_lstm_scalper
    echo.
)

echo ===== Pipeline Complete =====
echo Check logs directory for training metrics.
echo Models are stored in models\dqn_lstm_scalper directory.

REM Optional: Add timestamp to indicate when the pipeline was last run
echo Last run: %date% %time% > .last_training_run

goto :EOF

:display_help
    echo Usage: run_training_pipeline.bat [options]
    echo.
    echo Options:
    echo   --collect              Collect live data before training
    echo   --hours ^<hours^>        Number of hours to collect data (default: 1)
    echo   --continue             Continue training from a checkpoint
    echo   --train                Train a new model from scratch
    echo   --checkpoint ^<path^>    Path to checkpoint for continued training
    echo   --timesteps ^<steps^>    Number of timesteps for training (default: 1000000)
    echo   --envs ^<num^>           Number of parallel environments (default: 4)
    echo   --output ^<dir^>         Output directory for collected data (default: data\lob_data)
    echo.
    echo Example:
    echo   run_training_pipeline.bat --collect --hours 2 --continue --checkpoint models\dqn_lstm_scalper\dqn_lstm_scalper_final.zip --timesteps 500000
    goto :EOF 