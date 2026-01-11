@echo off
setlocal enableextensions enabledelayedexpansion

:: ------------------------------------------------------------
:: AI Orchestrator setup & run helper
:: - Создаёт venv, ставит зависимости
:: - По желанию скачивает GGUF модель, SD чекпоинт и whisper.cpp бинарь
:: - Записывает пути в .env (добавляет/обновляет)
:: ------------------------------------------------------------

set ROOT=%~dp0
pushd "%ROOT%"

:: --- выбрать Python
set PYTHON=
for %%P in (py -3.10 py -3 python) do (
    %%P -c "import sys" >nul 2>&1 && set PYTHON=%%P && goto :py_found
)

echo [ERROR] Не найден подходящий Python (нужен 3.10+). Установите и повторите.
goto :eof

:py_found
echo [INFO] Использую Python: %PYTHON%

:: --- venv
set VENV=%ROOT%venv
if not exist "%VENV%\Scripts\python.exe" (
    echo [INFO] Создаю venv...
    %PYTHON% -m venv "%VENV%" || goto :fail
) else (
    echo [INFO] venv уже существует
)

set PYTHON_VENV="%VENV%\Scripts\python.exe"

:: --- pip upgrade + requirements
%PYTHON_VENV% -m pip install --upgrade pip >nul
%PYTHON_VENV% -m pip install -r requirements.txt || goto :fail

echo.
echo ==============================================
echo  Настройка путей к моделям (.env обновится)
echo ==============================================

:: --- helper to append/update key in .env
set ENV_FILE=%ROOT%.env
if not exist "%ENV_FILE%" (
    echo [INFO] Создаю .env
    type nul > "%ENV_FILE%"
)

:ask_lm_model
echo Введите путь к GGUF модели мозга (или оставьте пустым, если не готовы):
set /p LM_PATH=
if not defined LM_PATH goto :ask_sd_model
if not exist "%LM_PATH%" echo [WARN] Файл не найден, путь будет записан как есть.
call :set_env_var BRAIN_MODEL_PATH "%LM_PATH%"

echo Введите ссылку HF для скачивания GGUF (опционально, Enter чтобы пропустить):
set /p LM_URL=
if defined LM_URL (
    echo [INFO] Пытаюсь скачать модель...
    call :download_file "%LM_URL%" "%ROOT%models" model.gguf
)

echo.
:ask_sd_model
echo Введите путь к SD чекпоинту (STABLE_DIFFUSION_MODEL_PATH), Enter чтобы пропустить:
set /p SD_PATH=
if defined SD_PATH call :set_env_var STABLE_DIFFUSION_MODEL_PATH "%SD_PATH%"

echo.
:ask_whisper
echo Использовать локальное распознавание аудио (whisper.cpp)? (Y/N, Enter=N):
set /p WHISPER_USE=
if /i "%WHISPER_USE%"=="Y" (
    if not exist "%ROOT%Release" mkdir "%ROOT%Release"
    set "PATTERN=whisper-bin-x64.zip"
    set "CUDA_VER="
    for /f "tokens=3" %%v in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do (
        set CUDA_VER=%%v
        goto :found_cuda
    )
    :found_cuda
    if defined CUDA_VER (
        echo [INFO] Обнаружен драйвер NVIDIA: %CUDA_VER%
        for /f "tokens=1 delims=." %%a in ("%CUDA_VER%") do set CUDA_MAJOR=%%a
        if defined CUDA_MAJOR (
            if %CUDA_MAJOR% LEQ 11 set "PATTERN=whisper-cublas-11.8.0-bin-x64.zip"
            if %CUDA_MAJOR% GEQ 12 set "PATTERN=whisper-cublas-12.4.0-bin-x64.zip"
        )
    )
    echo Авто-выбор whisper.cpp: %PATTERN%
    echo При необходимости выберите вручную:
    echo   1) CPU (whisper-blas-bin-x64.zip)
    echo   2) Generic CPU (whisper-bin-x64.zip)
    echo   3) CUDA 11.x (whisper-cublas-11.8.0-bin-x64.zip)
    echo   4) CUDA 12.x (whisper-cublas-12.4.0-bin-x64.zip)
    set /p WHISPER_CHOICE=Выбор [1-4, Enter=авто]:
    if defined WHISPER_CHOICE (
        if "%WHISPER_CHOICE%"=="1" set "PATTERN=whisper-blas-bin-x64.zip"
        if "%WHISPER_CHOICE%"=="2" set "PATTERN=whisper-bin-x64.zip"
        if "%WHISPER_CHOICE%"=="3" set "PATTERN=whisper-cublas-11.8.0-bin-x64.zip"
        if "%WHISPER_CHOICE%"=="4" set "PATTERN=whisper-cublas-12.4.0-bin-x64.zip"
    )
    echo [INFO] Итоговый выбор: %PATTERN%
    echo [INFO] Ищу актуальный релиз whisper.cpp (%PATTERN%)...
    call :download_whisper_asset "%PATTERN%" "%ROOT%Release" || echo [WARN] Автовыбор не удался, попробую прямую загрузку
    if not exist "%ROOT%Release\%PATTERN%" (
        :: попытка прямой загрузки по имени (latest tag может измениться)
        set "WH_FALLBACK=https://github.com/ggml-org/whisper.cpp/releases/latest/download/%PATTERN%"
        call :download_file "%WH_FALLBACK%" "%ROOT%Release" "%PATTERN%"
    )
    if exist "%ROOT%Release\%PATTERN%" (
        echo [INFO] Распаковываю %PATTERN%...
        tar -xf "%ROOT%Release\%PATTERN%" -C "%ROOT%Release" 2>nul
    ) else (
        echo [WARN] Архив whisper не найден, пропускаю распаковку
    )
    if exist "%ROOT%Release\whisper-cli.exe" (
        echo [INFO] whisper-cli.exe готов
    ) else (
        echo [WARN] whisper-cli.exe не найден, установите вручную в папку Release
    )
) else (
    echo [INFO] Пропускаю установку whisper.cpp
)

echo.
echo Все базовые шаги завершены.
echo Запускать сейчас main.py? (Y/N, Enter=Y):
set /p RUN_NOW=
if /i "%RUN_NOW%"=="N" goto :eof

set ARGS=
echo Запуск в web-режиме? (Y/N, Enter=N):
set /p WEBMODE=
if /i "%WEBMODE%"=="Y" set ARGS=--web

"%PYTHON_VENV%" "%ROOT%main.py" %ARGS%
goto :eof

:: ------------------------------------------------------------
:: Helpers
:: ------------------------------------------------------------
:set_env_var
:: %1=key, %2=value(with quotes)
set KEY=%1
set VAL=%~2
if not defined KEY goto :eof
:: Remove existing key
findstr /v /r "^%KEY%=.*" "%ENV_FILE%" > "%ENV_FILE%.tmp" 2>nul
move /y "%ENV_FILE%.tmp" "%ENV_FILE%" >nul
>>"%ENV_FILE%" echo %KEY%=%VAL%
echo [INFO] %KEY% записан в .env
exit /b 0

:download_file
:: %1=url %2=folder %3=filename
set URL=%~1
set FOLDER=%~2
set NAME=%~3
if not defined URL goto :eof
if not defined FOLDER set FOLDER=%ROOT%
if not exist "%FOLDER%" mkdir "%FOLDER%"
set OUTFILE=%FOLDER%\%NAME%
if exist "%OUTFILE%" del "%OUTFILE%"
:: пробуем curl, затем powershell Invoke-WebRequest
curl -L "%URL%" -o "%OUTFILE%" --retry 3 --retry-delay 2 >nul 2>nul
if exist "%OUTFILE%" (
    echo [INFO] Загружено: %OUTFILE%
    exit /b 0
)
powershell -Command "Try { Invoke-WebRequest -Uri '%URL%' -OutFile '%OUTFILE%' -UseBasicParsing } Catch { exit 1 }" >nul 2>nul
if exist "%OUTFILE%" (
    echo [INFO] Загружено: %OUTFILE%
) else (
    echo [WARN] Не удалось скачать %URL%
)
exit /b 0

:download_whisper_asset
:: %1=pattern (asset name contains), %2=folder
set PAT=%~1
set FOLDER=%~2
if not defined PAT exit /b 1
if not defined FOLDER set FOLDER=%ROOT%Release
if not exist "%FOLDER%" mkdir "%FOLDER%"
powershell -NoProfile -Command "try { $r = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest'; $a = $r.assets | Where-Object { $_.name -like '*%PAT%' } | Select-Object -First 1; if ($a -and $a.browser_download_url) { Invoke-WebRequest -Uri $a.browser_download_url -OutFile '%FOLDER%\%PAT%' -UseBasicParsing; exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>nul
if errorlevel 1 exit /b 1
if exist "%FOLDER%\%PAT%" (
    echo [INFO] Загружено: %FOLDER%\%PAT%
    exit /b 0
)
exit /b 1

:fail
echo [ERROR] Произошла ошибка. Проверьте вывод выше.
exit /b 1
