@echo off
setlocal enabledelayedexpansion

set "CLUS_JAR=C:\Users\Ningo\Desktop\clus\ClusProject\target\clus-2.12.8-deps.jar"
set "SETTINGS_DIR=..\data\swdpf\clus_ssl_settings"
set "OUTPUT_DIR=..\results\clus_ssl_outputs"

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

set SCALES=10 20 70
set FOLDS=1 2 3 4 5 6 7

for %%F in (%FOLDS%) do (
    for %%S in (%SCALES%) do (
        echo Running CLUS+ on fold %%F with scale %%S%%...
        set "SETTINGS_FILE=%SETTINGS_DIR%\fold%%F_ssl_%%S.s"
        set "OUTPUT_FILE=%OUTPUT_DIR%\fold%%F_ssl_%%S.out"

        java -jar "%CLUS_JAR%" -ssl "!SETTINGS_FILE!" > "!OUTPUT_FILE!"

        echo Output saved to !OUTPUT_FILE!
    )
)

echo.
echo All experiments completed successfully.
pause
