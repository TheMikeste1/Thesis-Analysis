@ECHO off

@REM Activate the environment
CALL venv\Scripts\activate

@REM Create the notebook backup directory if it doesn't exist
IF NOT EXIST .\notebooks\backup (
    MKDIR .\notebooks\backup
)
@REM Copy all the notebooks to the backup directory
FOR %%f IN (./notebooks/*.ipynb) DO (
    >NUL COPY /Y /B %cd%\notebooks\%%f %cd%\notebooks\backup\%%f.bak
)

@REM Sync the notebooks
call jupytext --sync notebooks\*.ipynb

