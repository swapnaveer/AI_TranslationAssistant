@echo off
echo =====================================
echo Installing TensorFlow (CPU version)...
echo =====================================
pip install tensorflow-cpu

echo.
echo =====================================
echo Cloning BLEURT repo...
echo =====================================
git clone https://github.com/google-research/bleurt.git
cd bleurt

echo.
echo =====================================
echo Installing BLEURT locally...
echo =====================================
pip install .

echo.
echo =====================================
echo (Optional) Downloading BLEURT model...
echo =====================================
mkdir bleurt_base
powershell -Command "Invoke-WebRequest https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip -OutFile bleurt-base-128.zip"
powershell -Command "Expand-Archive bleurt-base-128.zip -DestinationPath bleurt_base"

echo.
echo =====================================
echo BLEURT installed successfully.
echo You can now use it in your Python code.
echo =====================================
pause
