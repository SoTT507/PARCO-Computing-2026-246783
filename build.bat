@echo off
REM ==========================================
REM  Sparse Matrix Benchmark Build Script
REM ==========================================

echo ==========================================
echo   Sparse Matrix Benchmark Build Script
echo ==========================================

REM Check if build directory exists, if not create it
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

REM Navigate to build directory
echo Entering build directory...
cd build

REM Check if CMake is available
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: cmake could not be found. Please install CMake.
    pause
    exit /b 1
)

REM Fresh CMake configuration
echo Running CMake fresh configuration...
cmake --fresh ..

REM Check if CMake configuration was successful
if errorlevel 1 (
    echo Error: CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build .

REM Check if build was successful
if errorlevel 1 (
    echo Error: Build failed!
    pause
    exit /b 1
)

REM Return to project root
cd ..

echo ==========================================
echo   Build completed successfully!
echo ==========================================
echo.
echo Executables available in project root:
echo   - spmv_benchmark.exe
echo   - test_sparse_matrix.exe
echo   - test_performance.exe
echo.
echo To run the benchmark: double click on d1_SpVM.exe
pause
