# R8 Batch Executor — Run simplified R8 (Arm A + B GBM) on all 4 lokasi
# Sequential execution for stability

$PYTHON = "C:\Program Files\Python39\python.exe"
$TEMPLATE = "C:\Users\ariff\DuckDB_kalbar\train_ghi_1h_r8_batch_template.py"

# Lokasi configurations
$LOKASI_CONFIG = @{
    "Kalbar" = @{
        "db" = "C:\Users\ariff\DuckDB_kalbar\kalbar_local.db"
        "cwd" = "C:\Users\ariff\DuckDB_kalbar"
        "time_col" = "timestamp_wib"
        "target_point" = "ghi_target_60m"
        "target_avg" = "ghi_target_avg60m"
    }
    "Bengkulu" = @{
        "db" = "C:\Users\ariff\bengkulu_ghi_julius\bengkulu.duckdb"
        "cwd" = "C:\Users\ariff\bengkulu_ghi_julius"
        "time_col" = "ts_wib"
        "target_point" = "ghi_point_t60"
        "target_avg" = "ghi_avg_t10_t60"
    }
    "Jambi" = @{
        "db" = "C:\Users\ariff\DuckDB_jambi\jambi.duckdb"
        "cwd" = "C:\Users\ariff\DuckDB_jambi"
        "time_col" = "ts"
        "target_point" = "ghi_point_t60"
        "target_avg" = "ghi_avg_t10_t60"
    }
    "Banten" = @{
        "db" = "C:\Users\ariff\Duckdb_Banten\banten.duckdb"
        "cwd" = "C:\Users\ariff\Duckdb_Banten"
        "time_col" = "ts_wib"
        "target_point" = "ghi_point_t60"
        "target_avg" = "ghi_avg_t10_t60"
    }
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "R8 BATCH EXECUTOR — 4 LOKASI (Arm A + Arm B GBM)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$results = @{}

# Run each lokasi sequentially
foreach ($lokasi in @("Kalbar", "Bengkulu", "Jambi", "Banten")) {
    $config = $LOKASI_CONFIG[$lokasi]

    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Running $lokasi..."

    # Copy template to lokasi directory
    Copy-Item $TEMPLATE "$($config.cwd)\train_ghi_1h_r8_batch.py" -Force

    # Run batch script
    $cmd = @(
        $TEMPLATE,
        "--db", $config.db,
        "--lokasi", $lokasi,
        "--time-col", $config.time_col,
        "--target-point", $config.target_point,
        "--target-avg", $config.target_avg,
        "--output", "outputs_R8"
    )

    Set-Location $config.cwd
    $output = & $PYTHON $cmd 2>&1
    $exit_code = $LASTEXITCODE

    $results[$lokasi] = @{
        "success" = ($exit_code -eq 0)
        "exit_code" = $exit_code
    }

    if ($exit_code -eq 0) {
        Write-Host "  OK $lokasi complete" -ForegroundColor Green
    } else {
        Write-Host "  FAILED $lokasi (exit $exit_code)" -ForegroundColor Red
        Write-Host "  Last output:"
        $output | Select-Object -Last 20 | Write-Host
    }
}

# Summary
Write-Host "`n" + "="*70 -ForegroundColor Cyan
Write-Host "BATCH SUMMARY" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

foreach ($lokasi in @("Kalbar", "Bengkulu", "Jambi", "Banten")) {
    $status = if ($results[$lokasi].success) { "OK" } else { "FAILED" }
    $color = if ($results[$lokasi].success) { "Green" } else { "Red" }
    Write-Host "  $lokasi : $status" -ForegroundColor $color
}

$all_success = ($results.Values | Where-Object { -not $_.success }).Count -eq 0

if ($all_success) {
    Write-Host "`nAll lokasi complete! Compiling results..." -ForegroundColor Green

    # Run compilation
    Set-Location "C:\Users\ariff\DuckDB_kalbar"
    $compile_output = & $PYTHON "C:\Users\ariff\DuckDB_kalbar\compile_r8_results.py" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful!" -ForegroundColor Green
        Write-Host "Outputs: r8_compiled/" -ForegroundColor Green
        Write-Host "  - TABLE_2a_v2_feature_engineering.csv" -ForegroundColor Green
        Write-Host "  - TABLE_2c_model_architecture.csv" -ForegroundColor Green
        Write-Host "  - TABLE_2d_pruning_summary.csv" -ForegroundColor Green
    }
} else {
    Write-Host "`nSome lokasi failed. Check output above." -ForegroundColor Red
}

Write-Host "`nBatch execution complete." -ForegroundColor Cyan
