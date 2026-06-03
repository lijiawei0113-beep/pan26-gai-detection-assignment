param(
    [string]$Model = "qwen2.5:7b-instruct",
    [string]$HumanCsv = "C:\Users\Jiawei\Desktop\final_dataset_human.csv",
    [string]$PythonPath = "",
    [int]$TrainHuman = 200
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $PythonPath) {
    if (Test-Path "D:\Anaconda\python.exe") {
        $PythonPath = "D:\Anaconda\python.exe"
    }
    else {
        $PythonPath = "python"
    }
}

Write-Host "Using Python: $PythonPath"
Write-Host "Using Ollama model: $Model"

try {
    $tags = Invoke-RestMethod -Method Get -Uri "http://localhost:11434/api/tags"
}
catch {
    throw "Could not reach Ollama at http://localhost:11434. Start Ollama and try again."
}

$modelNames = @($tags.models | ForEach-Object { $_.name })
if ($modelNames -notcontains $Model) {
    Write-Host "Installed Ollama models:"
    $modelNames | ForEach-Object { Write-Host "  $_" }
    throw "Model '$Model' is not installed. Run: ollama pull $Model"
}

Write-Host ""
Write-Host "Running Ollama smoke test..."
& $PythonPath scripts/run_pipeline.py `
    --provider ollama `
    --model $Model `
    --limit 3 `
    --train-human 10 `
    --human-csv $HumanCsv `
    --out-dir outputs\ollama_smoke_test

if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed. Full experiment was not started."
}

Write-Host ""
Write-Host "Smoke test finished. Results are in outputs\ollama_smoke_test."
Write-Host "The full test set requires 400 GAI generations. Training with TrainHuman=$TrainHuman adds $($TrainHuman * 2) more GAI generations."
$answer = Read-Host "Run the full Ollama experiment now? Type YES to continue"

if ($answer -eq "YES") {
    Write-Host "Running full Ollama experiment..."
    & $PythonPath scripts/run_pipeline.py `
        --provider ollama `
        --model $Model `
        --limit 200 `
        --train-human $TrainHuman `
        --human-csv $HumanCsv `
        --out-dir outputs\ollama_full

    if ($LASTEXITCODE -ne 0) {
        throw "Full experiment failed. No results summary was produced."
    }

    Write-Host ""
    Write-Host "Full experiment finished."
    Write-Host "Results:"
    Get-Content outputs\ollama_full\results_summary.csv
}
else {
    Write-Host "Stopped after smoke test."
}
