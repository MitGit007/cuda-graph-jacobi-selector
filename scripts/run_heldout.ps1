param(
    [ValidateRange(1, 20)]
    [int]$Repetitions = 5
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $ProjectRoot "build-3050\jacobi.exe"
$ResultsDir = Join-Path $ProjectRoot "results"

if (-not (Test-Path $Exe)) {
    throw "Executable not found: $Exe. Build build-3050 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputCsv = Join-Path $ResultsDir "heldout_raw_$Stamp.csv"
$OrderCsv = Join-Path $ResultsDir "heldout_order_$Stamp.csv"
$EnvironmentFile = Join-Path $ResultsDir "heldout_environment_$Stamp.txt"

"repetition,rows,cols,iterations,mode,requested_batch" |
    Set-Content -Encoding ASCII $OrderCsv

@(
    "Held-out CUDA Graph experiment"
    "Started: $(Get-Date -Format o)"
    "Computer: $env:COMPUTERNAME"
    "Repetitions: $Repetitions"
    "Development grids excluded: 128, 512"
    "Held-out grids: 32, 1024, 2048"
) | Set-Content -Encoding UTF8 $EnvironmentFile

try {
    "" | Add-Content $EnvironmentFile
    "nvidia-smi before sweep:" | Add-Content $EnvironmentFile
    (& nvidia-smi 2>&1) | Add-Content $EnvironmentFile
} catch {
    "nvidia-smi unavailable: $($_.Exception.Message)" | Add-Content $EnvironmentFile
}

$GridSizes = @(32, 1024, 2048)
$IterationCounts = @(33, 100, 511, 1000, 5000)
$CandidateBatches = @(2, 4, 8, 16, 32, 64, 100, 128)

# Build one complete policy list, then randomize the whole list independently
# for every repetition. This spreads thermal and background-load effects across
# grids, iteration counts, and policies.
$Tasks = @()
foreach ($Grid in $GridSizes) {
    foreach ($Iterations in $IterationCounts) {
        $Tasks += [pscustomobject]@{
            Grid = $Grid; Iterations = $Iterations; Mode = "ordinary"; Batch = 0
        }

        foreach ($Batch in $CandidateBatches) {
            if ($Batch -le $Iterations) {
                $Tasks += [pscustomobject]@{
                    Grid = $Grid; Iterations = $Iterations
                    Mode = "graph"; Batch = $Batch
                }
            }
        }

        $Tasks += [pscustomobject]@{
            Grid = $Grid; Iterations = $Iterations; Mode = "selector"; Batch = 0
        }
    }
}

$TotalRuns = $Tasks.Count * $Repetitions
Write-Host "Held-out sweep: $TotalRuns runs ($Repetitions repetitions)."
Write-Host "Keep the laptop plugged in and do not use it during the sweep."

for ($Rep = 1; $Rep -le $Repetitions; $Rep++) {
    $RandomOrder = $Tasks | Sort-Object { Get-Random }

    foreach ($Task in $RandomOrder) {
        "$Rep,$($Task.Grid),$($Task.Grid),$($Task.Iterations),$($Task.Mode),$($Task.Batch)" |
            Add-Content -Encoding ASCII $OrderCsv

        Write-Host "Run: rep=$Rep grid=$($Task.Grid)x$($Task.Grid) N=$($Task.Iterations) mode=$($Task.Mode) batch=$($Task.Batch)"

        if ($Task.Mode -eq "graph") {
            & $Exe --mode graph --rows $Task.Grid --cols $Task.Grid `
                --iters $Task.Iterations --seed 42 --batch $Task.Batch `
                --csv $OutputCsv
        } else {
            & $Exe --mode $Task.Mode --rows $Task.Grid --cols $Task.Grid `
                --iters $Task.Iterations --seed 42 --csv $OutputCsv
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Jacobi run failed with exit code $LASTEXITCODE."
        }
    }
}

@(
    ""
    "Finished: $(Get-Date -Format o)"
    ""
    "nvidia-smi after sweep:"
) | Add-Content $EnvironmentFile
try {
    (& nvidia-smi 2>&1) | Add-Content $EnvironmentFile
} catch {
    "nvidia-smi unavailable: $($_.Exception.Message)" | Add-Content $EnvironmentFile
}

Write-Host ""
Write-Host "Held-out sweep completed."
Write-Host "Timing data: $OutputCsv"
Write-Host "Execution order: $OrderCsv"
Write-Host "Environment record: $EnvironmentFile"
