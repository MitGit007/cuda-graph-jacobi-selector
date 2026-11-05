$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $ProjectRoot "build-3050\jacobi.exe"
$CompareScript = Join-Path $PSScriptRoot "compare.py"
$ResultsDir = Join-Path $ProjectRoot "results"
$TempDir = Join-Path $ResultsDir "large_validation_temp"

if (-not (Test-Path $Exe)) {
    throw "Executable not found: $Exe. Build build-3050 first."
}
if (-not (Test-Path $CompareScript)) {
    throw "Comparison script not found: $CompareScript"
}

$PythonCommand = $null
$PythonPrefix = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
    $PythonPrefix = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = "python"
} else {
    throw "Python 3 was not found. Install Python or place it on PATH."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $ResultsDir "large_validation_$Stamp.txt"

$Cases = @(
    [pscustomobject]@{ Grid = 1024; Iterations = 511;  Batch = 16  },
    [pscustomobject]@{ Grid = 1024; Iterations = 5000; Batch = 128 },
    [pscustomobject]@{ Grid = 2048; Iterations = 511;  Batch = 32  },
    [pscustomobject]@{ Grid = 2048; Iterations = 5000; Batch = 128 }
)

"Large-grid output validation started: $(Get-Date -Format o)" |
    Set-Content -Encoding UTF8 $Log

function Run-Comparison {
    param(
        [string]$Reference,
        [string]$Test,
        [int]$Grid,
        [string]$Label
    )

    "" | Add-Content $Log
    "$Label" | Tee-Object -FilePath $Log -Append
    $Output = & $PythonCommand @PythonPrefix $CompareScript `
        $Reference $Test $Grid $Grid --tol 1e-5 2>&1
    $CompareExit = $LASTEXITCODE
    $Output | Tee-Object -FilePath $Log -Append
    if ($CompareExit -ne 0) {
        throw "Large-grid validation failed: $Label"
    }
}

foreach ($Case in $Cases) {
    $Grid = $Case.Grid
    $Iterations = $Case.Iterations
    $Batch = $Case.Batch
    $Prefix = "g${Grid}_n${Iterations}"
    $OrdinaryFile = Join-Path $TempDir "${Prefix}_ordinary.bin"
    $GraphFile = Join-Path $TempDir "${Prefix}_graph.bin"
    $SelectorFile = Join-Path $TempDir "${Prefix}_selector.bin"

    "" | Add-Content $Log
    "Case: ${Grid}x${Grid}, N=$Iterations, graph batch=$Batch" |
        Tee-Object -FilePath $Log -Append

    & $Exe --mode ordinary --rows $Grid --cols $Grid `
        --iters $Iterations --seed 42 --output $OrdinaryFile
    if ($LASTEXITCODE -ne 0) { throw "Ordinary run failed." }

    & $Exe --mode graph --rows $Grid --cols $Grid `
        --iters $Iterations --seed 42 --batch $Batch --output $GraphFile
    if ($LASTEXITCODE -ne 0) { throw "Graph run failed." }

    & $Exe --mode selector --rows $Grid --cols $Grid `
        --iters $Iterations --seed 42 --output $SelectorFile
    if ($LASTEXITCODE -ne 0) { throw "Selector run failed." }

    Run-Comparison $OrdinaryFile $GraphFile $Grid `
        "ordinary vs graph: ${Grid}x${Grid}, N=$Iterations, B=$Batch"
    Run-Comparison $OrdinaryFile $SelectorFile $Grid `
        "ordinary vs selector: ${Grid}x${Grid}, N=$Iterations"

    Remove-Item -Force -Path @($OrdinaryFile, $GraphFile, $SelectorFile)
}

"" | Add-Content $Log
"Overall: ALL LARGE-GRID COMPARISONS PASS" |
    Tee-Object -FilePath $Log -Append
Write-Host "Validation log: $Log"
