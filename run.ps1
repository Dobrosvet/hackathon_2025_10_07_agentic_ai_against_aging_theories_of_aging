# Microservices Launcher Script
# This script starts all microservices and the client application in background

Write-Host "Starting Microservices Dashboard..." -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = "$ScriptDir\.running_services.json"

# Function to stop all services from PID file
function Stop-SavedServices {
    if (Test-Path $PidFile) {
        Write-Host "Stopping previously running services..." -ForegroundColor Yellow

        try {
            $pids = Get-Content $PidFile | ConvertFrom-Json

            if ($pids.microservice) {
                try {
                    Stop-Process -Id $pids.microservice -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped microservice (PID: $($pids.microservice))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.microserviceA) {
                try {
                    Stop-Process -Id $pids.microserviceA -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped microservice A (PID: $($pids.microserviceA))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.microserviceB) {
                try {
                    Stop-Process -Id $pids.microserviceB -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped microservice B (PID: $($pids.microserviceB))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.microserviceC) {
                try {
                    Stop-Process -Id $pids.microserviceC -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped microservice C (PID: $($pids.microserviceC))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.microserviceD) {
                try {
                    Stop-Process -Id $pids.microserviceD -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped microservice D (PID: $($pids.microserviceD))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.microserviceX) {
                try {
                    Stop-Process -Id $pids.microserviceX -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped microservice X (PID: $($pids.microserviceX))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.client) {
                try {
                    Stop-Process -Id $pids.client -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped client (PID: $($pids.client))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.qdrant) {
                try {
                    Stop-Process -Id $pids.qdrant -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped Qdrant server (PID: $($pids.qdrant))" -ForegroundColor Gray
                } catch {}
            }

            if ($pids.webui) {
                try {
                    Stop-Process -Id $pids.webui -Force -ErrorAction SilentlyContinue
                    Write-Host "  - Stopped Qdrant Web UI (PID: $($pids.webui))" -ForegroundColor Gray
                } catch {}
            }
        } catch {
            Write-Host "  - Could not read PID file" -ForegroundColor Gray
        }

        Remove-Item $PidFile -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

# Function to stop processes on specific ports
function Stop-ProcessOnPort {
    param([int]$Port)

    $connections = netstat -ano | Select-String ":$Port\s" | Select-String "LISTENING"

    foreach ($conn in $connections) {
        $parts = $conn -split '\s+' | Where-Object { $_ -ne '' }
        $processId = $parts[-1]

        if ($processId -and $processId -match '^\d+$') {
            try {
                Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
                Write-Host "  - Stopped process on port $Port (PID: $processId)" -ForegroundColor Gray
                Start-Sleep -Milliseconds 500
            } catch {}
        }
    }
}

# Function to kill all Qdrant processes
function Stop-AllQdrantProcesses {
    Get-Process | Where-Object {
        $_.ProcessName -eq "qdrant" -and
        $_.Path -like "*hackathon_aaaa_251016*"
    } | ForEach-Object {
        try {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  - Stopped Qdrant process (PID: $($_.Id))" -ForegroundColor Gray
        } catch {}
    }
}

# Function to kill all Python processes from our project
function Stop-AllPythonProcesses {
    Get-Process | Where-Object {
        $_.ProcessName -eq "python" -and
        $_.Path -like "*hackathon_aaaa_251016*"
    } | ForEach-Object {
        try {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  - Stopped Python process (PID: $($_.Id))" -ForegroundColor Gray
        } catch {}
    }
}

# Function to kill all Node/Vite processes
function Stop-AllNodeProcesses {
    Get-Process | Where-Object {
        $_.ProcessName -eq "node" -and
        ($_.CommandLine -like "*vite*" -or $_.Path -like "*hackathon_aaaa_251016*")
    } | ForEach-Object {
        try {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  - Stopped Node process (PID: $($_.Id))" -ForegroundColor Gray
        } catch {}
    }
}

# Cleanup existing services
Write-Host "Cleaning up existing services..." -ForegroundColor Cyan

# Stop saved services first
Stop-SavedServices

# Stop by port
Stop-ProcessOnPort 8002  # Microservice A (URLs)
Stop-ProcessOnPort 8003  # Microservice B (Full texts)
Stop-ProcessOnPort 8004  # Microservice C (Classifier)
Stop-ProcessOnPort 8005  # Microservice D (Questions Classifier)
Stop-ProcessOnPort 8006  # Microservice X (DB Viewer & Export)
Stop-ProcessOnPort 5173  # Client
Stop-ProcessOnPort 6333  # Qdrant HTTP
Stop-ProcessOnPort 6334  # Qdrant gRPC
Stop-ProcessOnPort 8080  # Qdrant Web UI

# Force kill any remaining processes
Stop-AllQdrantProcesses
Stop-AllPythonProcesses
Stop-AllNodeProcesses

Write-Host "Cleanup complete." -ForegroundColor Green
Write-Host ""

# Load environment variables from .env if present
$envFile = Join-Path $ScriptDir ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment variables from .env..." -ForegroundColor Cyan
    $envLines = Get-Content $envFile
    foreach ($rawLine in $envLines) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            continue
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            Write-Host "  - Skipping malformed line: $line" -ForegroundColor Yellow
            continue
        }

        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        $value = $value.Trim('"')
        $value = $value.Trim("'")

        if (![string]::IsNullOrEmpty($key)) {
            [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
            Write-Host ("  - Loaded {0} from .env" -f $key) -ForegroundColor Gray
        }
    }

    Write-Host "Environment variables loaded." -ForegroundColor Green
    Write-Host ""
}

# Validate Hugging Face token before starting services
$hfToken = $Env:HF_TOKEN
if (-not $hfToken -or $hfToken.Trim().Length -eq 0) {
    Write-Host "HF_TOKEN environment variable is not set. Cannot start gated models." -ForegroundColor Red
    Write-Host "Set it in this PowerShell session before running the launcher, e.g.:" -ForegroundColor Red
    Write-Host "  $Env:HF_TOKEN = 'hf_xxx...'" -ForegroundColor Red
    Write-Host "Or persist it for future sessions with: setx HF_TOKEN \"hf_xxx...\"" -ForegroundColor Red
    exit 1
}

# Start services in background
Write-Host "Starting services..." -ForegroundColor Cyan

# Start Qdrant Server
Write-Host "  - Starting Qdrant Vector Database..." -ForegroundColor Green
$qdrantPath = "$ScriptDir\qdrant"
$qdrantConfigPath = "$qdrantPath\config\config.yaml"
$qdrantProcess = Start-Process -FilePath "$qdrantPath\qdrant.exe" `
    -ArgumentList "--config-path", $qdrantConfigPath `
    -WorkingDirectory $qdrantPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 5

# Start Qdrant Web UI (proxy server on port 8080)
Write-Host "  - Starting Qdrant Web UI with API proxy..." -ForegroundColor Green
$proxyScript = "$ScriptDir\qdrant_proxy.py"
$webUIProcess = Start-Process -FilePath "python" `
    -ArgumentList $proxyScript `
    -WorkingDirectory $ScriptDir `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 2

# Start Microservice A - URLs Fetcher
Write-Host "  - Starting PubMed URLs Fetcher..." -ForegroundColor Green
$microserviceAPath = "$ScriptDir\data_a_get_urls_list_papers"
$microserviceAProcess = Start-Process -FilePath "poetry" `
    -ArgumentList "run", "python", "main.py" `
    -WorkingDirectory $microserviceAPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

# Start Microservice B - Full Text Downloader
Write-Host "  - Starting Full Text Downloader..." -ForegroundColor Green
$microserviceBPath = "$ScriptDir\data_b_get_full_texts"
$microserviceBProcess = Start-Process -FilePath "poetry" `
    -ArgumentList "run", "python", "main.py" `
    -WorkingDirectory $microserviceBPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

# Start Microservice C - Aging Theory Classifier
Write-Host "  - Starting Aging Theory Classifier..." -ForegroundColor Green
$microserviceCPath = "$ScriptDir\data_c_aging_theory_or_not_classifier_and_their_names_extraction"

# Check and install dependencies for microservice C if needed
Write-Host "    Checking dependencies for Classifier..." -ForegroundColor Gray
$venvPython = "$microserviceCPath\.venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    # Check if PyYAML is installed
    $yamlCheck = & $venvPython -c "import yaml" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Installing PyYAML..." -ForegroundColor Yellow
        & $venvPython -m pip install -q pyyaml
    }

    # Check if websockets is installed
    $wsCheck = & $venvPython -c "import websockets" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Installing websockets..." -ForegroundColor Yellow
        & $venvPython -m pip install -q websockets
    }
}

$microserviceCProcess = Start-Process -FilePath "poetry" `
    -ArgumentList "run", "python", "main.py" `
    -WorkingDirectory $microserviceCPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

# Start Microservice D - Questions & Criterias Classifier
Write-Host "  - Starting Questions and Criterias Classifier..." -ForegroundColor Green
$microserviceDPath = "$ScriptDir\data_d_questions_and_criterias_classifier"

# Check if dependencies are installed
Write-Host "    Checking dependencies for Questions Classifier..." -ForegroundColor Gray
$lockFile = "$microserviceDPath\poetry.lock"

if (-not (Test-Path $lockFile)) {
    Write-Host "    Installing dependencies (first time setup)..." -ForegroundColor Yellow
    Push-Location $microserviceDPath
    & poetry install --no-root 2>&1 | Out-Null
    Pop-Location
}

# Check and install additional dependencies if needed
$venvPythonD = "$microserviceDPath\.venv\Scripts\python.exe"

if (Test-Path $venvPythonD) {
    # Check if PyYAML is installed
    $yamlCheck = & $venvPythonD -c "import yaml" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Installing PyYAML..." -ForegroundColor Yellow
        & $venvPythonD -m pip install -q pyyaml
    }

    # Check if websockets is installed
    $wsCheck = & $venvPythonD -c "import websockets" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Installing websockets..." -ForegroundColor Yellow
        & $venvPythonD -m pip install -q websockets
    }

    # Check if transformers is installed
    $transformersCheck = & $venvPythonD -c "import transformers" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Installing transformers and dependencies..." -ForegroundColor Yellow
        Push-Location $microserviceDPath
        & poetry install --no-root 2>&1 | Out-Null
        Pop-Location
    }
}

$microserviceDProcess = Start-Process -FilePath "poetry" `
    -ArgumentList "run", "python", "main.py" `
    -WorkingDirectory $microserviceDPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 5

# Start Microservice X - Database Viewer & Export
Write-Host "  - Starting Database Viewer and Export..." -ForegroundColor Green
$microserviceXPath = "$ScriptDir\data_x_view_db_and_export_to_tables"

# Check if dependencies are installed
Write-Host "    Checking dependencies for Database Viewer..." -ForegroundColor Gray
$lockFileX = "$microserviceXPath\poetry.lock"

if (-not (Test-Path $lockFileX)) {
    Write-Host "    Installing dependencies (first time setup)..." -ForegroundColor Yellow
    Push-Location $microserviceXPath
    & poetry install --no-root 2>&1 | Out-Null
    Pop-Location
}

$microserviceXProcess = Start-Process -FilePath "poetry" `
    -ArgumentList "run", "python", "main.py" `
    -WorkingDirectory $microserviceXPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

# Start Client
Write-Host "  - Starting Client Dashboard..." -ForegroundColor Green
$clientPath = "$ScriptDir\client"
if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Write-Host "    bun executable is not available in PATH. Install bun before starting the client." -ForegroundColor Red
    exit 1
}
$clientProcess = Start-Process -FilePath "bun" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $clientPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 2

# Save PIDs to file
$pids = @{
    qdrant = $qdrantProcess.Id
    webui = $webUIProcess.Id
    microserviceA = $microserviceAProcess.Id
    microserviceB = $microserviceBProcess.Id
    microserviceC = $microserviceCProcess.Id
    microserviceD = $microserviceDProcess.Id
    microserviceX = $microserviceXProcess.Id
    client = $clientProcess.Id
    timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

$pids | ConvertTo-Json | Set-Content $PidFile

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "Process IDs:" -ForegroundColor Yellow
Write-Host "  - Qdrant Server PID: $($qdrantProcess.Id)" -ForegroundColor White
Write-Host "  - Qdrant Web UI PID: $($webUIProcess.Id)" -ForegroundColor White
Write-Host "  - Microservice A (URLs) PID: $($microserviceAProcess.Id)" -ForegroundColor White
Write-Host "  - Microservice B (Full Texts) PID: $($microserviceBProcess.Id)" -ForegroundColor White
Write-Host "  - Microservice C (Classifier) PID: $($microserviceCProcess.Id)" -ForegroundColor White
Write-Host "  - Microservice D (Questions) PID: $($microserviceDProcess.Id)" -ForegroundColor White
Write-Host "  - Microservice X (DB Viewer) PID: $($microserviceXProcess.Id)" -ForegroundColor White
Write-Host "  - Client PID: $($clientProcess.Id)" -ForegroundColor White
Write-Host ""
Write-Host "Access points:" -ForegroundColor Yellow
Write-Host "  - Client Dashboard: http://localhost:5173" -ForegroundColor White
Write-Host "  - PubMed URLs API: http://127.0.0.1:8002" -ForegroundColor White
Write-Host "  - Full Text API: http://127.0.0.1:8003" -ForegroundColor White
Write-Host "  - Classifier API: http://127.0.0.1:8004" -ForegroundColor White
Write-Host "  - Questions Classifier API: http://127.0.0.1:8005" -ForegroundColor White
Write-Host "  - DB Viewer and Export API: http://127.0.0.1:8006" -ForegroundColor White
Write-Host "  - Qdrant Dashboard: http://localhost:8080" -ForegroundColor White
Write-Host "  - Qdrant API: http://localhost:6333" -ForegroundColor White
Write-Host ""
Write-Host "Services are running in background." -ForegroundColor Gray
Write-Host "To stop services, run this script again or use: Stop-Process -Id 'PID'" -ForegroundColor Gray
Write-Host ""
Write-Host "Tip: Check logs in the data\logs folder" -ForegroundColor Cyan
