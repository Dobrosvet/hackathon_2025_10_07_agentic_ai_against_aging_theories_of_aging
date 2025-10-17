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
Stop-ProcessOnPort 8002  # Microservice
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

# Start Microservice
Write-Host "  - Starting PubMed Data Fetcher..." -ForegroundColor Green
$microservicePath = "$ScriptDir\data_a_get_urls_list_papers"
$microserviceProcess = Start-Process -FilePath "poetry" `
    -ArgumentList "run", "python", "main.py" `
    -WorkingDirectory $microservicePath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

# Start Client
Write-Host "  - Starting Client Dashboard..." -ForegroundColor Green
$clientPath = "$ScriptDir\client"
$clientProcess = Start-Process -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $clientPath `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 2

# Save PIDs to file
$pids = @{
    qdrant = $qdrantProcess.Id
    webui = $webUIProcess.Id
    microservice = $microserviceProcess.Id
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
Write-Host "  - Microservice PID: $($microserviceProcess.Id)" -ForegroundColor White
Write-Host "  - Client PID: $($clientProcess.Id)" -ForegroundColor White
Write-Host ""
Write-Host "Access points:" -ForegroundColor Yellow
Write-Host "  - Client Dashboard: http://localhost:5173" -ForegroundColor White
Write-Host "  - PubMed API: http://127.0.0.1:8002" -ForegroundColor White
Write-Host "  - Qdrant Dashboard: http://localhost:8080" -ForegroundColor White
Write-Host "  - Qdrant API: http://localhost:6333" -ForegroundColor White
Write-Host ""
Write-Host "Services are running in background." -ForegroundColor Gray
Write-Host "To stop services, run this script again or use: Stop-Process -Id <PID>" -ForegroundColor Gray
Write-Host ""
Write-Host "Tip: Check logs in the data\logs folder" -ForegroundColor Cyan
