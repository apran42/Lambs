param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [int]$Port = 8765,
    [string]$HostAddress = "127.0.0.1"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$siteUrl = "http://127.0.0.1:$Port/"
$statusUrl = "${siteUrl}api/status"

try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $statusUrl -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host "Lambs training-data site is already running: $siteUrl" -ForegroundColor Green
        $status = $response.Content | ConvertFrom-Json
        Write-Host "Current dataset: $($status.dataset_path)"
        Write-Host "Stop the existing process with Ctrl+C before switching datasets."
        return
    }
} catch {
    # No Lambs status endpoint was found. Check whether another application owns the port.
}

$portClient = [System.Net.Sockets.TcpClient]::new()
try {
    $connection = $portClient.BeginConnect("127.0.0.1", $Port, $null, $null)
    $portInUse = $connection.AsyncWaitHandle.WaitOne(300)
    if ($portInUse) {
        $portClient.EndConnect($connection)
        Write-Error "Port $Port is already used by another application. Run again with -Port 8766 (or another free port)."
        exit 1
    }
} catch {
    # Connection refused means the port is available.
} finally {
    $portClient.Dispose()
}

Write-Host "Lambs training-data site: $siteUrl"
Write-Host "Dataset: $Dataset"
python tools/review_training_dataset.py --dataset $Dataset --host $HostAddress --port $Port
