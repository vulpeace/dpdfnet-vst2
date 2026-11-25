param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$release = 'https://github.com/Fovty/HushMic/releases/download/v0.10.2'
$name = 'dpdfnet8_48khz_hr.weights.f32'
$expected = '5a5bb67a8619090c54dc2793536fe813fe38123b236d445f3433aafe3075529d'

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$path = Join-Path $Destination $name
if ((Test-Path -LiteralPath $path -PathType Leaf) -and
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $expected) {
    return
}

$temp = "$path.$PID.partial"
try {
    curl.exe --fail --location --output "$temp" "$release/$name"
    if ($LASTEXITCODE -ne 0) { throw "Could not download $name" }
    if ((Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash -ne $expected) {
        throw "SHA-256 mismatch for $name"
    }
    Move-Item -LiteralPath $temp -Destination $path -Force
} finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
}
