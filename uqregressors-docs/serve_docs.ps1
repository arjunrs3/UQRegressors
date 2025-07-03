# Set the environment variable to silence Jupyter deprecation warnings (optional)
$env:JUPYTER_PLATFORM_DIRS = "1"

# Navigate to script directory (assumes script is in uqregressors-docs)
Set-Location -Path $PSScriptRoot

# Define source and destination folders
$srcFolder = "..\examples"
$dstFolder = "docs\examples"

# Remove old docs/examples folder if it exists
if (Test-Path $dstFolder) {
    Write-Host "Removing old $dstFolder"
    Remove-Item -Recurse -Force $dstFolder
}

# Recreate destination folder
Set-Location -Path $PSScriptRoot
Write-Host "Creating $dstFolder"
New-Item -ItemType Directory -Path $dstFolder | Out-Null

# Convert all .ipynb files to markdown
Write-Host "Converting notebooks to markdown..."
Get-ChildItem -Path $srcFolder -Filter *.ipynb -Recurse | ForEach-Object {
    $inputFile = $_.FullName
    $outputBaseName = $_.BaseName

    Write-Host "Converting $inputFile ..."
    # Use --output-dir to control where markdown goes, and --output for filename only
    jupyter nbconvert --to markdown $inputFile --output $outputBaseName --output-dir $dstFolder
}

# Run mkdocs serve
Write-Host "Starting MkDocs server..."
mkdocs serve