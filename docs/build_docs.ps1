# Set the environment variable to silence Jupyter deprecation warnings (optional)
$env:JUPYTER_PLATFORM_DIRS = "1"

# Navigate to script directory (assumes script is in uqregressors-docs)
Set-Location -Path $PSScriptRoot

# Define source and destination folders
$srcFolder = "..\examples"
$dstFolder = "docs\examples"

# Remove old docs/examples folder if it exists
if (Test-Path $dstFolder) {
    Remove-Item -Recurse -Force $dstFolder
}

# Recreate destination folder
New-Item -ItemType Directory -Path $dstFolder | Out-Null

# Convert all .ipynb files to markdown
Get-ChildItem -Path $srcFolder -Filter *.ipynb -Recurse | ForEach-Object {
    jupyter nbconvert --to markdown `
        $_.FullName `
        --output $_.BaseName `
        --output-dir $dstFolder
}

mkdocs build

# Run mkdocs serve
Write-Host "Starting MkDocs server..."
mkdocs serve