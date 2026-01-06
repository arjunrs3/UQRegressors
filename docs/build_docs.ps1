# Silence Jupyter warnings
$env:JUPYTER_PLATFORM_DIRS = "1"

Set-Location -Path $PSScriptRoot

$srcFolder = Join-Path $PSScriptRoot ".." "examples"
$dstFolder = Join-Path $PSScriptRoot "docs" "examples"

if (Test-Path $dstFolder) {
    Remove-Item -Recurse -Force $dstFolder
}

New-Item -ItemType Directory -Path $dstFolder | Out-Null

Get-ChildItem -Path $srcFolder -Filter *.ipynb -Recurse | ForEach-Object {
    jupyter nbconvert --to markdown `
        $_.FullName `
        --output $_.BaseName `
        --output-dir $dstFolder
}

mkdocs build -f mkdocs.yml -d site
