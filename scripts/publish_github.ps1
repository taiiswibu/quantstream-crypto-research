param(
    [Parameter(Mandatory = $true)]
    [string]$RepoUrl
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ($RepoUrl -notmatch '^https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9._-]+/?$') {
    throw "Use an HTTPS GitHub repository URL without embedded credentials."
}
if (-not (Test-Path (Join-Path $ProjectRoot ".git"))) {
    throw "Git history is missing. Extract the complete Git-ready package first."
}

$Branch = & git -C $ProjectRoot branch --show-current
if ($LASTEXITCODE -ne 0 -or $Branch -ne "main") {
    throw "Switch to main before initial publication."
}
$PendingChanges = & git -C $ProjectRoot status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect the working tree." }
if ($PendingChanges) { throw "Commit or review pending changes before publication." }

$OriginUrl = & git -C $ProjectRoot remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    & git -C $ProjectRoot remote add origin $RepoUrl
    if ($LASTEXITCODE -ne 0) { throw "Cannot add origin." }
} else {
    $Existing = ($OriginUrl.TrimEnd('/') -replace '\.git$', '').ToLowerInvariant()
    $Requested = ($RepoUrl.TrimEnd('/') -replace '\.git$', '').ToLowerInvariant()
    if ($Existing -ne $Requested) {
        throw "Origin points to another repository. Review git remote -v manually."
    }
}

# Authentication is handled by the user's installed Git credential manager.
# No force-push, token collection, credential changes or remote replacement.
& git -C $ProjectRoot push -u origin main
if ($LASTEXITCODE -ne 0) { throw "Push failed. Review the Git error; history was not force-pushed." }
& git -C $ProjectRoot push origin --tags
if ($LASTEXITCODE -ne 0) { throw "Branch pushed, but tag push failed." }
Write-Host "Published main and tags. Check GitHub Actions for the first remote CI result."
