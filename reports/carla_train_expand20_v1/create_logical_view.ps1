param([Parameter(Mandatory=$true)][string]$ManifestPath,[Parameter(Mandatory=$true)][string]$ViewRoot)
$ErrorActionPreference='Stop'
$manifest=Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$viewAbsolute=[System.IO.Path]::GetFullPath($ViewRoot)
New-Item -ItemType Directory -Path $viewAbsolute -Force | Out-Null
foreach($entry in $manifest.scenarios){
    if($entry.scenario_id -notmatch '^Town[0-9A-Za-z]+/scenario-[0-9]+$'){throw 'Invalid canonical TRAIN scenario ID'}
    $entryPath=Join-Path (Join-Path $viewAbsolute 'train') ($entry.scenario_id.Replace('/','\'))
    $entryAbsolute=[System.IO.Path]::GetFullPath($entryPath)
    if(-not $entryAbsolute.StartsWith($viewAbsolute+[System.IO.Path]::DirectorySeparatorChar)){throw 'Logical view path escapes root'}
    $targetAbsolute=[System.IO.Path]::GetFullPath($entry.actual_path)
    if(-not (Test-Path -LiteralPath $targetAbsolute -PathType Container)){throw 'Source scenario is absent'}
    if(Test-Path -LiteralPath $entryAbsolute){
        $item=Get-Item -LiteralPath $entryAbsolute
        $targets=@($item.Target)
        if($item.LinkType -ne 'Junction' -or $targets.Count -ne 1 -or [System.IO.Path]::GetFullPath($targets[0]) -ne $targetAbsolute){throw 'Existing logical view entry disagrees with frozen source'}
    }else{
        New-Item -ItemType Directory -Path (Split-Path -Parent $entryAbsolute) -Force | Out-Null
        New-Item -ItemType Junction -Path $entryAbsolute -Target $targetAbsolute | Out-Null
    }
}
Write-Output 'Read-only logical TRAIN view created; no scenario data copied.'
