$ErrorActionPreference = 'Stop'
$auditRoot = $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $auditRoot 'SHA256SUMS')) { throw 'Audit is sealed; verify manifest hashes read-only instead of regenerating artifacts.' }
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $auditRoot)
$expectedParent = '888a63a0f0bb190eb899cfe900674bea6804b354017993b45c235b41b1c26839'
$expectedHead = '15c24f8384aab61472fb347f518f1e2efadd676e'
$encoding = [System.Text.UTF8Encoding]::new($false)
function Write-JsonFile($name, $value) {
    $destination = Join-Path $auditRoot $name
    if (-not [System.IO.Path]::GetFullPath($destination).StartsWith($auditRoot + [System.IO.Path]::DirectorySeparatorChar)) { throw 'Write escaped audit root' }
    [System.IO.File]::WriteAllText($destination, (($value | ConvertTo-Json -Depth 30) + [Environment]::NewLine), $encoding)
}
function Get-GitBlobHash($path) {
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $header = [System.Text.Encoding]::UTF8.GetBytes(('blob ' + $bytes.Length + [char]0))
    $combined = [byte[]]::new($header.Length + $bytes.Length)
    [Array]::Copy($header, 0, $combined, 0, $header.Length)
    [Array]::Copy($bytes, 0, $combined, $header.Length, $bytes.Length)
    return [Convert]::ToHexString([System.Security.Cryptography.SHA1]::HashData($combined)).ToLower()
}
$sources = Join-Path $auditRoot 'sources'
$records = @()
$sourceCounts = [ordered]@{}
foreach ($repo in @('carla-gen', 'devkit', 'baselines')) {
    $treeName = if ($repo -eq 'carla-gen') { 'tree.json' } else { $repo + '_tree.json' }
    $tree = Get-Content -LiteralPath (Join-Path $sources $treeName) -Raw | ConvertFrom-Json
    if ($tree.truncated) { throw "Truncated current tree for $repo" }
    $sourceCounts[$repo] = 0
    foreach ($entry in $tree.tree) {
        if ($entry.type -ne 'blob') { continue }
        $sourcePath = Join-Path (Join-Path $sources $repo) $entry.path
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            if ($repo -ne 'carla-gen' -or $entry.path -ne 'assets/example.gif') { throw "Missing intended source $repo/$($entry.path)" }
            $records += [ordered]@{
                kind='public_repository_media_metadata_only'; repository="carlanomaly/$repo"; commit=$tree.sha
                path=$entry.path; bytes=$entry.size; advertised_git_blob_sha1=$entry.sha
                payload_fetched=$false; exclusion='Example animation intentionally not requested or interpreted'
            }
            continue
        }
        $blobHash = Get-GitBlobHash $sourcePath
        if ($blobHash -ne $entry.sha) { throw "Public source blob mismatch $repo/$($entry.path)" }
        $file = Get-Item -LiteralPath $sourcePath
        $records += [ordered]@{
            kind='public_repository_text'; repository="carlanomaly/$repo"; commit=$tree.sha; path=$entry.path
            local_path="sources/$repo/$($entry.path)"
            url="https://raw.githubusercontent.com/carlanomaly/$repo/$($tree.sha)/$($entry.path)"
            bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLower()
            advertised_git_blob_sha1=$entry.sha; verified_git_blob_sha1=$blobHash
            saved_at_utc=$file.LastWriteTimeUtc.ToString('o'); source_executed=$false
        }
        $sourceCounts[$repo]++
    }
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $sources 'historical_devkit') -File -Recurse) {
    $relative = [System.IO.Path]::GetRelativePath((Join-Path $sources 'historical_devkit'), $file.FullName).Replace('\','/')
    $blobHash = Get-GitBlobHash $file.FullName
    $matches = @()
    foreach ($treeFile in Get-ChildItem -LiteralPath (Join-Path $sources 'history/devkit') -File) {
        $tree = Get-Content -LiteralPath $treeFile.FullName -Raw | ConvertFrom-Json
        $entry = $tree.tree | Where-Object { $_.path -eq $relative -and $_.sha -eq $blobHash } | Select-Object -First 1
        if ($entry) { $matches += $treeFile.Name.Replace('_tree.json','') }
    }
    if (-not $matches.Count) { throw "Unbound historical public source $relative" }
    $records += [ordered]@{
        kind='historical_public_repository_text'; repository='carlanomaly/devkit'; path=$relative
        matching_commits=$matches; local_path="sources/historical_devkit/$relative"
        url="https://raw.githubusercontent.com/carlanomaly/devkit/$($matches[0])/$relative"
        bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()
        verified_git_blob_sha1=$blobHash; source_executed=$false
    }
}
$metadataUrls = @{
    'repository.json'='https://api.github.com/repos/carlanomaly/carla-gen'
    'tree.json'='https://api.github.com/repos/carlanomaly/carla-gen/git/trees/main?recursive=1'
    'commit_detail.json'='https://api.github.com/repos/carlanomaly/carla-gen/commits/9b3b284721b41e7aea87ffe2aa812766775be146'
    'owner_repositories.json'='https://api.github.com/users/carlanomaly/repos?per_page=100'
    'publisher_resources.html'='https://carlanomaly.de/resources/'
    'publisher_about.html'='https://carlanomaly.de/about/'
    'hydra_working_directory.html'='https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/working_directory/'
    'baselines_reproducible_runs_tree.json'='https://api.github.com/repos/carlanomaly/baselines/git/trees/reproducible-runs?recursive=1'
}
foreach ($file in Get-ChildItem -LiteralPath $sources -File) {
    $url = $metadataUrls[$file.Name]
    if (-not $url) {
        $repo = 'carla-gen'; $suffix = $file.BaseName
        foreach ($prefix in @('devkit','baselines')) {
            if ($suffix.StartsWith($prefix + '_')) { $repo=$prefix; $suffix=$suffix.Substring($prefix.Length+1); break }
        }
        $endpoint = if ($suffix -eq 'tree') {'git/trees/main?recursive=1'} elseif ($suffix -in @('issues','pulls')) {$suffix + '?state=all&per_page=100'} else {$suffix + '?per_page=100'}
        $url = "https://api.github.com/repos/carlanomaly/$repo/$endpoint"
    }
    $records += [ordered]@{
        kind=if ($file.Extension -eq '.html') {'public_documentation'} else {'public_GitHub_API_metadata'}
        local_path="sources/$($file.Name)"; url=$url; bytes=$file.Length
        sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()
        saved_at_utc=$file.LastWriteTimeUtc.ToString('o')
    }
}
foreach ($repo in @('devkit','baselines')) {
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $sources "history/$repo") -File) {
        $commit = $file.Name.Replace('_tree.json','')
        $tree = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
        if ($tree.truncated) {throw "Truncated historical tree $repo/$commit"}
        $records += [ordered]@{
            kind='public_historical_file_tree'; repository="carlanomaly/$repo"; commit=$commit
            local_path="sources/history/$repo/$($file.Name)"
            url="https://api.github.com/repos/carlanomaly/$repo/git/trees/$($commit)?recursive=1"
            bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()
            truncated=$false; historical_blob_contents_exhaustively_read=$false
        }
    }
}
$commitDetail = Get-Content -LiteralPath (Join-Path $sources 'commit_detail.json') -Raw | ConvertFrom-Json
$inventory = [ordered]@{
    schema_version=1; milestone='PUBLIC PROVENANCE RECOVERY / GENERATOR AUDIT ONLY'
    client_date='2026-10-02'; client_timezone='Asia/Calcutta'; recorded_at_utc=[DateTime]::UtcNow.ToString('o')
    parent_audit_manifest_sha256=$expectedParent
    repository='carlanomaly/carla-gen'; pinned_commit=$commitDetail.sha; commit_tree_object_sha1=$commitDetail.commit.tree.sha
    public_generator_history=@{commits=1;branches=1;tags=0;releases=0;issues=0;pull_requests=0;root_commit_parent_count=0}
    public_related_history=@{devkit_main_commits=6;baselines_main_commits=6;devkit_branches=1;baselines_branches=@('main','reproducible-runs')}
    preserved_current_text_counts=$sourceCounts; source_records=$records
    coverage='Generator complete public text and advertised history; related current public text and main historical file trees; removed devkit plan/docs; additional baselines branch file tree'
    limitations=@('Related historical blobs and additional branch blob contents not exhaustively searched','No unadvertised, deleted, private or future refs claimed inspected','No certified per-released-scenario binding to generator revision')
    test_payload_access=$false; official_archive_access=$false; public_source_executed=$false
    local_context=@{parent_audit='reports/carla_test_metadata_audit_v1'; protocol='reports/carla_final_evaluation_preregistration_v1'; protected_baseline='protected_audit_protocol_baseline.json'; payloads_or_model_outputs_rehashed=$false}
}
Write-JsonFile 'source_inventory.json' $inventory
$evidence = Get-Content -LiteralPath (Join-Path $auditRoot 'evidence_index.json') -Raw | ConvertFrom-Json
foreach ($property in $evidence.PSObject.Properties) {
    $e = $property.Value
    $path = Join-Path $sources ($e.repository.Replace('carlanomaly/','') + '/' + $e.path)
    $lines = [System.IO.File]::ReadAllLines($path)
    if ($e.line_start -lt 1 -or $e.line_end -gt $lines.Length -or $e.line_end -lt $e.line_start) { throw "Invalid evidence range $($property.Name)" }
}
$required = @('source_inventory.json','generator_structure.json','seed_route_audit.json','scenario_naming_audit.json','normal_anomaly_linkage_audit.json','released_dataset_mapping_audit.json','family_grouping_recommendation.json','report.md')
foreach ($name in $required) {
    $path = Join-Path $auditRoot $name
    if (-not (Test-Path -LiteralPath $path)) {throw "Missing required artifact $name"}
    if ($name.EndsWith('.json')) { $null=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
}
$baseline = Get-Content -LiteralPath (Join-Path $auditRoot 'protected_audit_protocol_baseline.json') -Raw | ConvertFrom-Json
foreach ($record in $baseline) {
    if ((Get-FileHash -LiteralPath $record.path -Algorithm SHA256).Hash.ToLower() -ne $record.sha256) { throw "Protected file changed $($record.path)" }
}
$parentRoot = Join-Path $workspaceRoot 'reports/carla_test_metadata_audit_v1'
$parentManifest = Join-Path $parentRoot 'SHA256SUMS'
if ((Get-FileHash -LiteralPath $parentManifest -Algorithm SHA256).Hash.ToLower() -ne $expectedParent) { throw 'Parent seal changed' }
$parentEntries = 0
foreach ($line in Get-Content -LiteralPath $parentManifest) {
    $parts=$line -split '  ',2
    if ((Get-FileHash -LiteralPath (Join-Path $parentRoot $parts[1]) -Algorithm SHA256).Hash.ToLower() -ne $parts[0]) {throw "Parent entry changed $($parts[1])"}
    $parentEntries++
}
$head = (git -C $workspaceRoot rev-parse HEAD).Trim()
if ($head -ne $expectedHead) {throw 'Git HEAD changed'}
$before = @(Get-Content -LiteralPath (Join-Path $auditRoot 'initial_git_status.txt') | Where-Object {$_ -and $_ -notmatch 'reports/carla_public_provenance_audit_v1/'})
$after = @(git -C $workspaceRoot status --porcelain=v1 | Where-Object {$_ -and $_ -notmatch 'reports/carla_public_provenance_audit_v1/'})
$diff = @(Compare-Object $before $after)
if ($diff.Count) {throw 'Git status changed outside audit directory'}
$integrity = [ordered]@{
    status='PASS'; required_artifacts_valid=$true; evidence_ranges_valid=$true
    public_text_blob_hashes_valid=$true; public_current_source_counts=$sourceCounts
    parent_manifest_sha256=$expectedParent; parent_entries_verified=$parentEntries
    protected_audit_protocol_files_unchanged=$baseline.Count
    git_head_before=$expectedHead; git_head_after=$head
    git_status_outside_new_audit_unchanged=$true
    changes_confined_to='reports/carla_public_provenance_audit_v1/'
    test_archives_requested=0; test_payload_files_opened=0; test_timestep_labels_opened=0; model_output_files_opened=0
    partition_executed=$false; inference_executed=$false; fitting_executed=$false; training_executed=$false
    protocol_amended=$false; commit_created=$false; pushed=$false
    code_verification='JSON/schema-presence checks, citation range checks and source/hash preservation only; no public software tests or simulator executed'
    checks_completed_at_utc=[DateTime]::UtcNow.ToString('o')
    scope='Audit/protocol metadata hashed; no broader raw payload or model-output rehash. Other workspace preservation evidenced by confined writes and unchanged Git status, not a full untracked-content checksum.'
}
Write-JsonFile 'integrity_verification.json' $integrity
Write-Output ("Verified {0} public current text blobs, {1} protected files, {2} parent entries." -f (($sourceCounts.Values | Measure-Object -Sum).Sum),$baseline.Count,$parentEntries)
Write-Output 'Audit artifacts are ready to seal. This script does not create a partition or amend a protocol.'
