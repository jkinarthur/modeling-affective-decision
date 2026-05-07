param(
  [Parameter(Mandatory=$true)] [string]$AccountId,
  [Parameter(Mandatory=$true)] [string]$Region,
  [Parameter(Mandatory=$true)] [string]$RoleArn,
  [Parameter(Mandatory=$true)] [string]$Bucket,
  [string]$EcrRepo = "addan",
  [string]$ImageTag = "latest",
  [string]$InstanceType = "ml.g5.2xlarge",
  [int]$InstanceCount = 1,
  [int]$VolumeSizeGB = 300,
  [int]$MaxRuntimeSeconds = 172800,
  [string]$JobName = ""
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
  Write-Host "`n==== $msg ====" -ForegroundColor Cyan
}

Write-Step "Preflight: AWS CLI identity"
aws --version
aws sts get-caller-identity | Out-Host

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv/Scripts/python.exe"
if (!(Test-Path $python)) {
  throw "Python venv not found at $python"
}

$imageUri = "$AccountId.dkr.ecr.$Region.amazonaws.com/$EcrRepo`:$ImageTag"
$s3Output = "s3://$Bucket/addan/jobs"

Write-Step "Create/verify ECR repository"
$repoExists = $false
try {
  aws ecr describe-repositories --repository-names $EcrRepo --region $Region | Out-Null
  $repoExists = $true
} catch {
  $repoExists = $false
}
if (-not $repoExists) {
  aws ecr create-repository --repository-name $EcrRepo --region $Region | Out-Host
}

Write-Step "ECR login"
aws ecr get-login-password --region $Region |
  docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"

Write-Step "Build and push Docker image"
Push-Location $root
try {
  docker build -t "$EcrRepo`:$ImageTag" .
  docker tag "$EcrRepo`:$ImageTag" $imageUri
  docker push $imageUri
} finally {
  Pop-Location
}

Write-Step "Create/verify S3 bucket"
$bucketExists = $false
try {
  aws s3api head-bucket --bucket $Bucket 2>$null
  if ($LASTEXITCODE -eq 0) { $bucketExists = $true }
} catch {
  $bucketExists = $false
}
if (-not $bucketExists) {
  if ($Region -eq "us-east-1") {
    aws s3api create-bucket --bucket $Bucket --region $Region | Out-Host
  } else {
    aws s3api create-bucket --bucket $Bucket --region $Region --create-bucket-configuration "LocationConstraint=$Region" | Out-Host
  }
}

Write-Step "Launch SageMaker full experiment"
$submitArgs = @(
  "aws/submit_sagemaker_job.py",
  "--role-arn", $RoleArn,
  "--image-uri", $imageUri,
  "--s3-output", $s3Output,
  "--instance-type", $InstanceType,
  "--instance-count", "$InstanceCount",
  "--volume-size-gb", "$VolumeSizeGB",
  "--max-runtime-seconds", "$MaxRuntimeSeconds",
  "--entrypoint", "full-experiment",
  "--region", $Region
)
if ($JobName -ne "") {
  $submitArgs += @("--job-name", $JobName)
}

& $python @submitArgs

Write-Step "Done"
Write-Host "Image URI: $imageUri"
Write-Host "S3 output prefix: $s3Output"
Write-Host "Monitor jobs: aws sagemaker list-training-jobs --name-contains addan --region $Region"
Write-Host "When complete, pull artifacts from model.tar.gz under your S3 output path."
