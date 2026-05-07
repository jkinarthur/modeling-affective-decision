"""Submit AD-DAN container jobs to AWS SageMaker.

This lets you move training/evaluation to AWS with one command, then pull
artifacts back for manuscript updates.

Example:
python aws/submit_sagemaker_job.py \
  --role-arn arn:aws:iam::<acct>:role/service-role/AmazonSageMaker-ExecutionRole \
  --image-uri <acct>.dkr.ecr.us-east-1.amazonaws.com/addan:latest \
  --s3-output s3://my-bucket/addan/jobs \
  --instance-type ml.g5.2xlarge \
  --instance-count 1 \
  --job-name addan-reviewer-suite \
  --entrypoint reviewer-suite
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import boto3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit SageMaker training job for AD-DAN")
    p.add_argument("--role-arn", required=True, type=str)
    p.add_argument("--image-uri", required=True, type=str)
    p.add_argument("--s3-output", required=True, type=str)
    p.add_argument("--instance-type", default="ml.g5.2xlarge", type=str)
    p.add_argument("--instance-count", default=1, type=int)
    p.add_argument("--volume-size-gb", default=200, type=int)
    p.add_argument("--job-name", default=None, type=str)
    p.add_argument(
        "--entrypoint",
        default="train-rl",
        choices=[
            "build",
            "train-sft",
            "train-rl",
            "evaluate",
            "reviewer-suite",
            "full-experiment",
            "smoke",
        ],
    )
    p.add_argument("--max-runtime-seconds", default=172800, type=int)
    p.add_argument("--dry-run", action="store_true", help="Print payload without submitting")
    p.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra container env var as KEY=VALUE (repeatable)",
    )
    p.add_argument("--region", default=None, type=str)
    return p.parse_args()


def parse_env_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value '{item}'. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env key in '{item}'")
        out[key] = value
    return out


def main() -> None:
    args = parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"addan-{args.entrypoint}-{stamp}"

    # SageMaker training jobs upload /opt/ml/model as model artifacts to S3.
    # Keep all outputs under this tree to persist checkpoints and reviewer tables.
    env = {
        "DATA_DIR": "/opt/ml/model/data",
        "CKPT_DIR": "/opt/ml/model/checkpoints",
        "ARTIFACT_DIR": "/opt/ml/model/checkpoints/artifacts",
        "HF_HOME": "/opt/ml/model/hf_cache",
        "TRANSFORMERS_CACHE": "/opt/ml/model/hf_cache",
        "TOKENIZERS_PARALLELISM": "false",
    }
    env.update(parse_env_kv(args.env))

    payload = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": args.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/app/entrypoint.sh", args.entrypoint],
        },
        "RoleArn": args.role_arn,
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": args.instance_count,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "OutputDataConfig": {"S3OutputPath": args.s3_output},
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "EnableManagedSpotTraining": False,
        "Environment": env,
    }

    if args.dry_run:
        print("Dry run mode: SageMaker payload")
        print(json.dumps(payload, indent=2))
        return

    session = boto3.session.Session(region_name=args.region)
    sm = session.client("sagemaker")

    response = sm.create_training_job(**payload)

    print("Submitted SageMaker training job")
    print(f"  name: {job_name}")
    print(f"  arn : {response['TrainingJobArn']}")


if __name__ == "__main__":
    main()
