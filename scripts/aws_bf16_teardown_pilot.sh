#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?set AWS_REGION}"
: "${BF16_INSTANCE_ID:?set BF16_INSTANCE_ID}"

echo "Stopping/terminating only the explicitly supplied BF16 pilot instance: $BF16_INSTANCE_ID" >&2
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$BF16_INSTANCE_ID" --output json
echo "The non-root EBS volume is intentionally preserved. Verify and delete it only after the local bundle copy is complete." >&2
