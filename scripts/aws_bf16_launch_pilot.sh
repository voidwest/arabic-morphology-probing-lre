#!/usr/bin/env bash
set -euo pipefail

# Deliberately requires all account/network choices from the caller. It cannot
# accidentally target a historical run or invent a subnet/security group.
: "${AWS_REGION:?set AWS_REGION, e.g. eu-north-1}"
: "${AWS_SUBNET_ID:?set AWS_SUBNET_ID}"
: "${AWS_SECURITY_GROUP_ID:?set AWS_SECURITY_GROUP_ID}"
: "${AWS_KEY_NAME:?set AWS_KEY_NAME}"
: "${AWS_AMI_ID:?set AWS_AMI_ID}"
: "${BF16_INSTANCE_TYPE:?set BF16_INSTANCE_TYPE, e.g. g6e.2xlarge}"
: "${BF16_RUN_ID:?set BF16_RUN_ID to a new unique ID}"
: "${BF16_MARKET_MODE:=on-demand}"

case "$BF16_RUN_ID" in
  lre-corrected-analysis-20260824-v1|lre-corrected-analysis-20260824-v2|lre-corrected-analysis-20260824-v3) echo "refusing frozen revision ID" >&2; exit 2;;
esac

root_device_mapping='[{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":350,"VolumeType":"gp3","Iops":6000,"Throughput":500,"DeleteOnTermination":false,"Encrypted":true}}]'
tag_specs='ResourceType=instance,Tags=[{Key=Name,Value=bf16-pilot-'"$BF16_RUN_ID"'},{Key=Project,Value=research-stack},{Key=Purpose,Value=bf16-hidden-state-pilot},{Key=AutoTeardown,Value=required}] ResourceType=volume,Tags=[{Key=Name,Value=bf16-volume-'"$BF16_RUN_ID"'},{Key=Project,Value=research-stack},{Key=Purpose,Value=bf16-hidden-state-pilot}]'

run_args=(
  aws ec2 run-instances
  --region "$AWS_REGION" \
  --image-id "$AWS_AMI_ID" \
  --instance-type "$BF16_INSTANCE_TYPE" \
  --key-name "$AWS_KEY_NAME" \
  --subnet-id "$AWS_SUBNET_ID" \
  --security-group-ids "$AWS_SECURITY_GROUP_ID" \
  --block-device-mappings "$root_device_mapping" \
  --tag-specifications $tag_specs \
  --count 1 \
  --query 'Instances[0].{InstanceId:InstanceId,State:State.Name,Type:InstanceType,AZ:Placement.AvailabilityZone}' \
  --output json
)

case "$BF16_MARKET_MODE" in
  on-demand) "${run_args[@]}" ;;
  spot)
    : "${BF16_SPOT_MAX_PRICE:?set BF16_SPOT_MAX_PRICE in USD/hour for spot mode}"
    run_args+=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$BF16_SPOT_MAX_PRICE"'","InstanceInterruptionBehavior":"terminate"}}')
    if ! "${run_args[@]}"; then
      if [[ "${BF16_SPOT_ONDEMAND_FALLBACK:-false}" != true ]]; then
        echo "Spot launch failed; refusing implicit On-Demand fallback. Set BF16_SPOT_ONDEMAND_FALLBACK=true explicitly if approved." >&2
        exit 1
      fi
      echo "Spot launch failed; explicit On-Demand fallback was enabled." >&2
      unset 'run_args[${#run_args[@]}-1]'
      "${run_args[@]}"
    fi
    ;;
  *) echo "BF16_MARKET_MODE must be on-demand or spot" >&2; exit 2 ;;
esac
