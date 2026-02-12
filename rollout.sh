#!/bin/bash
set -e

echo "=== LOCAL ROLLOUT ==="
cd /Users/prasadvellanki/work/mira-voice-ai-pipecat
echo "Stopping existing containers..."
docker compose down 2>&1 || true
echo "Starting local stack..."
docker compose up -d mira-voice open-webui 2>&1
echo ""
echo "Waiting for health checks..."
sleep 5
docker compose ps 2>&1
echo ""

echo "=== AZURE K8S ROLLOUT ==="
echo "Rolling restart pipecat-v2..."
kubectl rollout restart deployment/pipecat-v2 -n openwebui 2>&1
echo "Rolling restart openwebui-v2..."
kubectl rollout restart deployment/openwebui-v2 -n openwebui 2>&1
echo ""
echo "Waiting for rollouts..."
kubectl rollout status deployment/pipecat-v2 -n openwebui --timeout=120s 2>&1 || echo "pipecat-v2 rollout still in progress"
kubectl rollout status deployment/openwebui-v2 -n openwebui --timeout=120s 2>&1 || echo "openwebui-v2 rollout still in progress"
echo ""
echo "Pod status:"
kubectl get pods -n openwebui 2>&1
echo ""
echo "=== ROLLOUT COMPLETE ==="
