#!/bin/bash
# Upload curriculum data to Azure PVC and restart pipecat
set -e

cd "$(dirname "$0")"

echo "=== Uploading curriculum to Azure ==="

# Get the pipecat pod name
POD=$(kubectl get pods -n openwebui -l app=pipecat-v2 -o jsonpath='{.items[0].metadata.name}')
echo "Found pod: $POD"

# Copy curriculum file to the PVC via the pod
kubectl cp content/curriculum/ncert_science_10.json "openwebui/$POD:/content/curriculum/ncert_science_10.json"
echo "✅ Copied ncert_science_10.json to pod"

# Verify the file is there
kubectl exec -n openwebui "$POD" -- ls -la /content/curriculum/
echo ""

# Restart so curriculum_manager reloads
kubectl rollout restart deployment/pipecat-v2 -n openwebui
echo "✅ Rollout restart triggered"

# Wait for rollout
kubectl rollout status deployment/pipecat-v2 -n openwebui --timeout=120s
echo "✅ Pipecat v2 is back up — curriculum dropdown should now appear"
