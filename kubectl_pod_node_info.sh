#!/bin/bash
# Concise Kubernetes Pod and Node Information Script

set -e  # Exit immediately if a command exits with a non-zero status

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Not connected to a Kubernetes cluster"
    exit 1
fi

echo "===== KUBERNETES CLUSTER SUMMARY ====="

# Node summary
echo -e "\n[NODES]"
kubectl get nodes --no-headers

# Node taints
echo -e "\n[NODE TAINTS]"
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints --no-headers

# Pod count per node (only for pods assigned to nodes)
echo -e "\n[PODS PER NODE]"
kubectl get pods --all-namespaces -o wide | grep -v "NODE" | grep -v "<none>" | awk '{print $8}' | sort | uniq -c | sort -nr

# Node resource usage
echo -e "\n[NODE RESOURCE USAGE]"
kubectl top nodes 2>/dev/null || echo "metrics-server not installed"

# Pod status summary
echo -e "\n[POD STATUS SUMMARY]"
kubectl get pods --all-namespaces -o wide | grep -v "NAMESPACE" | awk '{print $4}' | sort | uniq -c | sort -nr

# Namespace summary
echo -e "\n[NAMESPACE SUMMARY]"
kubectl get pods --all-namespaces | grep -v "NAMESPACE" | awk '{print $1}' | sort | uniq -c | sort -nr

# Node capacity
echo -e "\n[MAX PODS PER NODE]"
kubectl get node -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.pods}{"\n"}{end}' | sort -k2 -nr | head -1

# Show help for additional commands
echo -e "\n[ADDITIONAL COMMANDS]"
echo "kubectl get pods -n NAMESPACE                # List pods in namespace"
echo "kubectl get pods -o wide                     # List pods with node info"
echo "kubectl top pods -n NAMESPACE                # Show pod resource usage"
echo "kubectl describe node NODE_NAME              # Show detailed node info"
echo "kubectl get pods -A --field-selector spec.nodeName=NODE_NAME  # Pods on specific node"
