#!/bin/bash
# Google Cloud SDK and GKE Authentication Plugin Installation Script

set -e  # Exit immediately if a command exits with a non-zero status

echo "[+] Installing Google Cloud SDK and GKE authentication plugin..."

# Create installation directory
INSTALL_DIR="$HOME/google-cloud-sdk"
mkdir -p "$HOME/tmp"
cd "$HOME/tmp"

# Download Google Cloud SDK
echo "[+] Downloading Google Cloud SDK..."
# curl -L -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-latest-linux-x86_64.tar.gz
curl -L -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-latest-linux-x86_64.tar.gz

# Extract the SDK
echo "[+] Extracting Google Cloud SDK..."
tar -xzf google-cloud-cli-latest-*.tar.gz -C "$HOME"

# Install the SDK
echo "[+] Installing Google Cloud SDK..."
"$INSTALL_DIR/install.sh" --quiet --usage-reporting=false --command-completion=true --path-update=true

# Update PATH in current shell
source "$INSTALL_DIR/path.bash.inc"

# Install GKE auth plugin and kubectl
echo "[+] Installing GKE authentication plugin and kubectl..."
gcloud components install gke-gcloud-auth-plugin kubectl --quiet

# Add to .bashrc for future sessions
if ! grep -q "source \$HOME/google-cloud-sdk/path.bash.inc" "$HOME/.bashrc"; then
  echo "[+] Adding Google Cloud SDK to PATH in .bashrc..."
  echo "# Google Cloud SDK" >> "$HOME/.bashrc"
  echo "source \$HOME/google-cloud-sdk/path.bash.inc" >> "$HOME/.bashrc"
  echo "source \$HOME/google-cloud-sdk/completion.bash.inc" >> "$HOME/.bashrc"
fi

# Clean up
echo "[+] Cleaning up..."
rm -rf "$HOME/tmp/google-cloud-cli-latest-*.tar.gz"

echo "[+] Installation complete!"
echo "[+] Please run 'source ~/.bashrc' or start a new terminal to use the Google Cloud SDK."
echo "[+] Then authenticate with: gcloud auth login"
echo "[+] Configure kubectl: gcloud container clusters get-credentials CLUSTER_NAME --zone ZONE --project PROJECT_ID"
echo ""
echo "[+] Verify installation with:"
echo "    kubectl version --client"
echo "    gke-gcloud-auth-plugin --version"
echo ""
echo "[+] If you're still experiencing 'system:anonymous' authentication errors, make sure to:"
echo "    1. Authenticate with: gcloud auth login"
echo "    2. Set your project: gcloud config set project YOUR_PROJECT_ID"
echo "    3. Get cluster credentials: gcloud container clusters get-credentials CLUSTER_NAME --zone ZONE --project PROJECT_ID"
echo "    4. Verify authentication: kubectl auth can-i get pods"
