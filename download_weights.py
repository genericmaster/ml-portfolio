# download_weights.py
from huggingface_hub import snapshot_download

print("Downloading the full model repository at once...")
# This pulls the entire repo (both subfolders) down in one go
snapshot_download(repo_id="THENDO1977/from-scratch-models")
print("Repository cached successfully.")
