import os
import kagglehub
import subprocess
import zipfile

# Download latest version into the local 'asl_dataset' folder
TARGET_DIR = os.path.join(os.path.dirname(__file__), "asl_dataset")
os.makedirs(TARGET_DIR, exist_ok=True)

# primary method: kagglehub (prefer), fallback: curl download from Kaggle API URL
KAGGLE_CURL_URL = "https://www.kaggle.com/api/v1/datasets/download/ayuraj/asl-dataset"

def download_with_curl(target_dir):
	zip_path = os.path.join(target_dir, "asl-dataset.zip")
	cmd = ["curl", "-L", "-o", zip_path, KAGGLE_CURL_URL]
	print("Attempting curl download:", " ".join(cmd))
	try:
		res = subprocess.run(cmd, check=False)
		if res.returncode == 0 and os.path.exists(zip_path):
			try:
				with zipfile.ZipFile(zip_path, 'r') as zf:
					zf.extractall(target_dir)
				try:
					os.remove(zip_path)
				except Exception:
					pass
				return target_dir
			except zipfile.BadZipFile:
				print("Downloaded file is not a valid zip or extraction failed.")
				return None
	except Exception as e:
		print("curl download failed:", e)
	return None


path = None
try:
	path = kagglehub.dataset_download("ayuraj/asl-dataset", path=TARGET_DIR)
except Exception as e:
	print("kagglehub download failed or returned cache path:", e)

if not path or not os.path.exists(path):
	# try curl fallback
	path = download_with_curl(TARGET_DIR)

print("Path to dataset files:", path)
