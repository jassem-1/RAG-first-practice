import requests
from pathlib import Path
from tqdm import tqdm

# Create a data folder if it doesn't exist
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Where we will save Paul Graham's essays
file_path = data_folder / "pg_essays.txt"


def download_essays():
    url = "https://raw.githubusercontent.com/dbredvick/paul-graham-to-kindle/main/paul_graham_essays.txt"
    
    # If we already have the file and it's bigger than 1 MB → skip download
    if file_path.exists() and file_path.stat().st_size > 1_000_000:
        print("Paul Graham essays already downloaded – skipping")
        return
    
    print("Downloading Paul Graham essays (~1.5 MB)…")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # will crash if URL is wrong
    
    total_size = int(response.headers.get("content-length", 0))
    with open(file_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="pg_essays.txt"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    
    print(f"Downloaded and saved to {file_path}")

if __name__ == "__main__":
    download_essays()