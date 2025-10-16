import os
import requests
import zipfile
from tqdm import tqdm

def download_tiny_imagenet(data_dir=".", url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"):
    """
    Downloads and extracts the Tiny ImageNet dataset.
    """
    dataset_path = os.path.join(data_dir, "tiny-imagenet-200")
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if os.path.exists(dataset_path):
        print("Tiny ImageNet dataset already exists. Skipping download.")
        return

    print(f"Downloading Tiny ImageNet from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return

    print("\nExtracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up zip file.")

    print("Dataset downloaded and extracted successfully.")

if __name__ == "__main__":
    download_tiny_imagenet()
