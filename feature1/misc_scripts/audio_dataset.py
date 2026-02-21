# Download ESC-50 dataset script
import os
import zipfile
import urllib.request

def download_esc50():
    """Download and extract ESC-50 dataset"""
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = "ESC-50.zip"
    
    if not os.path.exists("ESC-50-master"):
        print("Downloading ESC-50 dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        
        os.remove(zip_path)
        print("Done! Audio files are in ESC-50-master/audio/")
    
    return "ESC-50-master/audio/"

# Use it
audio_dir = download_esc50()
