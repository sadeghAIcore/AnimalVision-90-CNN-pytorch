import os
import zipfile
import kagglehub  # pip install kagglehub

def download_data(dataset_slug: str = "iamsouravbanerjee/animal-image-dataset-90-different-animals",
                   dest_dir: str = "../data/animals"):

    'we use this function for downloading our dara from kagglehub'
    os.makedirs(dest_dir, exist_ok=True)

    # starting the downlaoding
    cache_path = kagglehub.dataset_download(dataset_slug)
    print(f"Downloaded to the path : {cache_path}")

    # extracting the file
    for f in os.listdir(cache_path):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(cache_path, f), "r") as z:
                z.extractall(dest_dir)
    else:
        if not os.listdir(dest_dir):
            os.symlink(cache_path, dest_dir + "_link")

    return dest_dir
