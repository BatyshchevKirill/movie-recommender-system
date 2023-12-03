import os
import requests
from zipfile import ZipFile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEST_FOLDER = "data/raw"

if __name__ == "__main__":
    response = requests.get(DATA_URL)

    if response.status_code == 200:
        file_path = DEST_FOLDER + "/" + DATA_URL.split("/")[-1]

        # Save archive
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Extract data
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(DEST_FOLDER)

        # Remove archive
        os.remove(file_path)
        print("Downloaded the dataset to", DEST_FOLDER)
    else:
        print("Download failed. Code:", {response.status_code})
