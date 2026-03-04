import urllib.request
import os

# Create data folder
os.makedirs("data", exist_ok=True)

# URL for MNIST Labels
url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
filename = "data/train-labels-idx1-ubyte.gz"

if os.path.exists(filename):
    print(f"File {filename} already exists. Skipping download.")
else:
    print(f"Downloading MNIST Labels to {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Done!")