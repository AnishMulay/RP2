import urllib.request
import os

# Create data folder
os.makedirs("data", exist_ok=True)

url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
filename = "data/train-images-idx3-ubyte.gz"

print(f"Downloading MNIST to {filename}...")
urllib.request.urlretrieve(url, filename)
print("Done! You can now upload the 'data' folder.")