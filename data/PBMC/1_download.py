import os

url_prefix = "https://raw.githubusercontent.com/PeterZZQ/scMoMaT/main/data/real/ASAP-PBMC/"
save_dir = "./raw"
os.makedirs(save_dir, exist_ok=True)

files = ["genes.txt", "proteins.txt", "regions.txt", "GxR.npz"]
for i in range(1, 3):
    files.append("GxC%d.npz" % i)
for i in range(1, 5):
    files.append("PxC%d.npz" % i)
for i in range(3, 5):
    files.append("RxC%d.npz" % i)
for i in range(1, 5):
    files.append("meta_c%d.csv" % i)

for fi in files:
    os.system("wget -O %s %s" % (os.path.join(save_dir, fi), url_prefix + fi))
