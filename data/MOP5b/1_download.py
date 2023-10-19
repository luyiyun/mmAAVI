import os

url_prefix = "https://raw.githubusercontent.com/PeterZZQ/scMoMaT/main/data/real/MOp_5batches/"
save_dir = "./raw"
os.makedirs(save_dir, exist_ok=True)

files = ["genes.txt", "regions.txt", "GxR.npz", "MxC3.csv", "MxC3_raw.csv"]
for i in [1, 2, 4]:
    files.append("GxC%d.npz" % i)
for i in [1, 3, 5]:
    files.append("RxC%d.npz" % i)
    files.append("BxC%d.npz" % i)
for i in range(1, 6):
    files.append("meta_c%d.csv" % i)

for fi in files:
    os.system("wget -O %s %s" % (os.path.join(save_dir, fi), url_prefix + fi))
