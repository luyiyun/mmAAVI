url_pre="https://raw.githubusercontent.com/PeterZZQ/scMoMaT/main/data/real/ASAP-PBMC/"
save_dir="./data"

mkdir -p $save_dir

files=("genes.txt" "proteins.txt" "regions.txt" "GxR.npz")
# 使用循环迭代，添加元素到数组
for ((i=1; i<3; i++))
do
    files+=("GxC$i.npz")
done
for ((i=3; i<5; i++))
do
    files+=("RxC$i.npz")
done
for ((i=1; i<5; i++))
do
    files+=("PxC$i.npz")
    files+=("meta_c$i.csv")
done

for filei in "${files[@]}"
do
    wget -O ${save_dir}/${filei} ${url_pre}${filei}
done