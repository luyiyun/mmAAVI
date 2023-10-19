data_dir="./raw/"
files="scATAC_4138888 scATAC_4138889"

# 首先需要将文件合并在一起进行peak calling，这样才能得到一致的peaks
echo "combine files into one"
for fn in $files; do
  ffn="${data_dir}${fn}_fragments.tsv"
  cat $ffn >> fragment.bed
done

echo "sorting"
sort -k1,1 -k2,2n fragment.bed > fragment_sort.bed
rm fragment.bed

echo "peak calling"
macs3 callpeak -f BED \
 -t fragment_sort.bed \
 -g hs \
 --keep-dup all \
 --outdir call_peaks_res/ \
 -n scATAC \
 -B \
 --nomodel \
 --shift -100 \
 --extsize 200

for fn in $files; do
  echo $fn
  ffn="${data_dir}${fn}_fragments.tsv"
  echo " sorting"
  sort -k1,1 -k2,2n $ffn > fragment_sort.bed
  echo " bzip"
  bgzip -c fragment_sort.bed > fragments.gz
  tabix -p bed fragments.gz
  echo " create count matrix"
  python ./create_count_matrix.py \
    fragments.gz ./call_peaks_res/scATAC_peaks.narrowPeak \
    --save_prefix ${fn} --outdir ./raw --njobs 16
done

rm fragment_sort.bed fragments.gz fragments.gz.tbi
