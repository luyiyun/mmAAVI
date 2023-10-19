import GEOparse

gse = GEOparse.get_GEO(geo="GSE139369", destdir="./")
gse.download_supplementary_files()
