mkdir raw/
gzip -d -c ./GSE139369_Supp/Supp_GSM4138872_CITE_BMMC_D1T1/GSM4138872_scRNA_BMMC_D1T1.rds.gz > ./raw/scRNA_4138872.rds
gzip -d -c ./GSE139369_Supp/Supp_GSM4138873_CITE_BMMC_D1T2/GSM4138873_scRNA_BMMC_D1T2.rds.gz > ./raw/scRNA_4138873.rds
gzip -d -c ./GSE139369_Supp/Supp_GSM4138888_scATAC_BMMC_D5T1/GSM4138888_scATAC_BMMC_D5T1.fragments.tsv.gz > ./raw/scATAC_4138888_fragments.tsv
gzip -d -c ./GSE139369_Supp/Supp_GSM4138889_scATAC_BMMC_D6T1/GSM4138889_scATAC_BMMC_D6T1.fragments.tsv.gz > ./raw/scATAC_4138889_fragments.tsv
