# pip install gdown

zip_save_path="./data/wnn"
mkdir -p $zip_save_path
gdown https://docs.google.com/uc?id=12zum6CcNsPeVzbhbZPozkbAzDuYfh4GM -O ${zip_save_path}/wnn_demo_labels.zip
gdown https://drive.google.com/uc?id=1l6U-WdUIqNuE5Fc_e2f6Qz8nuXyukVFo -O ${zip_save_path}/wnn_demo_processed.zip
unzip ${zip_save_path}/wnn_demo_labels.zip -d ${zip_save_path}/labels && \
    mv ${zip_save_path}/labels/wnn_demo_labels/* ${zip_save_path}/labels/ && \
    rm -r ${zip_save_path}/labels/wnn_demo_labels
unzip ${zip_save_path}/wnn_demo_processed.zip -d ${zip_save_path}/processed && \
    mv ${zip_save_path}/processed/data/processed/* ${zip_save_path}/processed/ && \
    rm -r ${zip_save_path}/processed/data

zip_save_path="./data/dogma"
mkdir -p $zip_save_path
gdown https://docs.google.com/uc?id=1K1KR9xWRae-8TxUDlqAOP094_PXFUPTh -O ${zip_save_path}/dogma_demo_labels.zip
gdown https://drive.google.com/uc?id=1TBWfJ2GjnYQ5CwoG7JBDcOvVDiG9cJ6B -O ${zip_save_path}/dogma_demo_processed.zip
unzip ${zip_save_path}/dogma_demo_labels.zip -d ${zip_save_path}/labels && \
    mv ${zip_save_path}/labels/dogma_demo_labels/* ${zip_save_path}/labels/ && \
    rm -r ${zip_save_path}/labels/dogma_demo_labels
unzip ${zip_save_path}/dogma_demo_processed.zip -d ${zip_save_path}/processed