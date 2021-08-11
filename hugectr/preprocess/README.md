
```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/merlin-sandbox/hugectr/preprocess/nvt-preprocess.py \
--input_data_dir /data/criteo_parquet \
--out_dir /data/output_fullj \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--part_mem_frac 0.125 \
```