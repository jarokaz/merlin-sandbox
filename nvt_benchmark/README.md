
```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/merlin-sandbox/nvt_benchmark/nvt-preprocess.py \
--input_data_dir /data/criteo_parquet \
--output_dir /data/output_full \
--n_train_days 21 \
--n_val_days 3 \
--device_limit_frac 0.7 \
--device_pool_frac 0.7 \
--part_mem_frac 0.09
```