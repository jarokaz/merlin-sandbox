```
docker run -it --rm --gpus all \
-v /home/jupyter/src/merlin-sandbox/hugectr/preprocess:/src \
-v /mnt/disks/criteo:/criteo_data \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/preprocess_nvt.py \
--train_folder /criteo_data/train_parquet \
--valid_folder /criteo_data/valid_parquet \
--output_folder /criteo_data/criteo_processed \
--devices 0,1 \
--protocol tcp \
--device_limit_frac 0.8 \
--device_pool_frac 0.9 \
--num_io_threads 4 \
--part_mem_frac 0.08 \
--out_files_per_proc 8 \
--freq_limit 6 \
--shuffle PER_PARTITION



```


## Cardinalities

Based on training dataset day_0 - day_19

```

```