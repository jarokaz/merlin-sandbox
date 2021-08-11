
```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/merlin-sandbox/nvt_benchmark/dask-nvtabular-criteo-benchmark.py \
--data_path /data/criteo_parquet \
--output_path /data/output_benchmark \
--devices "0,1" \
--device_limit-frac 0.8 \
--device_pool_frac 0.9 \
--num_io_threads 0 \
--part_mem_frac 0.125 \
--profile /data/output_benchmark/dask-report.html
```