
```
docker run -it --rm --gpus all \
-v /mnt/disks/criteo:/data \
-v /home/jupyter/src:/src \
nvcr.io/nvidia/merlin/merlin-training:0.5.3 \
python /src/merlin-sandbox/hugectr/preprocess/nvt-preprocess.py \
--data-path /data/criteo_parquet \
--output-path /data/output_benchmark \
--devices "0,1" \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--num-io-threads 0 \
--part-mem-frac 0.125  
```