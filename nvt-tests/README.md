






```
python process.py \
--data-path gs://jk-vertex-us-central1/criteo_2 \
--out-path gs://jk-vertex-us-central1/nvt-tests \
--devices 0 \
--protocol tcp \
--device-limit-frac 0.8 \
--device-pool-frac 0.9 \
--part-mem-frac 0.125 \
--profile /home/jupyter/dask-profiles
```