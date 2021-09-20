```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200     -mca plm_rsh_args "-p 2222"   python -m trainer.task
```
```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200     -mca plm_rsh_args "-p 2222"   python  trainer/task.py 
```
```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200 \
-mca plm_rsh_args "-p 2222" \
-x PATH \
-x PYTHONPATH \
-x LD_LIBRARY_PATH \
-x CUDA_PATH \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
python  mpi-test.py
```

```
mpirun --allow-run-as-root -np 1 -host 10.128.15.200 \
-mca plm_rsh_args "-p 2222" \
-x PATH \
-x PYTHONPATH \
-x LD_LIBRARY_PATH \
-x CUDA_PATH \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
python  mpi-test.py
```

```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200 \
-mca plm_rsh_args "-p 2222" \
-x PATH \
-x PYTHONPATH \
-x LD_LIBRARY_PATH \
-x CUDA_PATH \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
python  trainer/task.py \
--max_iter 2000
```

```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200 \
-mca plm_rsh_args "-p 2222" \
-x PATH \
-x PYTHONPATH \
-x LD_LIBRARY_PATH \
-x CUDA_PATH \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
python trainer/task.py \
--max_iter 5000 \
--gpus "[[0,1]]"
```

```
mpirun --allow-run-as-root -np 1 -host 10.128.15.200 \
-mca plm_rsh_args "-p 2222" \
-x PATH \
-x PYTHONPATH \
-x LD_LIBRARY_PATH \
-x CUDA_PATH \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
python trainer/task.py \
--max_iter 5000 \
--gpus "[[0,1]]"


```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200 \
-mca plm_rsh_args "-p 2222" \
-mca btl_tcp_if_exclude 172.17.0.0/16 \
-x PATH \
-x PYTHONPATH \
-x LD_LIBRARY_PATH \
-x CUDA_PATH \
-x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
python trainer/task.py \
--max_iter 5000 \
--gpus "[[0,1],[0,1]]"
```