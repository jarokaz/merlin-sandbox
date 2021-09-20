```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200     -mca plm_rsh_args "-p 2222"   python -m trainer.task
```
```
mpirun --allow-run-as-root -np 2 -host 10.128.0.57,10.128.15.200     -mca plm_rsh_args "-p 2222"   python  trainer/task.py 
```