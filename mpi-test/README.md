```
mpirun --allow-run-as-root -np 2 -host 10.128.15.201,10.128.15.202    -mca plm_rsh_args "-p 2222"   python /src/mpi-test.py
```
