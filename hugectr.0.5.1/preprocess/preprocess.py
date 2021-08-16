# Copyright (c) 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Standard Libraries
import os
from time import time
import re
import shutil
import glob
import warnings
import argparse
import logging

from datetime import datetime

# External Dependencies
import numpy as np
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

# NVTabular
import nvtabular as nvt
from nvtabular.ops import Categorify, Clip, FillMissing, HashBucket, LambdaOp, LogOp, Rename, get_embedding_sizes, Normalize
from nvtabular.io import Shuffle
from nvtabular.utils import _pynvml_mem_size, device_mem_size


def run_preprocessing(input_path, base_dir, num_train_days, num_val_days, num_gpus):

    # Define paths to save artifacts
    dask_workdir = os.path.join(base_dir, "test_dask/workdir")
    output_path = os.path.join(base_dir, "test_dask/output")
    stats_path = os.path.join(base_dir, "test_dask/stats")

    logging.info(f"Dask Workdir: {dask_workdir}")
    logging.info(f"Output Path: {output_path}")

    # Make sure we have a clean worker space for Dask
    if os.path.isdir(dask_workdir):
        shutil.rmtree(dask_workdir)
    os.makedirs(dask_workdir)

    # Make sure we have a clean stats space for Dask
    if os.path.isdir(stats_path):
        shutil.rmtree(stats_path)
    os.mkdir(stats_path)

    # Make sure we have a clean output path
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    logging.info("Created output directories..")

    # This requires the data to be in this specific format eg. day_0.parquet, day_2.parquet etc.
    fname = 'day_{}.parquet'
    num_days = len([i for i in os.listdir(input_path) if re.match(fname.format('[0-9]{1,2}'), i) is not None])
    train_paths = [os.path.join(input_path, fname.format(day)) for day in range(num_train_days)]
    valid_paths = [os.path.join(input_path, fname.format(day)) for day in range(num_train_days, num_train_days + num_val_days)]

    logging.info(f"Training data: {train_paths}")
    logging.info(f"Validation data: {valid_paths}")

    # Deploy a Dask Distributed Cluster
    # Single-Machine Multi-GPU Cluster
    protocol = "tcp"             # "tcp" or "ucx"
    visible_devices = ",".join([str(n) for n in num_gpus])  # Delect devices to place workers
    device_limit_frac = args.device_limit_frac      # Spill GPU-Worker memory to host at this limit.
    device_pool_frac = args.device_pool_frac
    part_mem_frac = args.part_mem_frac # Desired maximum size of each partition as a fraction of total GPU memory.

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    part_size = int(part_mem_frac * device_size)
    logging.info(f"Partition size: {part_size}")

    # Deploy Dask Distributed cluster only if asked for multiple GPUs
    if len(num_gpus) > 1:

        device_limit = int(device_limit_frac * device_size)
        device_pool_size = int(device_pool_frac * device_size)

        logging.info("Checking if any device memory is already occupied..")
        # Check if any device memory is already occupied
        for dev in visible_devices.split(","):
            fmem = _pynvml_mem_size(kind="free", index=int(dev))
            used = (device_size - fmem) / 1e9
            if used > 1.0:
                warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

        cluster = None               # (Optional) Specify existing scheduler port
        if cluster is None:
            cluster = LocalCUDACluster(
                protocol = protocol,
                n_workers=len(visible_devices.split(",")),
                CUDA_VISIBLE_DEVICES = visible_devices,
                device_memory_limit = device_limit,
                local_directory=dask_workdir
            )

        logging.info("Create the distributed client..")
        # Create the distributed client
        client = Client(cluster)

        logging.info("Initialize memory pools..")
        # Initialize RMM pool on ALL workers
        def _rmm_pool():
            rmm.reinitialize(
                # RMM may require the pool size to be a multiple of 256.
                pool_allocator=True,
                initial_pool_size=(device_pool_size // 256) * 256,
            )

        client.run(_rmm_pool)

    # Preprocessing
    CONTINUOUS_COLUMNS = ['I' + str(x) for x in range(1,14)]
    CATEGORICAL_COLUMNS =  ['C' + str(x) for x in range(1,27)]
    LABEL_COLUMNS = ['label']
    COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + LABEL_COLUMNS

    cat_features = CATEGORICAL_COLUMNS >> Categorify(out_path=stats_path)
    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + LABEL_COLUMNS

    logging.info("Defining a workflow object..")
    if len(num_gpus) > 1:
        workflow = nvt.Workflow(features, client=client)
    else:
        workflow = nvt.Workflow(features)

    dict_dtypes={}

    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in CONTINUOUS_COLUMNS:
        dict_dtypes[col] = np.float32

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32


    train_dataset = nvt.Dataset(train_paths, engine='parquet', part_size=part_size)
    valid_dataset = nvt.Dataset(valid_paths, engine='parquet', part_size=part_size)

    output_train_dir = os.path.join(output_path, 'train/')
    logging.info(f"Creating train/ directory at: {output_train_dir}")
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)

    output_valid_dir = os.path.join(output_path, 'valid/')
    logging.info(f"Creating valid/ directory at: {output_valid_dir}")
    if not os.path.exists(output_valid_dir):
        os.makedirs(output_valid_dir)

    start_time = datetime.now()
    logging.info(f"Starting fitting the preprocessing workflow on a training dataset. Datetime: {start_time}")
    workflow.fit(train_dataset)
    end_time = datetime.now()
    logging.info('Fitting completed. Datetime: {}, Elapsed time: {}'.format(end_time, end_time-start_time))
    

    start_time = datetime.now()
    logging.info(f"Starting  the preprocessing workflow on a training dataset. Datetime: {start_time}")
    workflow.transform(train_dataset).to_parquet(output_path=output_train_dir,
                                             shuffle=nvt.io.Shuffle.PER_PARTITION,
                                             out_files_per_proc=args.out_files_per_proc,
                                             dtypes=dict_dtypes,
                                             cats=CATEGORICAL_COLUMNS,
                                             conts=CONTINUOUS_COLUMNS,
                                             labels=LABEL_COLUMNS)
    end_time = datetime.now()
    logging.info('Processing completed. Datetime: {}, Elapsed time: {}'.format(end_time, end_time-start_time))

    start_time = datetime.now()
    logging.info(f"Starting the preprocessing workflow on a validation datasets. Datetime: {start_time}")
    workflow.transform(valid_dataset).to_parquet(output_path=output_valid_dir,
                                                 dtypes=dict_dtypes,
                                                 out_files_per_proc=args.out_files_per_proc,
                                                 cats=CATEGORICAL_COLUMNS,
                                                 conts=CONTINUOUS_COLUMNS,
                                                 labels=LABEL_COLUMNS)
    end_time = datetime.now()
    logging.info('Processing completed. Datetime: {}, Elapsed time: {}'.format(end_time, end_time-start_time))

    # use these printed out cardinalities list in the  "slot_size_array" in the HugeCTR training "dcn_parquet.json"
    cardinalities = []
    for col in CATEGORICAL_COLUMNS:
        cardinalities.append(nvt.ops.get_embedding_sizes(workflow)[col][0])

    logging.info(f"Cardinalities for configuring slot_size_array: {cardinalities}")

    logging.info(f"Saving workflow object at: {output_path + '/workflow'}")
    workflow.save(output_path + '/workflow')

    logging.info("Done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--input_data_dir',
                        type=str,
                        required=False,
                        default='/data/input',
                        help='Path to Preprocessed Test Data Dir. Default is /data/input')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=False,
                        default='/data/output',
                        help='Path for Output directory. It will create a directory "test_dask" to store artifacts. Default is /var/lib/data/criteo-data/')

    parser.add_argument('-t',
                        '--n_train_days',
                        type=int,
                        required=False,
                        default=2,
                        help='Number of Criteo data days to use for training dataset. Default is 1. Keep n_train_days + n_val_days<=24')

    parser.add_argument('-v',
                        '--n_val_days',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of Criteo data days to take for validation set after n_train_days. Default is 1. Keep n_train_days + n_val_days<=24.')

    #parser.add_argument('-g',
    #                    '--num_gpus',
    #                    nargs='+',
    #                    type=int,
    #                    required=False,
    #                    default=[0,1],
    #                    help='GPU devices to use for Preprocessing')
    
    parser.add_argument('-g',
                        '--num_gpus',
                        type=str,
                        required=False,
                        default="0,1",
                        help='GPU devices to use for Preprocessing')

    parser.add_argument('--part_mem_frac',
                        type=float,
                        required=False,
                        default=0.12,
                        help='Desired maximum size of each partition as a fraction of total GPU memory')
    
    parser.add_argument('--device_limit_frac',
                        type=float,
                        required=False,
                        default=0.7,
                        help='Device limit fraction')

    parser.add_argument('--device_pool_frac',
                        type=float,
                        required=False,
                        default=0.7,
                        help='Device pool fraction')

    parser.add_argument('--out_files_per_proc',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of files per processor')

    args = parser.parse_args()
    args.num_gpus = list(map(int, args.num_gpus.split(',')))

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    start_time = datetime.now()
    run_preprocessing(input_path=args.input_data_dir,
                    base_dir=args.output_dir,
                    num_train_days=args.n_train_days,
                    num_val_days=args.n_val_days,
                    num_gpus=args.num_gpus)
    end_time = datetime.now()
    logging.info('The job completed successfully. Datetime: {}, Total Elapsed time: {}'.format(end_time, end_time-start_time))