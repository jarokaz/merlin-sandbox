

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import json
import logging
import os
import re
import shutil
import warnings
import time
import numpy as np

import nvtabular as nvt
import rmm

from datetime import datetime
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from nvtabular.ops import (
    Categorify,
    Clip,
    FillMissing,
    Normalize,
)
from nvtabular.utils import _pynvml_mem_size, device_mem_size


BASE_DIR = '/tmp'
DASK_CLUSTER_PROTOCOL = 'tcp'
DASHBOARD_PORT = '8787'

# Criteo columns
CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
LABEL_COLUMNS = ["label"]


def get_args():
    """Defines and parse commandline arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        default="/tmp",
        type=str,
    )
    
    parser.add_argument(
        "--training_data",
        default="/tmp/training",
        type=str,
    )
    
    parser.add_argument(
        "--validation_data",
        default="/tmp/validation",
        type=str,
    )
        
    parser.add_argument(
        "--gpus",
        default="0,1",
        type=str,
    )
        
    parser.add_argument(
        "--device_limit_frac",
        default=0.7,
        type=float,
    )
    
    parser.add_argument(
        "--device_pool_frac",
        default=0.8,
        type=float,
    )
    
    parser.add_argument(
        "--part_mem_frac",
        default=0.1,
        type=float,
    )

    return parser.parse_args()


def create_dask_cuda_cluster(
    gpus,
    device_size,
    device_limit_frac,
    device_pool_frac,
    dask_workdir,
):
    
    # Initialize RMM pool on ALL workers
    def _rmm_pool():
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,  # Use default size
        )
    
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    
    # Check if any device memory is already occupied
    for dev in gpus.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")
            
    cluster = LocalCUDACluster(
        protocol=DASK_CLUSTER_PROTOCOL,
        n_workers=len(gpus.split(",")),
        CUDA_VISIBLE_DEVICES=gpus,
        device_memory_limit=device_limit,
        local_directory=dask_workdir
    )  
    
    client = Client(cluster)
    client.run(_rmm_pool)
    
    return client


def create_preprocessing_workflow(
    client,
    stats_path,
    num_buckets=10000000
):
    
    categorify_op = Categorify(out_path=stats_path, max_size=num_buckets)
    cat_features = CATEGORICAL_COLUMNS >> categorify_op
    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + LABEL_COLUMNS
    workflow = nvt.Workflow(features, client=client)
    
    return workflow

def create_datasets(
    train_paths,
    valid_paths,
    part_mem_frac,
    device_size,
):
    
    dict_dtypes = {}

    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in CONTINUOUS_COLUMNS:
        dict_dtypes[col] = np.float32

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32
        
    part_size = int(part_mem_frac * device_size)
    train_dataset = nvt.Dataset(train_paths, engine="parquet", part_size=part_size)
    valid_dataset = nvt.Dataset(valid_paths, engine="parquet", part_size=part_size)
    
    return dict_dtypes, train_dataset, valid_dataset
    

def main():
    args = get_args()
    

    dask_workdir = os.path.join(BASE_DIR, "test_dask/workdir")
    stats_path = os.path.join(BASE_DIR, "test_dask/stats")

    # Make sure we have a clean worker space for Dask
    if os.path.isdir(dask_workdir):
        shutil.rmtree(dask_workdir)
    os.makedirs(dask_workdir)

    # Make sure we have a clean stats space for Dask
    if os.path.isdir(stats_path):
        shutil.rmtree(stats_path)
    os.mkdir(stats_path)
    

    fname = "day_{}.parquet"
    train_paths = [
        os.path.join(args.training_data, filename) for filename in os.listdir(args.training_data)]
    valid_paths = [
        os.path.join(args.validation_data, filename) for filename in os.listdir(args.validation_data)]
    
    logging.info(f"Training data path: {train_paths}")
    logging.info(f"Validation data path: {valid_paths}")
    
    logging.info("Creating Dask-Cuda cluster")
    device_size = device_mem_size(kind="total")
    client = create_dask_cuda_cluster(
        gpus=args.gpus,
        device_size=device_size,
        device_limit_frac=args.device_limit_frac,
        device_pool_frac=args.device_pool_frac,
        dask_workdir=dask_workdir,
    )
    logging.info("Cluster created")
    logging.info(str(client))
    
    logging.info("Creating workflow")
    workflow = create_preprocessing_workflow(
        client=client,
        stats_path=stats_path)
    logging.info("Workflow created")
    
    logging.info("Creating datasets")
    dict_dtypes, train_dataset, valid_dataset = create_datasets(
        train_paths=train_paths,
        valid_paths=train_paths,
        part_mem_frac=args.part_mem_frac,
        device_size=device_size,
    )
    logging.info("Datasets created")
    
    start_time = datetime.now()
    logging.info(f"Starting fitting the preprocessing workflow on a training dataset. Datetime: {start_time}")
    workflow.fit(train_dataset)
    end_time = datetime.now()
    logging.info('Fitting completed. Datetime: {}, Elapsed time: {}'.format(end_time, end_time-start_time))
    
    start_time = datetime.now()
    logging.info(f"Starting  the preprocessing workflow on a training dataset. Datetime: {start_time}")
    workflow.transform(train_dataset).to_parquet(
        output_path=f'{args.output_path}/train',
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        dtypes=dict_dtypes,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )
    end_time = datetime.now()
    logging.info('Processing completed. Datetime: {}, Elapsed time: {}'.format(end_time, end_time-start_time))
    
    start_time = datetime.now()
    logging.info(f"Starting the preprocessing workflow on a validation datasets. Datetime: {start_time}")
    workflow.transform(valid_dataset).to_parquet(
        output_path=f'{args.output_path}/valid',
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        dtypes=dict_dtypes,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )
    end_time = datetime.now()
    logging.info('Processing completed. Datetime: {}, Elapsed time: {}'.format(end_time, end_time-start_time))
    
    logging.info(f"Saving workflow to {args.output_path}")
    workflow.save(os.path.join(args.output_path, "workflow"))
    logging.info("Workflow saved")
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
