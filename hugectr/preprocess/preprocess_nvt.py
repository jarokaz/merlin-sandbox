import os
from re import I
import sys
import argparse
import glob
import logging
import numpy as np
import shutil
import time
import warnings

import cudf
from cudf.io.parquet import ParquetWriter

import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed

import rmm

import nvtabular as nvt
from nvtabular.io import Shuffle
from nvtabular.utils import _pynvml_mem_size, device_mem_size
from nvtabular.ops import Categorify, Clip, FillMissing, HashBucket, LambdaOp, Normalize, Rename, Operator, get_embedding_sizes


# define dataset schema
CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]
CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
LABEL_COLUMNS = ['label']
COLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS +  CATEGORICAL_COLUMNS


def bytesto(bytes, to, bsize=1024):
    """Computes a partition size in GBs."""
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** a[to])


def create_preprocessing_workflow(
    categorical_columns,
    continuous_columns,
    label_columns,
    freq_limit, 
    stats_path):
    """Defines a preprocessing graph."""

    categorify_op = Categorify(freq_threshold=freq_limit, out_path=stats_path)
    cat_features = categorical_columns >> categorify_op
    cont_features = continuous_columns >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + label_columns

    dict_dtypes={}
    for col in categorical_columns:
        dict_dtypes[col] = np.int64
    for col in continuous_columns:
        dict_dtypes[col] = np.float32
    for col in label_columns:
        dict_dtypes[col] = np.float32

    return features, dict_dtypes


def create_dask_cluster(
    protocol,
    gpus,
    device_memory_limit,
    device_pool_size,
):
    """Deploys DASK cluster."""

    def _rmm_pool():
        rmm.reinitialize(
            # RMM may require the pool size to be a multiple of 256.
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,
        ) 

    if protocol == "ucx":
        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")
        os.environ["UCX_TLS"] = UCX_TLS
        cluster = LocalCUDACluster(
            protocol = protocol,
            CUDA_VISIBLE_DEVICES = ','.join(gpus),
            n_workers = len(gpus),
            enable_nvlink=True,
            device_memory_limit = device_memory_limit,
            )
    else:
        cluster = LocalCUDACluster(
            protocol = protocol,
            n_workers = len(gpus),
            CUDA_VISIBLE_DEVICES = ','.join(gpus),
            device_memory_limit = device_memory_limit,
            )
        client = Client(cluster)

    if device_pool_size > 256:
        client.run(_rmm_pool)

    return client

def preprocess(args):
    "Preprocesses training and validation splits."

    train_output_folder = os.path.join(args.output_folder, 'train') 
    valid_output_folder = os.path.join(args.output_folder, 'valid') 
    stats_output_folder = os.path.join(args.output_folder, 'stats')
    workflow_output_folder = os.path.join(args.output_folder, 'workflow')

    # Make sure we have a clean parquet space for cudf conversion
    for one_path in [train_output_folder, valid_output_folder, stats_output_folder, workflow_output_folder]:
        if os.path.exists(one_path):
            shutil.rmtree(one_path)
        os.makedirs(one_path)

    logging.info("Checking if any device memory is already occupied..")
    gpus = args.devices.split(',')
    device_size = device_mem_size(kind="total")
    for dev in gpus:
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    client = None
    if len(gpus) > 1:
        logging.info("Checking if any device memory is already occupied..")
        client = create_dask_cluster(
            protocol = args.protocol, 
            gpus=gpus,
            device_memory_limit= int(args.device_limit_frac * device_size),
            device_pool_size=int(args.device_pool_frac * device_size)) 

    train_paths = glob.glob(os.path.join(args.train_folder, "*.parquet"))
    valid_paths = glob.glob(os.path.join(args.valid_folder, "*.parquet"))
    part_size = int(args.part_mem_frac * device_size)

    elapsed_times = {} 
    processing_start_time = time.time()

    logging.info("Creating datasets...")
    train_dataset = nvt.Dataset(train_paths, engine='parquet', part_size=part_size)
    valid_dataset = nvt.Dataset(valid_paths, engine='parquet', part_size=part_size)

    features, dict_dtypes = create_preprocessing_workflow(
        categorical_columns=CATEGORICAL_COLUMNS,
        continuous_columns=CONTINUOUS_COLUMNS,
        label_columns=LABEL_COLUMNS,
        freq_limit=args.freq_limit, 
        stats_path=stats_output_folder)

    workflow = nvt.Workflow(features, client=client)

    logging.info('Fitting the workflow') 
     
    start_time = time.time()  
    workflow.fit(train_dataset)
    end_time = time.time()
    elapsed_times['Workflow fitting'] = end_time - start_time
    logging.info('Fitting completed.')

    shuffle = None
    if args.shuffle == "PER_WORKER":
        shuffle = nvt.io.Shuffle.PER_WORKER
    elif args.shuffle == "PER_PARTITION":
        shuffle = nvt.io.Shuffle.PER_PARTITION

    for dataset, output_folder, name in zip([train_dataset, valid_dataset], 
                                            [train_output_folder, valid_output_folder],
                                            ['Training dataset processing', 'Validation dataset processing']):
        logging.info(f'{name} .....')
        start_time = time.time()
        workflow.transform(train_dataset).to_parquet(
            output_path=output_folder,
            dtypes=dict_dtypes,
            cats=CATEGORICAL_COLUMNS,
            conts=CONTINUOUS_COLUMNS,
            labels=LABEL_COLUMNS,
            shuffle=shuffle,
            out_files_per_proc=args.out_files_per_proc,
            num_threads=args.num_io_threads)
        end_time = time.time()
        elapsed_times[name] = end_time - start_time
        logging.info('Processing completed.')    

    cardinalities = []
    for col in CATEGORICAL_COLUMNS:
        cardinalities.append(nvt.ops.get_embedding_sizes(workflow)[col][0])
    
    logging.info(f"Cardinalities for configuring slot_size_array: {cardinalities}")

    logging.info(f"Saving workflow object at: {workflow_output_folder}")
    workflow.save(workflow_output_folder)

    ## Shutdown clusters
    client.close()

    end_time = time.time()
    elapsed_times['Total processing time'] = end_time - processing_start_time    

    logging.info("Dask-NVTabular Criteo Preprocessing")
    logging.info("--------------------------------------")
    logging.info(f"train_dir          | {args.train_folder}")
    logging.info(f"valid_dir          | {args.valid_folder}")
    logging.info(f"output_dir         | {args.output_folder}")
    logging.info(f"partition size     | {'%.2f GB'%bytesto(int(args.part_mem_frac * device_size),'g')}")
    logging.info(f"protocol           | {args.protocol}")
    logging.info(f"device(s)          | {args.devices}")
    logging.info(f"rmm-pool-frac      | {(args.device_pool_frac)}")
    logging.info(f"out-files-per-proc | {args.out_files_per_proc}")
    logging.info(f"num_io_threads     | {args.num_io_threads}")
    logging.info(f"shuffle            | {args.shuffle}")
    logging.info("======================================")
    for elapsed_time_name, elapsed_time in elapsed_times.items():
        logging.info(f"{elapsed_time_name}         | {elapsed_time}")
    logging.info("======================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description=("Multi-GPU Criteo Preprocessing"))

    parser.add_argument("--train_folder", type=str, help="Training data folder(Required)")
    parser.add_argument("--valid_folder", type=str, help="Valid data folder (Required)")
    parser.add_argument("--output_folder", type=str, help="Directory path to write output (Required)")
    parser.add_argument(
        "-d",
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use (Default 'tcp')",
    )
    parser.add_argument(
        "--device_limit_frac",
        default=0.7,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "
    )
    parser.add_argument(
        "--device_pool_frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "
        "The RMM pool frac is the same for all GPUs, make sure each one has enough memory size",
    )
    parser.add_argument(
        "--num_io_threads",
        default=0,
        type=int,
        help="Number of threads to use when writing output data (Default 0). "
        "If 0 is specified, multi-threading will not be used for IO.",
    )

    #
    # Data-Decomposition Parameters
    #

    parser.add_argument(
        "--part_mem_frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )
    parser.add_argument(
        "--out_files_per_proc",
        default=8,
        type=int,
        help="Number of output files to write on each worker (Default 8)",
    )

    #
    # Preprocessing Options
    #

    parser.add_argument(
        "-f",
        "--freq_limit",
        default=0,
        type=int,
        help="Frequency limit for categorical encoding (Default 0)",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        choices=["PER_WORKER", "PER_PARTITION", "NONE"],
        default="PER_PARTITION",
        help="Shuffle algorithm to use when writing output data to disk (Default PER_PARTITION)",
    )


    args = parser.parse_args()
    args.n_workers = len(args.devices.split(","))
    return args

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.root.setLevel(logging.NOTSET)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    args = parse_args()

    preprocess(args)
