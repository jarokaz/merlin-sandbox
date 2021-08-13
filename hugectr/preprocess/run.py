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
import argparse
import logging
import pprint
import time

from google.cloud import aiplatform


PREPROCESS_FILE = 'preprocess.py'


def run(args):

    aiplatform.init(
        project=args.project,
        location=args.region,
        staging_bucket=args.gcs_bucket
    )

    job_name = 'NVT_{}'.format(time.strftime("%Y%m%d_%H%M%S"))

    worker_pool_specs =  [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": args.accelerator_num,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.preprocess_image,
                "command": ["python", PREPROCESS_FILE],
                "args": [
                    '--input_data_dir=' + args.input_data_dir, 
                    '--output_dir=' + f'{args.output_dir}/{job_name}',
                    '--n_train_days=' + str(args.n_train_days),
                    '--n_val_days=' + str(args.n_val_days), 
                    '--device_limit_frac=' + str(args.device_limit_frac), 
                    '--device_pool_frac=' + str(args.device_pool_frac), 
                    '--part_mem_frac=' + str(args.part_mem_frac),
                    #'--num_gpus=' + str(args.num_gpus),
                ],
            },
        }
    ]

    logging.info(f'Starting job: {job_name}')

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
    )
    job.run(sync=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        type=str,
                        default='jk-mlops-dev',
                        help='Project ID')
    parser.add_argument('--region',
                        type=str,
                        default='us-central1',
                        help='Region')
    parser.add_argument('--gcs_bucket',
                        type=str,
                        default='gs://jk-vertex-us-central1',
                        help='GCS bucket')
    parser.add_argument('--vertex_sa',
                        type=str,
                        default='training-sa@jk-mlops-dev.iam.gserviceaccount.com',
                        help='Vertex SA')
    parser.add_argument('--machine_type',
                        type=str,
                        default='a2-highgpu-2g',
                        help='Machine type')
    parser.add_argument('--accelerator_type',
                        type=str,
                        default='NVIDIA_TESLA_A100',
                        help='Accelerator type')
    parser.add_argument('--accelerator_num',
                        type=int,
                        default=2,
                        help='Num of GPUs')
    parser.add_argument('--input_data_dir',
                        type=str,
                        default='/gcs/jk-vertex-us-central1/criteo-parquet/criteo-parque',
                        help='Criteo parquet data location')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/gcs/jk-vertex-us-central1/nvt-jobs',
                        help='Output GCS location')
    parser.add_argument('--preprocess_image',
                        type=str,
                        default='gcr.io/jk-mlops-dev/merlin-preprocess',
                        help='Training image name')
    parser.add_argument('--n_train_days',
                        type=int,
                        default=2,
                        help='Num of train days')
    parser.add_argument('--n_val_days',
                        type=int,
                        default=1,
                        help='Num of validation days')
    parser.add_argument('--num_gpus',
                        nargs='+',
                        type=int,
                        default=[0,1],
                        help='GPU devices to use for Preprocessing')
    parser.add_argument('--part_mem_frac',
                        type=float,
                        required=False,
                        default=0.10,
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

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    logging.info(f"Args: {args}")

    run(args)