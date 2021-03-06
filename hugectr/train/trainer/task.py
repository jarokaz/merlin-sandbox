import argparse
import json
import logging
import os
import time


from trainer.model import create_model

SNAPSHOT_PREFIX = 'deepfm'

def main(args):

    repeat_dataset = False if args.num_epochs > 0 else True

    model = create_model(
        train_data=[args.train_data],
        valid_data=args.valid_data,
        slot_size_array=args.slot_size_array,
        batchsize=args.batchsize,
        lr=args.lr,
        gpus=args.gpus,
        repeat_dataset=repeat_dataset)

    model.summary()

    model.fit(
        num_epochs=args.num_epochs,
        max_iter=args.max_iter,
        display=args.display_interval, 
        eval_interval=args.eval_interval, 
        snapshot=args.snapshot, 
        snapshot_prefix=SNAPSHOT_PREFIX)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train_data',
                        type=str,
                        required=False,
                        default='/criteo_data/output/train/_file_list.txt',
                        help='Path to training data _file_list.txt')
    parser.add_argument('-v',
                        '--valid_data',
                        type=str,
                        required=False,
                        default='/criteo_data//output/valid/_file_list.txt',
                        help='Path to validation data _file_list.txt')
    parser.add_argument('-i',
                        '--max_iter',
                        type=int,
                        required=False,
                        default=1000,
                        help='Number of training iterations')
    parser.add_argument('--num_epochs',
                        type=int,
                        required=False,
                        default=0,
                        help='Number of training epochs')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=16384,
                        help='Batch size')
    parser.add_argument('-s',
                        '--snapshot',
                        type=int,
                        required=False,
                        default=0,
                        help='Saves a model snapshot after given number of iterations')
    parser.add_argument('--gpus',
                        type=str,
                        required=False,
                        default="[[0]]",
                        help='GPU devices to use for Preprocessing')
    parser.add_argument('-r',
                        '--eval_interval',
                        type=int,
                        required=False,
                        default=1000,
                        help='Run evaluation after given number of iterations')
    parser.add_argument('--display_interval',
                        type=int,
                        required=False,
                        default=200,
                        help='Display progress after given number of iterations')
    parser.add_argument('--slot_size_array',
                        type=str,
                        required=False,
                        default="[2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35]", 
                        help='Categorical variables cardinalities')
    parser.add_argument('--workspace_size_per_gpu',
                        type=int,
                        required=False,
                        default=1000,
                        help='Workspace size per gpu in MB')
    parser.add_argument('--lr',
                        type=float,
                        required=False,
                        default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    return args  

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    args = parse_args()
    args.gpus = json.loads(args.gpus)
    args.slot_size_array = json.loads(args.slot_size_array)


    logging.info(f"Args: {args}")
    start_time = time.time()
    logging.info("Starting training")

    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Training completed. Elapsed time: {}".format(elapsed_time))

