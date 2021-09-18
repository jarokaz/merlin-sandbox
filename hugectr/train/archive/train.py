import argparse
import logging
import os
import time

import hugectr

from mpi4py import MPI


def train(
    train_data,
    valid_data, 
    slot_size_array,
    max_iter,
    num_epochs,
    snapshot,
    eval_interval,
    display_interval,
    batchsize,
    workspace_size_per_gpu,
    gpus):

    solver = hugectr.CreateSolver(max_eval_batches = 300,
                                  batchsize_eval = batchsize,
                                  batchsize = batchsize,
                                  lr = 0.001,
                                  vvgpu = gpus,
                                  #repeat_dataset = False,
                                  repeat_dataset = False,
                                  i64_input_key = True)

    reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                      source = [train_data],
                                      eval_source = valid_data,
                                      check_type = hugectr.Check_t.Non,
                                      slot_size_array = slot_size_array)

    optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                        update_type = hugectr.Update_t.Local,
                                        beta1 = 0.9,
                                        beta2 = 0.999,
                                        epsilon = 0.0000001)

    model = hugectr.Model(solver, reader, optimizer)

    model.add(hugectr.Input(label_dim = 1, label_name = "label",
                            dense_dim = 0, dense_name = "dense",
                            data_reader_sparse_param_array = 
                            [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                                workspace_size_per_gpu_in_mb = workspace_size_per_gpu,
                                embedding_vec_size = 64,
                                combiner = "sum",
                                sparse_embedding_name = "sparse_embedding1",
                                bottom_name = "data1",
                                optimizer = optimizer))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                                bottom_names = ["sparse_embedding1"],
                                top_names = ["reshape1"],
                                leading_dim=1664))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["reshape1"],
                                top_names = ["fc1"],
                                num_output=200))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc1"],
                                top_names = ["relu1"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu1"],
                                top_names = ["fc2"],
                                num_output=200))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc2"],
                                top_names = ["relu2"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu2"],
                                top_names = ["fc3"],
                                num_output=200))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                                bottom_names = ["fc3"],
                                top_names = ["relu3"]))
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                                bottom_names = ["relu3"],
                                top_names = ["fc4"],
                                num_output=1))                                                                                           
    model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                                bottom_names = ["fc4", "label"],
                                top_names = ["loss"]))
 
    model.compile()
    model.summary()
    model.fit(#max_iter = max_iter, i
              num_epochs=num_epochs,
              display = display_interval, 
              eval_interval = eval_interval, 
              snapshot = snapshot, 
              snapshot_prefix = "deepfm")


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
                        default=20000,
                        help='Number of training iterations')
    parser.add_argument('--num_epochs',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of training epochs')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=8192,
                        help='Batch size')
    parser.add_argument('-s',
                        '--snapshot',
                        type=int,
                        required=False,
                        default=10000,
                        help='Saves a model snapshot after given number of iterations')
    parser.add_argument('-g',
                        '--gpus',
                        type=str,
                        required=False,
                        default="0,1,3,4",
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
                        default=100,
                        help='Display progress after given number of iterations')
    parser.add_argument('--slot_size_array',
                        type=str,
                        required=False,
                        default='2839307,28141,15313,7229,19673,4,6558,1297,63,2156343,327548,178478,11,2208,9517,73,4,957,15,2893928,1166099,2636476,211349,10776,92,35',
                        help='Categorical variables cardinalities')
    parser.add_argument('--workspace_size_per_gpu',
                        type=int,
                        required=False,
                        default=1000,
                        help='Workspace size per gpu in MB')

    args = parser.parse_args()

    return args  

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    args = parse_args()
    args.gpus = list(map(int, args.gpus.split(',')))
    args.slot_size_array = list(map(int, args.slot_size_array.split(','))) 
    logging.info(f"Args: {args}")

    start_time = time.time()
    logging.info("Starting training")

    train(
        train_data=args.train_data,
        valid_data=args.valid_data, 
        slot_size_array=args.slot_size_array,
        max_iter=args.max_iter,
        num_epochs=args.num_epochs,
        snapshot=args.snapshot,
        eval_interval=args.eval_interval,
        display_interval=args.display_interval,
        batchsize=args.batchsize,
        workspace_size_per_gpu=args.workspace_size_per_gpu,
        gpus=[args.gpus])

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Training completed. Elapsed time: {}".format(elapsed_time))

