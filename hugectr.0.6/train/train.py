import argparse
import logging
import os
import time

import hugectr

from mpi4py import MPI

SLOT_SIZE_ARRAY = [1461, 558, 335378, 211710, 306, 20, 12136, 634, 4, 51298, 5302, 332600, 3179, 27, 12191, 301211, 11, 4841, 2086, 4, 324273, 17, 16, 79734, 96, 58622] 

def build_model(
    solver,
    reader,
    optimizer,

):
    model = hugectr.Model(solver, reader, optimizer)
    model.add(hugectr.Input(label_dim = 1, label_name = "label",
                            dense_dim = 0, dense_name = "dense",
                            data_reader_sparse_param_array = 
                            [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))
    model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                                workspace_size_per_gpu_in_mb = 425,
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

    return model


def train(
    input_train,
    input_val,
    max_iter,
    snapshot,
    eval_interval,
    batchsize,
    gpus):

    solver = hugectr.CreateSolver(max_eval_batches = 300,
                                  batchsize_eval = batchsize,
                                  batchsize = batchsize,
                                  lr = 0.001,
                                  vvgpu = gpus,
                                  repeat_dataset = True,
                                  i64_input_key = True)
    reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                      source = [input_train],
                                      eval_source = input_val,
                                      check_type = hugectr.Check_t.Non,
                                      slot_size_array = [1461, 558, 335378, 211710, 306, 20, 12136, 634, 4, 51298, 5302, 332600, 3179, 27, 12191, 301211, 11, 4841, 2086, 4, 324273, 17, 16, 79734, 96, 58622])
    optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                        update_type = hugectr.Update_t.Local,
                                        beta1 = 0.9,
                                        beta2 = 0.999,
                                        epsilon = 0.0000001)

    model = build_model(solver, reader, optimizer)
    model.compile()
    model.summary()
    model.fit(max_iter = max_iter, 
              display = 10, 
              eval_interval = eval_interval, 
              snapshot = snapshot, 
              snapshot_prefix = "deepfm")


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--input_train',
                        type=str,
                        required=False,
                        default='/data/output/test_dask/output/train/_file_list.txt',
                        help='Path to training data _file_list.txt')

    parser.add_argument('-v',
                        '--input_val',
                        type=str,
                        required=False,
                        default='/data/output/test_dask/output/valid/_file_list.txt',
                        help='Path to validation data _file_list.txt')

    parser.add_argument('-i',
                        '--max_iter',
                        type=int,
                        required=False,
                        default=20000,
                        help='Number of training iterations')

    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=2048,
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
                        default="0,1",
                        help='GPU devices to use for Preprocessing')

    parser.add_argument('-r',
                        '--eval_interval',
                        type=int,
                        required=False,
                        default=1000,
                        help='Run evaluation after given number of iterations')

    args = parser.parse_args()

    return args  

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')

    args = parse_args()
    args.gpus = list(map(int, args.gpus.split(',')))
    logging.info(f"Args: {args}")

    start_time = time.time()
    logging.info("Starting training")
    train(
        input_train=args.input_train,
        input_val=args.input_val,
        max_iter=args.max_iter,
        snapshot=args.snapshot,
        eval_interval=args.eval_interval,
        batchsize=args.batchsize,
        gpus=[args.gpus])
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Training completed. Elapsed time: {}".format(elapsed_time))

