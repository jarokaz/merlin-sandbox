import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 300,
                              batchsize_eval = 4096,
                              batchsize = 4096,
                              lr = 0.001,
                              vvgpu = [[0,1]],
                              repeat_dataset = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,
                                  source = ["/criteo_data/output/train/_file_list.txt"],
                                  eval_source = "/criteo_data/output/valid/_file_list_test.txt",
                                  slot_size_array = [2839307, 28141, 15313, 7229, 19673, 4, 6558, 1297, 63, 2156343, 327548, 178478, 11, 2208, 9517, 73, 4, 957, 15, 2893928, 1166099, 2636476, 211349, 10776, 92, 35],
                                  check_type = hugectr.Check_t.Non)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0000001)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 2000,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["concat1"],
                            top_names = ["slice11", "slice12"],
                            ranges=[(0,429),(0,429)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                            bottom_names = ["slice11"],
                            top_names = ["multicross1"],
                            num_layers=6))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["slice12"],
                            top_names = ["fc1"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu1"],
                            top_names = ["dropout1"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout1"],
                            top_names = ["fc2"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu2"],
                            top_names = ["dropout2"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout2"],
                            top_names = ["fc3"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc3"],
                            top_names = ["relu3"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu3"],
                            top_names = ["dropout3"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout3"],
                            top_names = ["fc4"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc4"],
                            top_names = ["relu4"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu4"],
                            top_names = ["dropout4"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout4"],
                            top_names = ["fc5"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc5"],
                            top_names = ["relu5"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu5"],
                            top_names = ["dropout5"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout5"],
                            top_names = ["fc6"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc6"],
                            top_names = ["relu6"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu6"],
                            top_names = ["dropout6"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout6"],
                            top_names = ["fc7"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc7"],
                            top_names = ["relu7"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu7"],
                            top_names = ["dropout7"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["dropout7", "multicross1"],
                            top_names = ["concat2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["concat2"],
                            top_names = ["fc8"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 2300, display = 200, eval_interval = 1000, snapshot = 1000000, snapshot_prefix = "dcn")

