import dgl
import numpy as np
import os
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import GraphConv
from signal import signal, SIGINT
from sys import exit
from threading import Thread
from torch import cuda
from torch.multiprocessing import Process, set_start_method
import time

import adapm

from cli import parse_arguments
from data import PartitionedOGBDataset
from common.ps_models import PSDense, PSEmbedGraphConv
from common.optimizer import PSAdagrad, ReduceLROnPlateau
from common.iterator import PrefetchIterator
import common.process_setup

class Model(nn.Module):
    def __init__(self, num_nodes, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = PSEmbedGraphConv(num_nodes, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.h_feats = h_feats

    def forward(self, mfgs, inputs):
        h = self.conv1(mfgs[0], inputs)
        h = F.relu(h)
        h = self.conv2(mfgs[1], h)
        return h

def run_worker(worker_id, args, kv):
    train(worker_id, args, kv)
    kv.barrier() # wait for all workers to finish
    kv.finalize()

def train(worker_id, args, kv):
    print(f"Worker {worker_id} training on {args.device}")

    opt = PSAdagrad(
        lr = args.learning_rate,
        initial_accumulator_value=args.initial_accumulator_value,
        eps = args.epsilon,
    )
    scheduler = ReduceLROnPlateau(
        opt,
        factor = args.rlop_factor,
        mode = "max",
        patience = args.rlop_patience,
        cooldown = args.rlop_cooldown,
        threshold = args.rlop_threshold,
        min_lr = args.rlop_min_lr,
        eps = args.rlop_eps,
    )

    print("loading training split")
    if args.world_size*args.workers_per_node > 1:
        train_ids = args.data.load_split(worker_id)
    else:
        train_ids = args.data.idx_split["train"]
    print("creating dataloaders..")
    sampler = dgl.dataloading.NeighborSampler([args.num_sampling_neighbors, args.num_sampling_neighbors])
    train_dataloader = dgl.dataloading.DataLoader(
        args.data.graph,    # The graph
        train_ids,          # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        #device=args.device, # Put the sampled MFGs on CPU or GPU
        use_ddp=False,      # Make it work with distributed data parallel
        batch_size=args.batch_size, # Per-device batch size.
                            # The effective batch size is this number times the number of GPUs.
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        args.data.graph,
        args.data.idx_split["valid"],
        sampler,
        device=args.device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    kv.begin_setup()
    print("init kv-parameters..")
    model = args.model
    torch.manual_seed(args.model_seed)
    model.kv_init(kv, 0, opt, worker_id==0, args.intent_ahead!=0)
    model.to(args.device)
    kv.end_setup()
    kv.wait_sync()
    print("init done.")

    # full replication
    if args.enforce_full_replication > 0:
        print(f"Enforcing full replication: signal intent for keys 0..{args.num_keys} in time [0, {sys.maxsize}]")
        all_keys = torch.arange(args.num_keys)
        for keys in all_keys.split(2**20):
            kv.intent(keys, 0, sys.maxsize)
            time.sleep(3)
    kv.wait_sync()
    kv.wait_sync()

    def collate(batch):
        input_nodes, _, mfgs = batch
        if args.intent_ahead != 0:
            target_time = kv.current_clock() + args.intent_ahead
            model.model.conv1.intent(input_nodes, target_time)
        return input_nodes, [block.to(args.device, non_blocking=True) for block in mfgs]

    best_accuracy = 0
    best_model_path = './model.pt'
    total_train_time = 0

    kv.barrier()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        with tqdm.tqdm(PrefetchIterator(kv, train_dataloader, args.intent_ahead, collate=collate),  position=worker_id, disable=(not args.progress_bar)) as tq:
            for step, (input_nodes, mfgs) in enumerate(tq):
                model.pull()

                labels = mfgs[-1].dstdata['label'].long()
                predictions = model(mfgs, input_nodes)

                loss = F.cross_entropy(predictions, labels)
                loss.backward()

                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
                kv.advance_clock()

        # synchronize replicas
        kv.wait_sync()
        kv.barrier()
        kv.wait_sync()
        epoch_stop = time.time()
        total_train_time += epoch_stop - epoch_start

        model.eval()

        # Evaluate on only the first GPU.
        if worker_id == 0:
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader, disable=(not args.progress_bar)) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['_ID']
                    labels.append(mfgs[-1].dstdata['label'].long().cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print(f"All workers finished epoch {epoch+1} (epoch: {epoch_stop-epoch_start:.3f}s, total: {total_train_time:.3f}s). Validation accuracy: {accuracy}", flush=True)
                if args.reduce_lr_on_plateau:
                    scheduler.step(accuracy)

        # synchronize workers
        kv.barrier()

        # maximum time
        if (args.max_runtime != 0 and
                (total_train_time > args.max_runtime or
                 total_train_time + (epoch_stop-epoch_start) > args.max_runtime * 1.05)):
            print(f"Worker {worker_id} stops after epoch {epoch+1} because max. time is reached: {total_train_time}s (+ 1 epoch) > {args.max_runtime}s", flush=True)
            break

processes = []
if __name__ == "__main__":
    # run cli
    args = parse_arguments()

    # read environment variables when running with tracker
    if args.tracker:
        args = common.process_setup.configure_tracker_launch(args)

    # load the dataset
    if args.nodes == 0 and args.external_num_keys != None:
        # no need to load the dataset in the scheduler if the number of keys was passed externally
        print(f"skipping the dataset read in the scheduler. manually setting num_keys={args.external_num_keys} instead")
        args.num_keys = args.external_num_keys
    else:
        # setup dataset
        print(f"loading dataset: {args.dataset}")
        args.data = PartitionedOGBDataset(args.dataset, args.world_size*args.workers_per_node, args.data_root)

        # setup optimizer and model
        print(f"create model with {args.data.num_features()} nodes")
        args.model = PSDense(Model(args.data.num_features(), args.embedding_dim, args.data.num_classes()))

        # calculate parameter lens
        args.lens = args.model.lens()
        args.num_keys = len(args.lens)

        if args.external_num_keys != None:
            assert args.num_keys == args.external_num_keys

    # start the necessary processes on this node
    common.process_setup.start_processes(worker_func=run_worker, args=args)
