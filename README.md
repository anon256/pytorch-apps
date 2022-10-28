# Install

Dependencies are [PyTorch](https://pytorch.org/), [DGL (Deep Graph Library)](https://www.dgl.ai/), and [Open Graph Benchmark](https://ogb.stanford.edu/). To install with CUDA 11.6:

```bash
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install ogb
```

To install [AdaPM](https://github.com/anon256/AdaPM/) and its PyTorch bindings, [**follow the steps described here**](https://github.com/anon256/AdaPM/tree/main/bindings#installation).

Get the code
```bash
git clone https://github.com/anon256/pytorch-apps/
cd pytorch-apps
```

Modify `PYTHONPATH` so Python finds modules located in the current work directory.
```bash
export PYTHONPATH="${PYTHONPATH}:."
# ALTERNATIVE:
# export PYTHONPATH="${PYTHONPATH}:/path/to/code/directory/"
```
# Usage examples

## GCN: Training graph convolutional networks

To run on the example data:

```bash
python gcn/run.py --dataset ogbn-arxiv --data_root example_data/gnn/ --no_cuda
```

See `python gcn/run.py --help` for more info.

## CTR: Click-through-rate prediction

To run on the example data:
```bash
python ctr/run.py --embedding_dim 4 --dataset_dir example_data/ctr/criteo-subset/ --no_cuda
```

See `python ctr/run.py --help` for more info.

# Options


## Distributed training

### Launching with tracker scrips
Distributed training can be launched with the [tracker scripts](https://github.com/anon256/AdaPM/tree/main/tracker) of AdaPM. For that, use the `--tracker` option. For example:
```bash

python ../AdaPM/tracker/dmlc_ssh.py -s 2 -H [HOSTFILE] python gcn/run.py --dataset ogbn-arxiv --data_root example_data/gnn/ --no_cuda --tracker
```

The hostfile should contain the host name of one node per line.

### Launching manually
Distributed training can also be launched manually by starting the appropriate processes on the corresponding nodes manually. To do so, pass the IP of the node that hosts the scheduler process, an open port, and the world size to the processes as seen below. We recommend launching with tracker scripts.

```bash
# start the scheduler process (run this once):
python gcn/run.py --nodes 0 --root_uri "[SCHEDULER_IP]" --root_port "9091" --world_size 2 --scheduler

# start one nodes process (run this on `world_size` nodes):
python gcn/run.py --nodes 1 --root_uri "[SCHEDULER_IP]" --root_port "9091" --world_size 2
```


## CUDA options
The script automatically makes use of available CUDA devices. The this can be disabled by using `--no_cuda`.
By default workers are assigned round robin to CUDA devices. Use `--device_ids` to provide an alternative assignment (one device ID for each worker thread).
```bash
python gcn/run.py --nodes 2 --workers_per_node 1 --device_ids 2 3
```
