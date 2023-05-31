import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset
from cli import parse_arguments
import os

args = parse_arguments()
print(args)

data = DglNodePropPredDataset(name=args.dataset, root=args.data_root)
graph, labels = data[0]
graph = dgl.add_reverse_edges(graph)
labels = labels[:, 0]
graph.ndata['labels'] = labels

splitted_idx = data.get_idx_split()
train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
train_mask[train_nid] = True
val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
val_mask[val_nid] = True
test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
test_mask[test_nid] = True
graph.ndata['train_mask'] = train_mask
graph.ndata['val_mask'] = val_mask
graph.ndata['test_mask'] = test_mask

parts = args.nodes
path = os.path.join(args.data_root , args.dataset.replace('-','_'), f"{parts}part_data")

dgl.distributed.partition_graph(graph, graph_name=args.dataset, num_parts=parts, out_path=path, balance_ntypes=graph.ndata['train_mask'], balance_edges=True)
