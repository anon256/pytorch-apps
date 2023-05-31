import torch
from torch import nn

from ctr.models import wdl_hugectr
"""
Setup of model, DataLoaders and everything else needed for computation.
Train and test functions.
"""

# final data manipulation to avoid needing to modify collate_fn
def prepare_and_move_data(device, samples):
    dense_features = samples["dense_features"].to(device)
    sparse_features = samples["sparse_features"].to(device)
    squeezed_labels = samples['labels']
    labels = torch.unsqueeze(squeezed_labels, dim=1).to(device)

    return dense_features, sparse_features, labels


def setup_model(feature_dim, embed_dim, output_dim, device, model_type, for_adapm):
    if model_type == "wdl_hugectr":
        model = wdl_hugectr.WdlHugeCtr(feature_dim, embed_dim, output_dim=output_dim, for_adapm=for_adapm).to(device)
    else:
        raise ValueError(f" supplied model type '{model_type}' not found")

    return model

