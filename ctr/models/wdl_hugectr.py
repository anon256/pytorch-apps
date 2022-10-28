import torch
from numpy.lib.function_base import _parse_input_dimensions
from torch import nn
from .wide import Wide

import sys
from common.ps_models import PSEmbedding


class WdlHugeCtr(nn.Module):
    """
    Adapted from Wide & Deep (original Paper adapted for Criteo)
    and HugeCTR's implementation as detailed in https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/samples/wdl/wdl.py
    """

    def __init__(self, feature_dim, embed_dim, output_dim=256, for_adapm=False):  # wide_in_dim, wide_out_dim, hardcoded for now
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        # wide component; corresponding to layer SparseEmbedding2
        self.wide = Wide(input_dim=feature_dim, pred_dim=1, for_adapm=for_adapm)

        if embed_dim != 16:
            print(f" Warning: embed_dim={embed_dim} HugeCtr's example uses embed_dim=16. \n"
                  f"output_dim ignored")

        # deep components
        if for_adapm:
            self.sparse_embedding1 = PSEmbedding(num_embeddings=feature_dim, embedding_dim=embed_dim)
        else:
            self.sparse_embedding1 = nn.Embedding(num_embeddings=feature_dim, embedding_dim=embed_dim)

        self.reshape1_res = None
        self.fc1 = nn.Linear(13 + (26 * embed_dim), 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def get_regularization(self, l2_reg_wide, l2_reg_embedding):
        regularization_loss = 0

        # embedding part
        regularization_loss += l2_reg_embedding * (self.reshape1_res**2).sum()

        # wide part
        regularization_loss += l2_reg_wide * (self.wide_embeddings**2).sum()

        return regularization_loss

    def forward(self, dense_features, sparse_features):
        # wide component
        wide_result, self.wide_embeddings = self.wide(sparse_features)

        # deep component
        if self.sparse_embedding1.__class__ == PSEmbedding:
          self.reshape1_res = self.flatten(self.sparse_embedding1(sparse_features, self.fc1.bias.device))
        else:
          self.reshape1_res = self.flatten(self.sparse_embedding1(sparse_features))

        concat1_res = torch.cat((self.reshape1_res, dense_features), 1)
        fc1res = self.relu(self.fc1(concat1_res))
        dropout1_res = self.dropout(fc1res)
        dropout2_res = self.dropout(self.relu(self.fc2(dropout1_res)))
        fc3_res = self.fc3(dropout2_res)

        # combine outputs
        add1 = wide_result.add(fc3_res)

        return add1
