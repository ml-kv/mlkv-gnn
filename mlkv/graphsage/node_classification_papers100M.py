import argparse
import time

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.storages.mlkv import MLKV

lookahead_caching = True

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))

        self.bns.append(nn.BatchNorm1d(hid_size))
        self.bns.append(nn.BatchNorm1d(hid_size))

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.bns[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

def train(args, device, g, dataset, model, mlkv):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [5, 5, 5],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_labels=['label']
    )
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(20):
        sample_time = 0
        pull_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0

        epoch_start = tmp_start = time.time()
        model.train()
        total_loss = 0

        train_dataloader_iterator = iter(train_dataloader)
        if epoch == 0 and lookahead_caching:
            (input_nodes_lookahead, _, blocks_lookahead) = next(train_dataloader_iterator)
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            sample_time += time.time() - tmp_start
            start1 = time.time()
            if epoch == 0 and lookahead_caching:
                if it != len(train_dataloader) - 1:
                    (input_nodes, _, blocks) = (input_nodes_lookahead, _, blocks_lookahead)
                    (input_nodes_lookahead, _, blocks_lookahead) = next(train_dataloader_iterator)
                x = mlkv.pull_and_lookahead_data_from_mlkv(4, input_nodes.cpu(), input_nodes_lookahead.cpu())
            else:
                (input_nodes, output_nodes, blocks) = next(train_dataloader_iterator)
                x = mlkv.pull_data_from_mlkv(input_nodes)

            x = x.to(device)
            y = blocks[-1].dstdata['label']
            pull_time += time.time() - tmp_start

            tmp_start = time.time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            forward_time += time.time() - tmp_start

            tmp_start = time.time()
            loss.backward()
            backward_time += time.time() - tmp_start

            tmp_start = time.time()
            opt.step()
            update_time += time.time() - tmp_start

            total_loss += loss.item()
            tmp_start = time.time()
        epoch_end = time.time()

        print(
            "Epoch {:05d} | Loss {:.4f} | Seconds {:.3f} |".format(
                epoch, total_loss / (it+1), epoch_end - epoch_start
            )
        )
        print(
            'Epoch {:05d} sample: {:.3f}, pull: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                epoch, sample_time, pull_time, forward_time, backward_time, update_time
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default='mixed',
        choices=['cpu', 'mixed', 'puregpu'],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')

    # load and preprocess dataset
    print('Loading data')

    import os
    import shutil
    root_path = './'
    ogbn_dataset = 'ogbn-papers100M'
    save_path = os.path.join(root_path, ogbn_dataset)
    mlkv_dir = os.path.join(root_path, 'testdb')
    if os.path.exists(mlkv_dir):
        shutil.rmtree(mlkv_dir)
    os.makedirs(mlkv_dir)

    with open(os.path.join(save_path, 'feat'), 'rb') as f:
        feat = torch.load(f)
        in_size = feat.shape[1]
        mlkv = MLKV(table_size_bytes = 128 * pow(2, 20), log_size_bytes = 64 * pow(2, 30),
            path = mlkv_dir, log_mutable_fraction = 0.9, feat_dim = in_size, feat_dtype = feat.dtype)
        mlkv.open_mlkv()
        feat_ids = torch.arange(start=0, end=feat.shape[0], dtype=torch.int64, device='cpu')
        mlkv.parallel_push_data_to_mlkv(num_thread = 16, id_tensor = feat_ids, data_tensor = feat)
        del feat

    with open(os.path.join(save_path, 'dataset'), 'rb') as f:
        dataset = torch.load(f)

    g = dataset[0]
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    g.ndata['label'] = g.ndata['label'].long()
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    import gc
    gc.collect()

    # model training
    print('Training...')
    train(args, device, g, dataset, model, mlkv)
