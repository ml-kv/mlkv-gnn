import argparse
import os
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

lookahead_caching = False
recover = False
checkpoint = True
# Copy your checkpoint token here
# e.g. "b86302dc-0d99-47a2-a6f9-97c7ca719ca7"
checkpoint_token = None
mlkv_dir = os.path.join(os.getcwd() + '/testdb')


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, mlkv):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = None
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=mlkv.feat_dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            if feat is not None:
                feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                if l == 0:
                    x = mlkv.pull_data_from_mlkv(input_nodes.cpu())
                    x = x.to(device)
                else:
                    x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes, mlkv):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = mlkv.pull_data_from_mlkv(input_nodes.cpu())
            x = x.to(device)
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size, mlkv):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size, mlkv
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes, mlkv):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [5, 5, 5],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        sample_time = 0
        forward_time = 0
        pull_time = 0
        backward_time = 0
        update_time = 0

        epoch_start = tmp_start = time.time()
        model.train()
        total_loss = 0

        train_dataloader_iterator = iter(train_dataloader)
        if epoch == 0 and lookahead_caching:
            (input_nodes_lookahead, _, blocks_lookahead) = next(train_dataloader_iterator)
        for it in range(0, len(train_dataloader)):
            if epoch == 0 and lookahead_caching:
                if it != len(train_dataloader) - 1:
                    (input_nodes, _, blocks) = (input_nodes_lookahead, _, blocks_lookahead)
                    (input_nodes_lookahead, _, blocks_lookahead) = next(train_dataloader_iterator)
                sample_time += time.time() - tmp_start
                tmp_start = time.time()
                x = mlkv.pull_and_lookahead_data_from_mlkv(4, input_nodes.cpu(), input_nodes_lookahead.cpu())
            else:
                (input_nodes, output_nodes, blocks) = next(train_dataloader_iterator)
                sample_time += time.time() - tmp_start
                tmp_start = time.time()
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

        acc = evaluate(model, g, val_dataloader, num_classes, mlkv)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Seconds {:.3f} ".format(
                epoch, total_loss / (it + 1), acc.item(), epoch_end - epoch_start
            )
        )
        print(
            "Epoch {:05d} | sample: {:.3f} | pull: {:.3f} | forward: {:.3f} | backward: {:.3f} | update: {:.3f}".format(
                epoch, sample_time, pull_time, forward_time, backward_time, update_time
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
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
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # open mlkv
    mlkv = MLKV(table_size_bytes = 128 * pow(2, 20), log_size_bytes = 16 * pow(2, 30),
            path = mlkv_dir, log_mutable_fraction = 0.9, feat_dim = in_size, feat_dtype = g.ndata['feat'].dtype)

    if recover == True:
        mlkv.recover_mlkv(checkpoint_token = checkpoint_token)
    else:
        import os
        import shutil
        if os.path.exists(mlkv_dir):
            shutil.rmtree(mlkv_dir)
        os.makedirs(mlkv_dir)

        mlkv.open_mlkv()
        feat_ids = torch.arange(start=0, end=g.ndata['feat'].shape[0], dtype=torch.int64, device='cpu')
        mlkv.parallel_push_data_to_mlkv(num_thread = 4, id_tensor = feat_ids, data_tensor = g.ndata['feat'])

    g.ndata.pop('feat')

    # model training
    print("Training...")
    train(args, device, g, dataset, model, num_classes, mlkv)

    # test the model
    print("Testing...")
    acc = layerwise_infer(
        device, g, dataset.test_idx, model, num_classes, batch_size=4096, mlkv=mlkv
    )
    print("Test Accuracy {:.4f}".format(acc.item()))

    if checkpoint == True:
        if recover == False or lookahead_caching == True:
            mlkv.checkpoint_mlkv()
