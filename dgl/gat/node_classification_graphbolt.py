import argparse
import os
import time

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl import graphbolt as gb
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset


class GAT(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, num_heads, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                (in_feats, in_feats),
                n_hidden,
                num_heads=num_heads,
                activation=activation,
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    (n_hidden * num_heads, n_hidden * num_heads),
                    n_hidden,
                    num_heads=num_heads,
                    activation=activation,
                )
            )
        self.layers.append(
            dglnn.GATConv(
                (n_hidden * num_heads, n_hidden * num_heads),
                n_classes,
                num_heads=num_heads,
                activation=None,
            )
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)

    def inference(self, g, device, batch_size, num_heads, gb_feat):
        feat = None
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = torch.zeros(
                    g.num_nodes(),
                    self.n_hidden * num_heads
                    if l != len(self.layers) - 1 else self.n_classes,
                    dtype=feat.dtype,
                    device=buffer_device,
                    pin_memory=pin_memory,
                )
            else:
                y = torch.zeros(
                    g.num_nodes(),
                    self.n_hidden
                    if l != len(self.layers) - 1 else self.n_classes,
                    dtype=feat.dtype,
                    device=buffer_device,
                    pin_memory=pin_memory,
                )
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                if feat is not None:              
                    h = feat[input_nodes]
                else:
                    h = gb_feat.read(input_nodes.cpu())
                    h = h.to(device)
                h_dst = h[: blocks[0].num_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(blocks[0], (h, h_dst)).flatten(1)
                else:
                    h = layer(blocks[0], (h, h_dst))
                    h = h.mean(1)
                    h = h.log_softmax(dim=-1)

                y[output_nodes] = h.to(buffer_device)
            feat = y
            feat = feat.to(device)
        return y


def evaluate(model, graph, dataloader, num_classes, gb_feat):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = gb_feat.read(input_nodes.cpu())
            x = x.to(dataloader.device)
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size, gb_feat):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size, 4, gb_feat
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes, gb_feat):
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
        backward_time = 0
        update_time = 0

        epoch_start = tmp_start = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = gb_feat.read(input_nodes.cpu())
            x = x.to(device)
            y = blocks[-1].dstdata["label"]
            sample_time += time.time() - tmp_start

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

        acc = evaluate(model, g, val_dataloader, num_classes, gb_feat)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Seconds {:.3f} ".format(
                epoch, total_loss / (it + 1), acc.item(), epoch_end - epoch_start
            )
        )
        print(
            "Epoch {:05d} | sample: {:.3f} | forward: {:.3f} | backward: {:.3f} | update: {:.3f}".format(
                epoch, sample_time, forward_time, backward_time, update_time
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
    g = g.remove_self_loop().add_self_loop()
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = GAT(in_size, 256, out_size, 3, 4, F.relu).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # prepare for graphbolt
    feat = g.ndata.pop("feat")
    feat = feat.numpy()
    path = os.path.join(os.getcwd() + "/ogbn-products.npy")
    np.save(path, feat)
    del feat
    gb_feat = gb.DiskBasedFeature(path=path)

    # model training
    print("Training...")
    train(args, device, g, dataset, model, num_classes, gb_feat)

    # test the model
    print("Testing...")
    acc = layerwise_infer(
        device, g, dataset.test_idx, model, num_classes, batch_size=4096, gb_feat=gb_feat
    )
    print("Test Accuracy {:.4f}".format(acc.item()))
