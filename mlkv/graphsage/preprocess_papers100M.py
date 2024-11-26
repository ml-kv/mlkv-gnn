import torch
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    ogbn_dataset = 'ogbn-papers100M'
    dataset = AsNodePredDataset(DglNodePropPredDataset(ogbn_dataset))
    g = dataset[0]

    import os
    import shutil
    root_path = './'
    save_path = os.path.join(root_path, ogbn_dataset)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    feat = g.ndata.pop('feat')
    torch.save(feat, os.path.join(save_path, 'feat'))
    torch.save(dataset, os.path.join(save_path, 'dataset'))
