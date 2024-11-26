## Operating system and software dependency requirements

Ubuntu (20.04 LTS x86/64) packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip make cmake
```

Python packages

```bash
conda create -n MLKV-GNN python=3.11
conda activate MLKV-GNN
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ogb pydantic torchmetrics
```

## DGL

Build

```
git clone -b dgl-v2.3.0 git@github.com:ml-kv/mlkv-gnn.git dgl-v2.3.0
cd dgl-v2.3.0
git submodule update --init --recursive
mkdir build && cd build 
cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j16
cd ../python
python3 setup.py install --user
pip3 uninstall dgl
```

Benchmark

```
git clone -b main git@github.com:ml-kv/mlkv-gnn.git mlkv-gnn
cd mlkv-gnn/dgl/graphsage/
python3 node_classification.py --mode [cpu|mixed]
python3 node_classification_papers100M.py --mode [cpu|mixed]
```

## DGL with MLKV

Build

```
sudo apt-get install uuid-dev libaio-dev libtbb-dev
git clone -b dgl-v2.3.0-mlkv git@github.com:ml-kv/mlkv-gnn.git dgl-v2.3.0-mlkv
cd dgl-v2.3.0-mlkv
git submodule update --init --recursive
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j16
cd ../python
python3 setup.py install --user
pip3 uninstall dgl
```

Benchmark

```
git clone -b main git@github.com:ml-kv/mlkv-gnn.git mlkv-gnn
cd mlkv-gnn/mlkv/graphsage
python3 node_classification.py --mode [cpu|mixed]

python3 preprocess_papers100M.py
DGL_PREFETCHER_TIMEOUT=600 python3 node_classification_papers100M.py --mode [cpu|mixed]
```

## DGL with FASTER

Build

```bash
sudo apt-get install uuid-dev libaio-dev libtbb-dev
git clone -b dgl-v2.3.0-faster git@github.com:ml-kv/mlkv-gnn.git dgl-v2.3.0-faster
cd dgl-v2.3.0-faster
git submodule update --init --recursive
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j16
cd ../python
python3 setup.py install --user
pip3 uninstall dgl
```

Benchmark

```
git clone -b main git@github.com:ml-kv/mlkv-gnn.git mlkv-gnn
cd mlkv-gnn/faster/graphsage
python3 node_classification.py --mode [cpu|mixed]

python3 preprocess_papers100M.py
DGL_PREFETCHER_TIMEOUT=600 python3 node_classification_papers100M.py --mode [cpu|mixed]
```
