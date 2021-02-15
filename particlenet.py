import os
import logging
import requests
import time
import functools
import pathlib
import shutil

import awkward
import uproot_methods

import numpy as np
import pandas as pd

import torch
import torch_geometric
import tqdm.auto as tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    Downloading the Dataset
'''

def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path

os.makedirs(os.path.join(PROJECT_DIR, 'original'), exist_ok=True)
# download('https://zenodo.org/record/2603256/files/train.h5?download=1', os.path.join(PROJECT_DIR, 'original/train.h5'))
# download('https://zenodo.org/record/2603256/files/val.h5?download=1', os.path.join(PROJECT_DIR, 'original/val.h5'))
# download('https://zenodo.org/record/2603256/files/test.h5?download=1', os.path.join(PROJECT_DIR, 'original/test.h5'))

def _transform(df, start=0, stop=-1):
    """
    Takes a DataFrame and converts it into a Awkward array representation
    with features relevant to our model.

    :param df: Pandas DataFramem, The DataFrame with all the momenta-energy coordinates for all the particles
    :param start: int, First element of the DataFrame
    :param stop: int, Last element of the DataFrame
    :return v: OrderedDict, A Ordered Dictionary with all properties of interest

    Here the function is just computing 4 quantities of interest:
    * Eta value relative to the jet
    * Phi value relative to the jet
    * Transverse Momentum of the Particle (log of it) 
    * Energy of the Particle (log of it) 
    """
    from collections import OrderedDict
    v = OrderedDict()

    def _col_list(prefix, max_particles=200):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]

    df = df.iloc[start:stop]
    # We take the values in the dataframe for all particles of a single event in each row
    # px, py, pz, e are in separate arrays
    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values

    # We filter out the non-0 non-negative energy particles
    mask = _e > 0
    n_particles = np.sum(mask, axis=1) # Number of particles for each event where energy is greater than 0
    # _p[mask] filters out the >0 energy particles, and flattens them, so that they can be recollected for each event from counts array.
    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])
    # These are jagged arrays with each row for 1 event, and all particles in the row

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    jet_p4 = p4.sum() # Sum of Lorentz Vectors for all particles in the jet

    # Getting the Labels
    _label = df['is_signal_new'].values # the target labels, QCD or Top
    v['label'] = np.stack((_label, 1-_label), axis=-1) # Making it categorical [Top, QCD]
    # Transformed features relative to the Jet and log features
    v['part_pt_log'] = np.log(p4.pt)
    v['part_e_log'] = np.log(energy)
    # Flip particle ETA if Jet Eta is negative
    # Particle's phi relative to the Jet
    _jet_etasign = np.sign(jet_p4.eta)
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = (p4.eta - jet_p4.eta) * _jet_etasign
    v['part_phirel'] = p4.delta_phi(jet_p4)

    del px, py, pz, energy, _px, _py, _pz, _e, p4, jet_p4, df
    return v

def convert(source, destdir, basename, step=None, limit=None):
    """
    Converts the DataFrame into an Awkward array and performs the read-write
    operations for the same. Also performs Batching of the file into smaller
    Awkward files.

    :param source: str, The location to the H5 file with the dataframe
    :param destdir: str, The location we need to write to
    :param basename: str, Prefix for all the output file names
    :param step: int, Number of rows per awkward file, None for all rows in 1 file
    :param limit: int, Number of rows to read.
    """
    df = pd.read_hdf(source, key='table')
    logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
        logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]

    idx = 0
    # Generate files as batches based on step size, only 1 batch is default
    for start in range(0, df.shape[0], step):
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.awkd'%(basename, idx))
        logging.info(output)
        if os.path.exists(output):
            logging.warning('... file already exist: continue ...')
            continue
        v = _transform(df, start=start, stop=start+step)
        awkward.save(output, v, mode='x')
        idx += 1

    del df, output

convert('original/val.h5', destdir='converted', basename='train_file')
convert('original/test.h5', destdir='converted', basename='test_file')

def pad_array(jagged_array, max_len, value=0., dtype='float32'):
    rectangluar_array = np.full(shape=(len(jagged_array), max_len), fill_value=value, dtype=dtype)
    for idx, jagged_element in enumerate(jagged_array):
        if len(jagged_element) != 0:
            trunc = jagged_element[:max_len].astype(dtype)
            rectangluar_array[idx, :len(trunc)] = trunc
    return rectangluar_array


'''
    Preparing the Dataset
'''

class AwkwardDataset:
    def __init__(self, filepath, value_cols = None, label_cols='label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.value_cols = value_cols if value_cols is not None else {
            'points': ['part_etarel', 'part_phirel'],
            'features': ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel'],
            'mask': ['part_pt_log']
        }
        self.label_cols = label_cols
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        # Here we make the arrays which will keep out data and load the first batch
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            # Load output labels from the awkward array
            self._label = a[self.label_cols]

            for k in self.value_cols:
                cols = self.value_cols[k]
                assert isinstance(cols, (list, tuple))
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        return self._label if key == self.label_cols else self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        # Get a random permutation
        if seed is not None: np.random.seed(seed)
        shuffle_indices = np.random.permutation(self.__len__())
        # Reorder the table
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

train_awkward = AwkwardDataset('converted/train_file_0.awkd', data_format='channel_last')
val_awkward = AwkwardDataset('converted/test_file_0.awkd', data_format='channel_last')


class ParticleDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, awkward_dataset, directory=PROJECT_DIR, transform=None, pre_transform=None, pre_filter=None):
        self.awkward_dataset = awkward_dataset
        os.makedirs(directory, exist_ok=True)
        super(ParticleDataset, self).__init__(directory, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data_1.pt"]

    def process(self):
        data_list = []
        ys = np.argmax(self.awkward_dataset.y, axis=1)
        for i in range(self.awkward_dataset.X['points'].shape[0]):
            if np.where(self.awkward_dataset.X["mask"][i]  == 0)[0].shape[0] != 0:
                feat = torch.tensor(self.awkward_dataset.X["features"][i][:np.where(self.awkward_dataset.X["mask"][i]  == 0)[0][0]], dtype=torch.float)
                pos = torch.tensor(self.awkward_dataset.X["points"][i][:np.where(self.awkward_dataset.X["mask"][i]  == 0)[0][0]], dtype=torch.float)
            else:
                feat = torch.tensor(self.awkward_dataset.X["features"][i], dtype=torch.float)
                pos = torch.tensor(self.awkward_dataset.X["points"][i], dtype=torch.float)
            
            data_list.append(torch_geometric.data.Data(x=feat, pos=pos, y=ys[i]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

train_dataset = ParticleDataset(train_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'train'))
train_dataloader = torch_geometric.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = ParticleDataset(val_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'val'))
val_dataloader = torch_geometric.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

'''
    ParticleNet Implementation
'''

class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, k):
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)


class ParticleNet(torch.nn.Module):

    def __init__(self, settings):
        super().__init__()
        previous_output_shape = settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K).to(DEVICE))
            previous_output_shape = channels[-1]



        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units


        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos

        for idx, layer in enumerate(self.conv_process):
          fts = layer(pts, fts, batch.batch)
          pts = fts

        x = torch_geometric.nn.global_mean_pool(fts, batch.batch)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        x = self.output_activation(x)
        return x

settings = {
    "conv_params": [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ],
    "fc_params": [
        (0.1, 256)
    ],
    "input_features": 4,
    "output_classes": 2,
}

model = ParticleNet(settings)
model = model.to(DEVICE)

print(model)

