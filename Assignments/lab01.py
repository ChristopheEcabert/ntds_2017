import configparser
import os

import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse, stats, spatial
import scipy.sparse.linalg
from sklearn import preprocessing, decomposition
import librosa
import IPython.display as ipd

plt.rcParams['figure.figsize'] = (17, 5)


tracks = pd.read_csv('../data/fma_tracks.csv', index_col=0)
print('You got {} track IDs.'.format(len(tracks)))

def func(x):
    print(x.name)
    x.genre = x.name

tracks.head(10).apply(func, axis=1)

print(tracks.head(10))
a = 0
"""
# Read the confidential API key.
credentials = configparser.ConfigParser()
credentials.read(os.path.join('.', 'credentials.ini'))
api_key = credentials.get('freemusicarchive', 'api_key')


def get_genre(track_id):
    Returns the genre of a track by querying the API.
    :param track_id:    ID of the track to use as query
    :return:            Genre ID for this track (The first one)
    BASE_URL = 'https://freemusicarchive.org/api/get/'
    url = BASE_URL +  '%s.%s?api_key=%s&track_id=%d' % ('tracks', 'json', api_key, track_id)
    response = requests.get(url=url).json()
    return int(response['dataset'][0]['track_genres'][0]['genre_id'])

# A correct implementation should pass the below test.
assert get_genre(1219) == 38

tracks = pd.read_csv('../data/fma_tracks.csv', index_col=0)
print('You got {} track IDs.'.format(len(tracks)))


def get_genre_functor(s):
    
    Functor to iterate over a given pandas.Series and search for corresponding track's genre
    :param s:   Series to iterate over ( provide by apply method)
    
     for ids in s.index.values[:10]:
         s.at[ids] = get_genre(ids)

tracks.apply(get_genre_functor)
print(tracks.head(10))

def get_path(track_id):
    return os.path.join('..', 'data', '{:06d}.mp3'.format(track_id))

# 1. Get the path to the first file.
filepath = get_path(tracks.index[0])
print('File: {}'.format(filepath))

# 2. Decode the mp3 and load the audio in memory.
audio, sampling_rate = librosa.load(filepath, sr=None, mono=True)
print('Duration: {:.2f}s, {} samples'.format(audio.shape[-1] / sampling_rate, audio.size))

# 3. Load the audio in the browser and play it.
start, end = 7, 17
ipd.Audio(data=audio[start*sampling_rate:end*sampling_rate], rate=sampling_rate)

N_MFCC = 20

def compute_mfcc(track_id):
    # Get sample's path + Load
    filepath = get_path(track_id=track_id)
    data, sr = librosa.load(filepath, sr=None, mono=True)
    # Compute features
    mfcc = librosa.feature.mfcc(data, sr=sr, n_mfcc=N_MFCC)
    return mfcc


mfcc = compute_mfcc(tracks.index[0])
assert mfcc.ndim == 2
assert mfcc.shape[0] == N_MFCC

features = pd.read_csv('../data/fma_features.csv', index_col=0, header=[0, 1, 2])
assert (tracks.index == features.index).all()
features.tail(4)


for tid in tqdm(tracks.index[:10]):
    mfcc = compute_mfcc(tid)
    features.at[tid, ('mfcc', 'mean')] = np.mean(mfcc, axis=1)
    features.at[tid, ('mfcc', 'std')] = np.std(mfcc, axis=1)
    features.at[tid, ('mfcc', 'skew')] =  scipy.stats.skew(mfcc, axis=1)
    features.at[tid, ('mfcc', 'kurtosis')] =  scipy.stats.kurtosis(mfcc, axis=1)
    features.at[tid, ('mfcc', 'median')] =  np.median(mfcc, axis=1)
    features.at[tid, ('mfcc', 'min')] =  np.min(mfcc, axis=1)
    features.at[tid, ('mfcc', 'max')] =  np.max(mfcc, axis=1)

features -= features.mean(axis=0)
features /= features.std(axis=0)

features.to_csv('features.csv')
features = pd.read_csv('features.csv', index_col=0, header=[0, 1, 2])


distances = scipy.spatial.distance.pdist(features, metric='cosine')
distances = scipy.spatial.distance.squareform(distances)

plt.figure(1)
plt.hist(distances.reshape(-1), bins=50)
plt.show(block=False)

kernel_width = distances.mean()
weights = np.zeros((distances.shape[0], distances.shape[1]), dtype=np.float32)
for u in range(0, distances.shape[0]):
    for v in range(u + 1, distances.shape[1]):
        weights[u, v] = np.exp(-(distances[u, v]**2.0) / (kernel_width ** 2.0))
        weights[v, u] = weights[u, v]
# Your code here.
print(weights[0:5, 0:5])

np.save('weights.npy', weights)
"""

weights = np.load('weights.npy')
"""
fix, axes = plt.subplots(2, 2, figsize=(17, 8))
def plot(weights, axes):
    axes[0].spy(weights)
    axes[1].hist(weights[weights > 0].reshape(-1), bins=50)
plot(weights, axes[:, 0])
"""

NEIGHBORS = 100



for k in range(0, weights.shape[0] - 1):
    idx = np.argsort(weights[k,:], axis=None)
    idx = idx[(len(idx) - NEIGHBORS):]
    sz = len(idx)
    rdata = weights[k, idx]
    weights[k, :] = 0.0
    weights[k, idx] = rdata
    weights[:,k] = weights[k, :]

    nnz = np.count_nonzero(weights[k, :])


a = 0




#for k in range(0, weights.shape[0] - 1):
sweights = np.zeros(weights.shape, weights.dtype)
not_valid = dict()
for k in range(weights.shape[0]):
    not_valid[k] = []
k = 0
while k < weights.shape[0] - 1:
    nnz_left = np.count_nonzero(sweights[k, 0:k])
    if nnz_left > NEIGHBORS:
        # Already to many strong edge on the left side ... try to only select the strongest one
        n_edge = nnz_left - NEIGHBORS
        to_rmv = np.argsort(sweights[k, 0:k])
        to_rmv = to_rmv[(len(to_rmv) - nnz_left):(len(to_rmv)-NEIGHBORS)]
        for rm in to_rmv:
            sweights[k, rm] = 0.0
            sweights[rm, k] = 0.0
            if rm in not_valid:
                not_valid[rm].append(k)
            else:
                not_valid[rm] = [k]
        k = to_rmv[0]

    npick = NEIGHBORS - np.count_nonzero(sweights[k, 0:k]) if k > 0 else NEIGHBORS
    #assert np.count_nonzero(sweights[k, 0:k]) <= NEIGHBORS
    assert npick >= 0
    idx = np.argsort(weights[k, k + 1:], axis=None)
    lidx = list(k + idx + 1)
    for rm in not_valid[k]:
        if rm in lidx:
            lidx.remove(rm)

    idx = np.asarray(lidx)
    sel = idx[(len(idx) - npick):]
    # Set new entries
    sweights[k, sel] = weights[k, sel]
    sweights[sel, k] = weights[sel, k]
    nnz = np.count_nonzero(sweights[k, :])
    k += 1
    #assert nnz == NEIGHBORS
    if nnz != NEIGHBORS:
        print('TOO Many neighbors selected for row %d' % (k-1))



#plot(weights, axes[:, 1])

a = 0