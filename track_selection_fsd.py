from efield import *
import numpy as np
import math
import h5flow
from h5flow.data import dereference
import argparse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import statistics
import matplotlib.pyplot as plt
import os
import sys
import re
from tqdm import tqdm

# Selection parameters
pixel_pitch = 0.372
min_size = 90
max_rel_spread = 0.07
max_axis_dist = 3*pixel_pitch
#max_break_dist = 15*pixel_pitch
max_break_dist = 25*pixel_pitch
d = 2 #Distance away from a TPC face
v = 0.973 #Explained variance minimum
b = 55 #cm 

# Initialize DBSCAN
dbscan = DBSCAN(eps=max_break_dist, metric='euclidean', min_samples=1)

# Initialize PCA
pca = PCA(3)

parser = argparse.ArgumentParser(description='Input and counter determiner.')
parser.add_argument('input_file', type=str, help='input filename')
parser.add_argument('output_file', type=str, help='output filename')
parser.add_argument('is2x2', type=str2bool, help='2x2 (true) or FSD (false)')
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file
is2x2 = args.is2x2
#flist.append(input_file)
muon_track_number = 0

# Extract all digits
digits = re.findall(r'\d', input_file)

# Take the last N digits - 14 gives you full file timestamp for FSD
last_digits = digits[-14:] 
a2a = []
a_array = []
l_array = []
true_hits = []
reco_hits = []
PCA_dir = []
pca_var = []
pca_qual = []
charge = []
pdg = []
labels = []
drift_time = []
io_group = []

f_manager = h5flow.data.H5FlowDataManager(input_file, 'r', mpi=False)
events = f_manager["charge/events/data"]
length_of_file = len(f_manager["charge/events/data"])
#module_bounds = f_manager['geometry_info'].attrs['module_RO_bounds']
lar_detector_bounds = f_manager['geometry_info'].attrs['lar_detector_bounds']
print("Bounds!")
print(is2x2)
if(is2x2):
    x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
    y_boundaries = np.array([-42 - 19.8543 - 0.027712, -42 + 103.8543 + 0.027712]) #0.027 term is to account for module 2
    z_boundaries = np.array([-64.3163,  -2.6837, 2.6837, 64.3163 + 0.027712]) #0.027 term is to account for module 2
else:
    x_boundaries = lar_detector_bounds[:,0]
    y_boundaries = lar_detector_bounds[:,1]
    z_boundaries = lar_detector_bounds[:,2]
min_boundaries = np.array([x_boundaries[0], y_boundaries[0], z_boundaries[0]])
max_boundaries = np.array([x_boundaries[-1], y_boundaries[-1], z_boundaries[-1]])
print(y_boundaries)
print(min_boundaries)
print(max_boundaries)
ev = f_manager["charge/events/data"].shape[0]
#ev = 5000
len_label = 0
print("Looping through events")
for i_evt in range(ev):
    track_number = 0
    step = i_evt/10
    progress_str = f"Event {i_evt} out of {ev}."
    sys.stdout.write('\r' + progress_str)
    sys.stdout.flush()
    
    #load dataset, checking for emptiness or bad references
    try:
        pev = f_manager["charge/events", "charge/calib_prompt_hits", i_evt]
        if len(pev) == 0:
            print(f"Skipping event {i_evt} (no hits)")
            continue
    except IndexError:
        print(f"Skipping event {i_evt} (invalid reference)")
        continue
    trigs = f_manager["charge/events", "charge/ext_trigs", i_evt]
    trigio = trigs['iogroup'].data
    PromptHits_ev = pev[0]
    hits = np.column_stack([
        PromptHits_ev['x'].data,
        PromptHits_ev['y'].data,
        PromptHits_ev['z'].data
    ])
    io = PromptHits_ev['io_group'].data
    qin = PromptHits_ev['Q'].data
    t_drift = PromptHits_ev['t_drift'].data
    valid_mask = ~np.isnan(hits).any(axis=1)
    hits = hits[valid_mask]
    qin = qin[valid_mask]
    io = io[valid_mask]
    t_drift = t_drift[valid_mask]

    #some conditionals to account for different detectors
    if is2x2:
        if not np.any(np.isin(trigio,6)):
            continue # for 2x2 data, skip event if there isn't a light trigger

    # Cluster hits in the image with DBSCAN
    clust_labels = dbscan.fit(hits).labels_
    len_label += len(clust_labels)

    # Loop over the clusters and apply some quality cuts
    for c in np.unique(clust_labels):
        # Restrict points to the relevant cluster
        index = np.where(clust_labels == c)[0]
        points_c = hits[index]
        q = qin[index]
        t = t_drift[index]
        iog = io[index]
        if len(index) < min_size:
            continue
    
        # If the relative spread of the points w.r.t. the main axis is too large, skip
        decomp = pca.fit(points_c)
        axis = decomp.components_[0]
        var = decomp.explained_variance_
        rel_spread = np.sqrt((var[1] + var[2])/var[0])
        if rel_spread > max_rel_spread:
            continue
    
        # Project all points on the principal axis, filter out those too far from the main axis
        cent = np.mean(points_c, axis=0)
        dists = np.linalg.norm(np.cross(points_c - cent, axis), axis=1)
        points_c = points_c[dists < max_axis_dist]
        q = q[dists < max_axis_dist]
        t = t[dists < max_axis_dist]
        iog = iog[dists < max_axis_dist]
        if len(points_c) < min_size:
            continue

        # PCA fit of line
        a, p, output = PCAs(points_c)
        l, start, end = length_track(points_c)

        #if satisfies all criteria now, we output
        if np.logical_and(a > v, l > b):
            a_array.append(a)
            l, start, end = length_track(points_c)
            l_array.append(l)
            n_hits = len(output['true'])
            #Check if it is a2a (crossing in x)
            if min_boundaries[0]-d < np.min(points_c[:,0]) < (min_boundaries[0] + d) and (max_boundaries[0] -d) < np.max(points_c[:,0]) < max_boundaries[0]+d:
                anode = np.tile([True, True, True], (n_hits,1)) # size 3 only for output purposes
            else:
                anode = np.tile([False, False, False], (n_hits,1))
            PCA_dir_tiled = np.tile(output['v_dir'], (n_hits, 1))
            pca_q = np.tile(np.hstack([a, a, a]), (n_hits,1))
            if(ev == 0 ):
                print(len(q))
                print(n_hits)
            qout = np.tile(q[:, None], (1, 3))
            tout = np.tile(t[:, None], (1, 3))
            iogout = np.tile(iog[:, None], (1, 3))
            track_number += 1
            #make unique label for track
            other_digits = list(str(ev) + str(muon_track_number) + str(track_number)) 
            combined_digits = last_digits + other_digits 
            # Convert to int
            lab = int(''.join(combined_digits))
            ev_labels = np.tile([lab, lab, lab], (n_hits,1))  # cluster by event, so we keep run-wide reference to track

            if not np.any(true_hits):
                true_hits = output['true']
                reco_hits = output['reco']
                PCA_dir = PCA_dir_tiled
                labels = ev_labels
                a2a = anode
                pca_qual = pca_q
                charge = qout
                drift_time = tout
                io_group = iogout
            else:
                true_hits = np.concatenate([true_hits, output['true']])
                reco_hits = np.concatenate([reco_hits, output['reco']])
                PCA_dir = np.concatenate([PCA_dir, PCA_dir_tiled])
                labels = np.concatenate([labels, ev_labels])
                a2a = np.concatenate([a2a, anode])
                pca_qual = np.concatenate([pca_qual, pca_q])
                charge = np.concatenate([charge, qout])
                drift_time = np.concatenate([drift_time, tout])
                io_group = np.concatenate([io_group, iogout])
         
    muon_track_number += track_number
print("Total clusters "+str(len_label))
print(np.shape(true_hits))
print(np.shape(reco_hits))
print(np.shape(PCA_dir))
print(np.shape(labels))
print(np.shape(a2a))
print(np.shape(pca_qual))
print(np.shape(charge))
print(np.shape(drift_time))
print(np.shape(io_group))
outputArr = np.array([true_hits,
                      reco_hits,
                      PCA_dir,
                      labels,
                      a2a,
                      pca_qual,
                      charge,
                      drift_time,
                      io_group])

np.save(output_file,outputArr)
print('Done!')
