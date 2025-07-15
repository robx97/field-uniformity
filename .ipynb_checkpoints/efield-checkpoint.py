import numpy as np
import math
import h5flow
from h5flow.data import dereference
import argparse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
#import statistics
import matplotlib.pyplot as plt
#import glob
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_segs(segs, mask = [], **kwargs):
    def to_list(axis):
        if mask == []:
            return np.column_stack([
                segs[f'{axis}_start'],
                segs[f'{axis}_end'],
                np.full(len(segs), None)
            ]).flatten().tolist()
        else:
            return np.column_stack([
                segs[f'{axis}_start'],
                segs[f'{axis}_end'],
                np.full(len(segs), None)
            ]).flatten()[mask].tolist()
        
    x, y, z = (to_list(axis) for axis in 'xyz')
    #had to swap x and z because of a larndsim bug, should be fixed eventually
    #not an issue in MiniRun4
    trace = go.Scatter3d(x=x, y=y, z=z, **kwargs)
    return trace

def draw_cathode_planes(x_boundaries, y_boundaries, z_boundaries, **kwargs):
    traces = []
    for i_z in range(int(len(z_boundaries)/2)):
        for i_x in range(int(len(x_boundaries)/2)):
            z, y = np.meshgrid(np.linspace(z_boundaries[i_z * 2], z_boundaries[i_z * 2 + 1], 2), np.linspace(y_boundaries.min(), y_boundaries.max(),2))
            x = (x_boundaries[i_x * 2] + x_boundaries[i_x * 2 + 1]) * 0.5 * np.ones(z.shape)
            traces.append(go.Surface(x=x, y=y, z=z, **kwargs))

    return traces

def draw_anode_planes(x_boundaries, y_boundaries, z_boundaries, **kwargs):

    traces = []
    for i_z in range(int(len(z_boundaries)/2)):
        for i_x in range(int(len(x_boundaries))):           
            z, y = np.meshgrid(np.linspace(z_boundaries[i_z * 2], z_boundaries[i_z * 2 + 1], 2), np.linspace(y_boundaries.min(), y_boundaries.max(),2))
            x = x_boundaries[i_x] * np.ones(z.shape)

            traces.append(go.Surface(x=x, y=y, z=z, **kwargs))
    
    return traces

#Functions
def cluster(i_evt, PromptHits_ev):
    #hits = np.zeros((PromptHits_ev.shape[0],9))
    hits = np.zeros((PromptHits_ev.shape[0],9))
    #print('Hit Max z PromptHitsev: '+str(np.max(PromptHits_ev['z'].data[0])))

    #code for DBSCAN to ignore NaN values
    for i in range(PromptHits_ev.shape[0]):
        if math.isnan(PromptHits_ev['x'].data[i]) == True:
        #if math.isnan(PromptHits_ev[i][1]) == True:
            return hits
        elif math.isnan(PromptHits_ev['y'].data[i]) == True:
        #if math.isnan(PromptHits_ev[i][2]) == True:
            return hits
        elif math.isnan(PromptHits_ev['z'].data[i]) == True:
        #if math.isnan(PromptHits_ev[i][3]) == True:
            return hits
        hits[i][0] = PromptHits_ev['x'].data[i]
        hits[i][1] = PromptHits_ev['y'].data[i]
        hits[i][2] = PromptHits_ev['z'].data[i]

    hit_cluster = DBSCAN(eps = 5, min_samples = 1).fit(hits)
    label = hit_cluster.labels_
    #print("Hit cluster!")
    #print(hit_cluster.components_[compmask])
    for i in range(PromptHits_ev.shape[0]):
        hits[i][3] = hit_cluster.labels_[i]
    return hits

def PCAs(hits_of_track):
    scaler = StandardScaler()
    output = {}
    X_train = hits_of_track
    #print("Before transformation: "+str(np.max([h[0] for h in hits_of_track])))
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    pca = PCA(1) # 1 component

    pca.fit(X_train)
    #len_label = len(np.unique(pca.fit(X_train).labels_))
    v_dir = pca.components_[0]
    l, start, end = length_track(hits_of_track)
    true_track = track_hits(hits_of_track, start, end)
    #print(true_track['z'])   
    # include t_par for sorting later
    output['reco'] = hits_of_track
    output['true'] = true_track
    output['v_dir'] = v_dir
    #print("After fit: "+str(np.max([h[0] for h in true_track])))
    explained_var = pca.explained_variance_ratio_
    variance = pca.explained_variance_
    

    return explained_var,variance,output


def length_track(hits_of_track):
    far_hit = np.max(hits_of_track[:,2])
    close_hit = np.min(hits_of_track[:,2])

    if far_hit != close_hit:
        ind_far = np.where(hits_of_track[:,2]==np.max(hits_of_track[:,2]))[0][0]
        ind_close = np.where(hits_of_track[:,2]==np.min(hits_of_track[:,2]))[0][0]
        length = np.linalg.norm(hits_of_track[ind_far][:3]-hits_of_track[ind_close][:3])

    elif far_hit == close_hit:
        far_hit_y = np.max(hits_of_track[:,1])
        close_hit_y = np.min(hits_of_track[:,1])
        ind_far = np.where(hits_of_track[:,1]==np.max(hits_of_track[:,1]))[0][0]
        ind_close = np.where(hits_of_track[:,1]==np.min(hits_of_track[:,1]))[0][0]
        length = np.linalg.norm(hits_of_track[ind_far][:3]-hits_of_track[ind_close][:3])

        if far_hit_y == close_hit_y:
            far_hit_x = np.max(hits_of_track[:,0])
            close_hit_x = np.min(hits_of_track[:,0])
            ind_far = np.where(hits_of_track[:,0]==np.max(hits_of_track[:,0]))[0][0]
            ind_close = np.where(hits_of_track[:,0]==np.min(hits_of_track[:,0]))[0][0]
            length = np.linalg.norm(hits_of_track[ind_far][:3]-hits_of_track[ind_close][:3])

    return length, hits_of_track[ind_close][:3], hits_of_track[ind_far][:3]

def track_hits(points, start_point, end_point):
    track = []
    # using https://math.stackexchange.com/questions/1521128/given-a-line-and-a-point-in-3d-how-to-find-the-closest-point-on-the-line:
    # and https://stackoverflow.com/questions/27733634/closest-point-on-a-3-dimensional-line-to-another-point

    # first, find the normalized direction vector of the line:
    start_and_end_diff = np.array(end_point) - np.array(start_point)
    norm_length_vector = start_and_end_diff / np.sqrt(np.dot(start_and_end_diff, start_and_end_diff))
    # then, find the normalized vector between the point you're using and the start point of the line:
    for p in points:
        vector_from_start_to_used_point_normalized = (p - start_point) /np.sqrt(np.dot(start_and_end_diff, start_and_end_diff))
    
    # take dot product between these two vectors 
        t = np.dot(vector_from_start_to_used_point_normalized, norm_length_vector)

        point_on_line_closest_to_used_point = np.array(start_point) + (np.array(end_point) -np.array(start_point)) * t
        track.append(point_on_line_closest_to_used_point)
    return track
