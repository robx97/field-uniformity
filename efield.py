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
        #hits[i][0] = PromptHits_ev[i][1]
        #hits[i][1] = PromptHits_ev[i][2]
        #hits[i][2] = PromptHits_ev[i][3]

    hit_cluster = DBSCAN(eps = 5, min_samples = 1).fit(hits)
    label = hit_cluster.labels_
    #print("Hit cluster!")
    #print(hit_cluster.components_[compmask])
    for i in range(PromptHits_ev.shape[0]):
        hits[i][3] = hit_cluster.labels_[i]
        #hits[i][4] = PromptHits_ev['id'].data[0][i]
        #hits[i][5] = PromptHits_ev['t_drift'].data[0][i]
        #hits[i][6] = PromptHits_ev['ts_pps'].data[0][i]
        #hits[i][7] = PromptHits_ev['Q'].data[0][i]
        #hits[i][8] = label[i]
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


def select_muon_track(hits, x_boundaries, y_boundaries, z_boundaries, clus_arr, muon_track_number, a_array, l_array, true_hits, reco_hits, PCA_dir, pca_qual, labels, a2a):
    tracks = np.unique(hits[:,3]) #This gets how many tracks there are
    muon_hits = []
    anode = []
    track_number = 0

    #Bottom face for each coordinate
    min_boundaries = np.array([x_boundaries[0], y_boundaries[0], z_boundaries[0]])

    #Top face for each coordinate
    max_boundaries = np.array([x_boundaries[-1], y_boundaries[-1], z_boundaries[-1]])

    #middle of x boundaries:
    #middle_x_boundaries = np.array([x_boundaries[1], x_boundaries[2]])
    for n in range(len(np.unique(tracks))):
        hits1 = hits[:,3] == n #get the hits with track number equal to n
        hits_with_track = hits[hits1] #position of the hits with their associated track number
        hits_of_track = np.delete(hits_with_track, (3,4,5,6,7,8), axis = 1) #hits without their track number
        hits_of_cluster = [] #for clustering hits in a specific way

        for i in range(len(max_boundaries)):
            ix = i
            d = 2 #Distance away from a TPC face

            v = 0.982 #Explained variance minimum

            b = 55 #minimum track length requirement cm- should also apply to FSD though
            
            #Does the track penetrate both z-boundaries?
            #if 1==1:
            if min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i] -d) < np.max(hits_of_track[:,i]) < max_boundaries[i]+d:
                a, p, output = PCAs(hits_of_track)
                a_array.append(a)
                l, start, end = length_track(hits_of_track)
                l_array.append(l)
                n_hits = len(output['true'])
                PCA_dir_tiled = np.tile(output['v_dir'], (n_hits, 1))
                if np.logical_and(a > v, l > b):
                    muon_hits.append(hits_with_track)
                    pca_q = np.tile(np.hstack([a, a, a]), (n_hits,1))
                    track_number += 1
                    ev_labels = np.tile([muon_track_number+track_number, muon_track_number+track_number, muon_track_number+track_number], (n_hits,1))  # cluster by event, so we keep file-wide reference to track
                    if i == 0: #x dimension, where anode to anode crossings happen!
                        anode = np.tile([True, True, True], (n_hits,1)) # size 3 only for output purposes
                    else:
                        anode = np.tile([False, False, False], (n_hits,1))
                    
                    if not np.any(true_hits):
                        true_hits = output['true']
                        reco_hits = output['reco']
                        PCA_dir = PCA_dir_tiled
                        labels = ev_labels
                        a2a = anode
                        pca_qual = pca_q
                    else:
                        true_hits = np.concatenate([true_hits, output['true']])
                        reco_hits = np.concatenate([reco_hits, output['reco']])
                        PCA_dir = np.concatenate([PCA_dir, PCA_dir_tiled])
                        labels = np.concatenate([labels, ev_labels])
                        a2a = np.concatenate([a2a, anode])
                        pca_qual = np.concatenate([pca_qual, pca_q])
                    break

            elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (min_boundaries[i-1] -d) < np.min(hits_of_track[:,i-1]) < min_boundaries[i-1]+d:
                a, p, output = PCAs(hits_of_track)
                a_array.append(a)
                l, start, end = length_track(hits_of_track)
                l_array.append(l)
                n_hits = len(output['true'])
                PCA_dir_tiled = np.tile(output['v_dir'], (n_hits, 1))
                if np.logical_and(a > v, l > b):
                    track_number += 1
                    pca_q = np.tile(np.hstack([a, a, a]), (n_hits,1))
                    anode = np.tile([False, False, False], (n_hits,1))
                    ev_labels = np.tile([muon_track_number+track_number, muon_track_number+track_number, muon_track_number+track_number], (n_hits,1)) 
                    muon_hits.append(hits_with_track)
                    if not np.any(true_hits):
                        true_hits = output['true']
                        reco_hits = output['reco']
                        PCA_dir = PCA_dir_tiled
                        labels = ev_labels
                        a2a = anode
                        pca_qual = pca_q
                    else:
                        true_hits = np.concatenate([true_hits, output['true']])
                        reco_hits = np.concatenate([reco_hits, output['reco']])
                        PCA_dir = np.concatenate([PCA_dir, PCA_dir_tiled])
                        labels = np.concatenate([labels, ev_labels])
                        a2a = np.concatenate([a2a, anode])
                        pca_qual = np.concatenate([pca_qual, pca_q])
                    break

            elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (min_boundaries[i-2] -d) < np.min(hits_of_track[:,i-2]) < min_boundaries[i-2]+d:
                a, p, output = PCAs(hits_of_track)
                a_array.append(a)
                l, start, end = length_track(hits_of_track)
                l_array.append(l)
                n_hits = len(output['true'])
                PCA_dir_tiled = np.tile(output['v_dir'], (n_hits, 1))
                if np.logical_and(a > v, l > b):
                    track_number += 1
                    pca_q = np.tile(np.hstack([a, a, a]), (n_hits,1))
                    anode = np.tile([False, False, False], (n_hits,1))
                    ev_labels = np.tile([muon_track_number+track_number, muon_track_number+track_number, muon_track_number+track_number], (n_hits,1))
                    muon_hits.append(hits_with_track)
                    if not np.any(true_hits):
                        true_hits = output['true']
                        reco_hits = output['reco']
                        PCA_dir = PCA_dir_tiled
                        labels = ev_labels
                        a2a = anode
                        pca_qual = pca_q
                    else:
                        true_hits = np.concatenate([true_hits, output['true']])
                        reco_hits = np.concatenate([reco_hits, output['reco']])
                        PCA_dir = np.concatenate([PCA_dir, PCA_dir_tiled])
                        labels = np.concatenate([labels, ev_labels])
                        a2a = np.concatenate([a2a, anode])
                        pca_qual = np.concatenate([pca_qual, pca_q])
                    break

            elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i-1] -d) < np.max(hits_of_track[:,i-1]) < max_boundaries[i-1]+d:
                a, p, output = PCAs(hits_of_track)
                a_array.append(a)
                l, start, end = length_track(hits_of_track)
                l_array.append(l)
                n_hits = len(output['true'])
                PCA_dir_tiled = np.tile(output['v_dir'], (n_hits, 1))
                if np.logical_and(a > v, l > b):
                    track_number += 1
                    pca_q = np.tile(np.hstack([a, a, a]), (n_hits,1))
                    anode = np.tile([False, False, False], (n_hits,1))
                    ev_labels = np.tile([muon_track_number+track_number, muon_track_number+track_number, muon_track_number+track_number], (n_hits,1))
                    muon_hits.append(hits_with_track)
                    if not np.any(true_hits):
                        true_hits = output['true']
                        reco_hits = output['reco']
                        PCA_dir = PCA_dir_tiled
                        labels = ev_labels
                        a2a = anode
                        pca_qual = pca_q
                    else:
                        true_hits = np.concatenate([true_hits, output['true']])
                        reco_hits = np.concatenate([reco_hits, output['reco']])
                        PCA_dir = np.concatenate([PCA_dir, PCA_dir_tiled])
                        labels = np.concatenate([labels, ev_labels])
                        a2a = np.concatenate([a2a, anode])
                        pca_qual = np.concatenate([pca_qual, pca_q])
                    break

            elif min_boundaries[i]-d < np.min(hits_of_track[:,i]) < (min_boundaries[i] + d) and (max_boundaries[i-2] -d) < np.max(hits_of_track[:,i-2]) < max_boundaries[i-2]+d:
                a, p, output = PCAs(hits_of_track)
                a_array.append(a)
                l, start, end = length_track(hits_of_track)
                l_array.append(l)
                n_hits = len(output['true'])
                PCA_dir_tiled = np.tile(output['v_dir'], (n_hits, 1))
                if np.logical_and(a > v, l > b):
                    pca_q = np.tile(np.hstack([a, a, a]), (n_hits,1))
                    anode = np.tile([False, False, False], (n_hits,1))
                    track_number += 1
                    ev_labels = np.tile([muon_track_number+track_number, muon_track_number+track_number, muon_track_number+track_number], (n_hits,1))
                    muon_hits.append(hits_with_track)
                    if not np.any(true_hits):
                        true_hits = output['true']
                        reco_hits = output['reco']
                        PCA_dir = PCA_dir_tiled
                        labels = ev_labels
                        a2a = anode
                        pca_qual = pca_q
                    else:
                        true_hits = np.concatenate([true_hits, output['true']])
                        reco_hits = np.concatenate([reco_hits, output['reco']])
                        PCA_dir = np.concatenate([PCA_dir, PCA_dir_tiled])
                        labels = np.concatenate([labels, ev_labels])
                        a2a = np.concatenate([a2a, anode])
                        pca_qual = np.concatenate([pca_qual, pca_q])
                    break

        if hits_of_cluster != []:
            clus_arr.append(hits_of_cluster)
    return true_hits, reco_hits, PCA_dir, labels, muon_hits, track_number, a2a, pca_qual

