from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

# AUXILIARY FUNCTIONS

# RULES

rule neighbors:
    output:
        neighs='cache/neighbors/per_well/{well_id}/neighbors.pkl.gz',
        dists='cache/neighbors/per_well/{well_id}/neighbor_distances.pkl.gz'
    resources:
        mem_gib=5
    run:
        well_id = wildcards.well_id

        tracks = DATA_MANAGER.get_tracks(well_id)

        K = 25

        neighs_all = {
            'track_id': [],
            'time_in_seconds': [],
            'neighs': [],   
            'dists': [],
        }

        for t, frame in tracks.groupby('time_in_seconds')[['nuc_center_x', 'nuc_center_y']]:
            nn = NearestNeighbors(n_neighbors=K+1)
            nn.fit(frame.to_numpy())
            dists, neighs = nn.kneighbors(frame.to_numpy())
            neighs_all['track_id'].append(frame.index.get_level_values('track_id'))
            neighs_all['time_in_seconds'].append(frame.index.get_level_values('time_in_seconds'))
            neighs_all['neighs'].append(frame.index.get_level_values('track_id').to_numpy()[neighs])
            neighs_all['dists'].append(dists)

        neighs_df = pd.DataFrame(
            np.concat([
                np.concat(neighs_all['track_id']).reshape(-1, 1),
                np.concat(neighs_all['time_in_seconds']).reshape(-1, 1),
                np.concat(neighs_all['neighs']),
            ], axis=1),
            columns=['track_id', 'time_in_seconds'] + list(range(K+1)),
            ).set_index(['track_id', 'time_in_seconds']).sort_index()

        neighs_df.to_pickle(str(output.neighs))

        del neighs_df


        dists_df = pd.DataFrame(
            np.concat([
                np.concat(neighs_all['track_id']).reshape(-1, 1),
                np.concat(neighs_all['time_in_seconds']).reshape(-1, 1),
                np.concat(neighs_all['dists']),
            ], axis=1),
            columns=['track_id', 'time_in_seconds'] + list(range(K+1)),
            ).set_index(['track_id', 'time_in_seconds']).sort_index()

        dists_df.to_pickle(str(output.dists))


rule neighbors_responding_fraction:
    input:
        tracks_mi='cache/tracks_mi/per_well/{well_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
        neighs='cache/neighbors/per_well/{well_id}/neighbors.pkl.gz',
    output:
         'cache/neighbors/per_well/{well_id}/neighbors_responding_fractions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv',
    run:
        neighs = pd.read_pickle(input.neighs)
        tracks_mi = pd.read_csv(input.tracks_mi, index_col='track_id')

        are_nearest_transmitting = pd.DataFrame({
                i: tracks_mi['is_transmitting'].reindex(neighs[i]).to_numpy()
                for i in neighs.columns
            }, 
            index=neighs.index,
        )

        neighbors_responding_fractions = are_nearest_transmitting.groupby(0).mean()
        neighbors_responding_fractions.to_csv(str(output))



rule track_confluence:
    input:
        neigh_dists='cache/neighbors/per_well/{well_id}/neighbor_distances.pkl.gz',
    output:
         'cache/neighbors/per_well/{well_id}/track_confluence.csv.gz',
    run:
        neigh_dists = pd.read_pickle(input.neigh_dists)

        mean_10th_neigh_dist = neigh_dists.groupby('track_id')[10].mean()

        track_confluence = 1. / (10 * np.pi * mean_10th_neigh_dist**2)

        track_confluence.name = 'confluence'

        track_confluence.to_csv(str(output))


rule track_confluence_set:
    input:
        inputs_set(['cache/neighbors/per_well/{well_id}/track_confluence.csv.gz'])
    output:
         'cache/neighbors/per_set/{set_id}/track_confluence.csv.gz',
    run:

        set_id = wildcards.set_id
        track_confluence = pd.concat([
                pd.read_csv(input[i], index_col='track_id')
                for i in range(len(SET_ID_TO_WELL_IDS[set_id]))
            ],
            names=['well_id'],
            keys=SET_ID_TO_WELL_IDS[set_id],
        )

        track_confluence.to_csv(str(output))

