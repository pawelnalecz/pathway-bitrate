import pandas as pd
import numpy as np

# RULES

rule mean_per_to_nuc_trajectory:
    output:
        'cache/per_to_nuc/{per}/{well_or_set_id}/mean_per_to_nuc_trajectory.csv.gz',
    resources:
        mem_gib=12,
    run:
        per_well = wildcards.per == 'per_well'

        index_col = 'track_id' if per_well else ['well_id', 'track_id']
        well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id] 
    
        well_info = pd.concat(
            [
                pd.concat([
                    DATA_MANAGER.get_well_info(well_id), 
                    DATA_MANAGER.get_experiment_info(DATA_MANAGER.get_experiment(well_id)),
                    DATA_MANAGER.get_shuttletracker_scales_offsets(well_id),
                    ]).to_frame(well_id)
                for well_id in well_ids
            ],
            axis=1,
        ).T.infer_objects()

        well_info.index.name = 'well_id'
        assert len(well_info['reporter_channel'].drop_duplicates()) == 1
        reporter_channel = well_info['reporter_channel'].iloc[0]

        tracks = pd.concat((
                DATA_MANAGER.get_tracks_with_perinucs(well_id)
                for well_id in well_ids),
            names=['well_id'],
            keys=well_ids,
            )

        background_correction = (well_info[f'{reporter_channel}_background'] - well_info[f'{reporter_channel}_offset']) * 2 ** -(well_info[f'{reporter_channel}_scale'] - 8)
        tracks = tracks.join(background_correction.to_frame('background_correction'), on='well_id')
        tracks[f'per_to_nuc_{reporter_channel}_intensity_mean'] = (tracks[f'per_{reporter_channel}_intensity_mean'] - tracks['background_correction']) / (tracks[f'nuc_{reporter_channel}_intensity_mean'] - tracks['background_correction'])
        tracks['log_per_to_nuc_translocation'] = np.log(tracks[f'per_to_nuc_{reporter_channel}_intensity_mean'] )
        mean_trajectory = tracks.groupby('time_in_seconds')['log_per_to_nuc_translocation'].mean()

        mean_trajectory.to_csv(str(output))

