from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from PIL import Image

from config.local_config import INPUT_TYPE
from src.internal_abbreviations import format_positive_inhibitors, format_well_id
from src import jax_protocol
from config import parameters

PREDEFINED_PROTOCOLS = '_predefined_protocols'

class DataManager:
    def __init__(self, data_dir, verbose=False):
        self.data_dir = Path(data_dir).resolve()
        self.verbose = verbose
        self._load_metadata()

    def get_image(self, well_id, time_in_seconds, channel_name):
        experiment = self.get_experiment(well_id)
        seconds_per_timepoint = self.get_seconds_per_timepoint(experiment)
        time_point_index = time_in_seconds // seconds_per_timepoint
        
        path_images_relative = self.wells.loc[well_id, 'path_images']
        path_images = self.data_dir / experiment / path_images_relative
        path_stt_metadata = path_images / 'shuttletracker_metadata.txt'

        channel_name_to_id = {metadata['name']: channel_id  for channel_id, metadata in _parse_shuttletracker_metadata(path_stt_metadata).items()}
        path_img = path_images / f"Img_t{time_point_index:04d}_c{channel_name_to_id[channel_name]}.tif"
        return _load_img(path_img)


    def get_stt_tracks(self, well_id):
        # find
        experiment = self.get_experiment(well_id)
        path_tracks_relative = self.wells.loc[well_id, 'path_tracks']
        path_tracks = self.data_dir / experiment / '01-tracking-per' / path_tracks_relative.split('/')[-1].replace('.pkl.gz', '-stt.pkl.gz')

        # load
        if INPUT_TYPE == 'PICKLE':
            tracks = pd.read_pickle(path_tracks)
        elif INPUT_TYPE == 'CSV':
            tracks = pd.read_csv(path_tracks, index_col=['track_id', 'time_in_seconds'])
        else:
            raise ValueError(f"INPUT_TYPE must be either 'PICKLE' or 'CSV, not {INPUT_TYPE}")

        # annotate valid
        valid_start = self.experiments.loc[experiment, 'valid_start']
        valid_end = self.experiments.loc[experiment, 'valid_end']
        tracks['valid'] = (
            (tracks.index.get_level_values('time_in_seconds') >= valid_start)
          & (tracks.index.get_level_values('time_in_seconds') <= valid_end)
        )

        return tracks


    def get_tracks(self, well_id):
        # find
        experiment = self.get_experiment(well_id)
        path_tracks_relative = self.wells.loc[well_id, 'path_tracks']
        path_tracks = self.data_dir / experiment / path_tracks_relative
        
        # load
        if INPUT_TYPE == 'PICKLE':
            tracks = pd.read_pickle(path_tracks)
        elif INPUT_TYPE == 'CSV':
            tracks = pd.read_csv(path_tracks, index_col=['track_id', 'time_in_seconds'])
        else:
            raise ValueError(f"INPUT_TYPE must be either 'PICKLE' or 'CSV, not {INPUT_TYPE}")

        # annotate valid
        valid_start = self.experiments.loc[experiment, 'valid_start']
        valid_end = self.experiments.loc[experiment, 'valid_end']
        tracks['valid'] = (
            (tracks.index.get_level_values('time_in_seconds') >= valid_start)
          & (tracks.index.get_level_values('time_in_seconds') <= valid_end)
        )

        return tracks


    def get_pulses(self, experiment):
        return self.pulses.loc[experiment]


    def get_experiment_info(self, experiment):
        return self.experiments.loc[experiment]
    

    def get_shuttletracker_metadata(self, well_id):
        experiment = self.get_experiment(well_id)
        
        path_images_relative = self.wells.loc[well_id, 'path_images']
        path_images = self.data_dir / experiment / path_images_relative
        path_stt_metadata = path_images / 'shuttletracker_metadata.txt'

        channel_metadata = _parse_shuttletracker_metadata(path_stt_metadata)
        all_metadata = {}
        for metadata in channel_metadata.values():
            all_metadata.update({
                f"{metadata['name']}_scale": metadata['scale'],
                f"{metadata['name']}_offset": metadata['offset'],
            })

        return pd.Series(all_metadata, name=well_id, dtype='object')


    def get_effective_time_range(self, experiment):
        pulses = self.pulses.loc[experiment]
        pulses = pulses[pulses['valid']]
        return pulses['time_in_seconds'].min(), pulses['time_in_seconds'].max()


    def get_receptor_thresholds(self, experiment):
        experiment_info = self.get_experiment_info(experiment)
        return experiment_info[['receptor_lower_thr', 'receptor_upper_thr']]


    def get_seconds_per_timepoint(self, experiment):
        return self.experiments.loc[experiment, 'seconds_per_timepoint']


    def get_seconds_per_timepoint_for_experiment_list(self, experiments):
        seconds_per_timepoint_s = set(self.get_seconds_per_timepoint(experiment) for experiment in experiments)
        assert len(seconds_per_timepoint_s) == 1, "Incompatible seconds per timepoint"
        return seconds_per_timepoint_s.pop()


    def get_date(self, experiment):
        return self.experiments.loc[experiment, 'date']


    def get_experiment(self, well_id):
        return self.wells.loc[well_id, 'experiment']


    def get_inhibitors(self):
        return self.inhibitors


    def get_well_info(self, well_id):
        return self.wells.loc[well_id]


    def set_type_to_well_ids(self, well_id, set_type):
        well_info = self.wells.loc[well_id]

        if set_type.endswith('-self'):
            set_type = set_type[:-len('-self')]
            no_self = True
        else:
            no_self = False

        match set_type:
            case 'main':
                set_id = 'main'
                wells_selected = (
                    self.wells[
                        self.wells['collection'] == 'main'
                    ]
                )

            case 'main+cell':
                cell_line = well_info['cell_line']
                set_id = f'main+{cell_line}'
                wells_selected = (
                    self.wells[
                        (self.wells['collection'] == 'main')
                      & (self.wells['cell_line'] == cell_line)
                    ]
                )

            case 'main+cell+inh':
                cell_line = well_info['cell_line']
                inhibitor_str = format_positive_inhibitors(well_info)
                set_id = f'main+{cell_line}+{inhibitor_str}'
                wells_selected = (
                    self.wells[
                        (self.wells['collection'] == 'main')
                      & (self.wells['cell_line'] == cell_line)
                      & (self.wells[self.inhibitor_cols] == well_info[self.inhibitor_cols]).all(axis=1)
                    ]
                )


            case 'main+cell+inhtype':
                cell_line = well_info['cell_line']
                inhibitor_str = format_positive_inhibitors(well_info, types_only=True)
                set_id = f'main+{cell_line}+{inhibitor_str}'
                wells_selected = (
                    self.wells[
                        (self.wells['collection'] == 'main')
                      & (self.wells['cell_line'] == cell_line)
                      & ((self.wells[self.inhibitor_cols] > 0) == (well_info[self.inhibitor_cols] > 0)).all(axis=1)
                    ]
                )

            case 'exp':
                experiment = well_info['experiment']
                set_id = f'{experiment}'
                wells_selected = (
                    self.wells[
                        (self.wells['experiment'] == experiment)
                    ]
                )

            case 'exp+inh':
                experiment = well_info['experiment']
                inhibitor_str = format_positive_inhibitors(well_info)
                set_id = f'{experiment}--{inhibitor_str}'
                wells_selected = (
                    self.wells[
                        (self.wells['experiment'] == well_info['experiment'])
                      & (self.wells[self.inhibitor_cols] == well_info[self.inhibitor_cols]).all(axis=1)
                    ]
                )

            case 'self':
                set_id = well_id
                wells_selected = self.wells.loc[[well_id]]

            case _:
                assert False, f'Unknown {set_type=}'

        if no_self:
            if well_id in wells_selected.index:
                set_id += f'--wo--{well_id}'
                wells_selected = wells_selected.drop(well_id)

        well_ids = (
            wells_selected
            .index
            .get_level_values('well_id')
            .tolist()
        )

        assert well_ids, f'No wells for training found for: {well_id=}, {set_type=}, {no_self=}!'

        return set_id, well_ids


    def _load_metadata(self):
        # load predefined protocols from data_dir/_predefined_protocols
        self.predefined_protocols = {}
        if (self.data_dir / PREDEFINED_PROTOCOLS).exists():
            for protocol_file in (self.data_dir / PREDEFINED_PROTOCOLS).iterdir():
                protocol = jax_protocol.Protocol(path=protocol_file)
                self.predefined_protocols.update({protocol.name: protocol})

        # parse data_dir and load metadata
        experiments = []
        pulses = []
        wells = []
        inhibitors = set()

        for path_experiment in self.data_dir.iterdir():
            if path_experiment.name.startswith('_'): 
                continue
            path_experiment_metadata = path_experiment / 'metadata.yaml'
            if not path_experiment_metadata.exists():
                if self.verbose:
                    print(f'WARNING: Experiment {path_experiment.name} has no metadata. Skipping...')
                continue

            with open(path_experiment_metadata) as f:
                metadata = yaml.safe_load(f)

            experiment = metadata['experiment']
            assert path_experiment.name == experiment, (
                "Metadata experiment name and directory name don't match: "
                f"{metadata['experiment']} != {path_experiment.name}"
            )

            experiment_pulses = metadata['pulses']
            experiment_wells = metadata['wells']
            del metadata['pulses']
            del metadata['wells']

            for well in experiment_wells:
                path_tracks = path_experiment / well['path_tracks']
                assert path_tracks.exists(), f"Tracks file does not exist: {path_tracks}"

            experiments.append({
                'collection': 'UNKNOWN',
                'valid_start': float('-inf'),
                'valid_end': float('+inf'),
                **parameters.DEFAULT_METADATA,
            } | metadata)

            pulses += [
                {
                    'experiment': experiment,
                    'pulse': pulse_id,
                    'time_in_seconds': pulse,
                }
                for pulse_id, pulse in enumerate(experiment_pulses)
            ]

            wells += [
                {
                    'experiment': experiment,
                } | well_info
                for well_info in experiment_wells
            ]

            for well in wells:
                if 'inhibitors' in well:
                    for inh in well['inhibitors']:
                        well[f"inh_{inh}"] = well['inhibitors'][inh] 
                        inhibitors.add(inh)
                    del well['inhibitors']
            
        experiments = pd.DataFrame(experiments).set_index('experiment')
        pulses = pd.DataFrame(pulses).set_index(['experiment', 'pulse'])
        wells = pd.DataFrame(wells)
        
        # No information about inhibitor means no inhibitor
        self.inhibitors = list(inhibitors)
        self.inhibitor_cols = [f"inh_{inh}" for inh in inhibitors]
        for inh_col in self.inhibitor_cols:
            wells[inh_col] = wells[inh_col].fillna(0.)

        # Add well_id and annotate valid pulses
        wells['well_id'] = wells.apply(format_well_id, axis=1)
        pulses['valid'] = (
            (pulses['time_in_seconds'].ge(experiments['valid_start']))
          & (pulses['time_in_seconds'].le(experiments['valid_end']))
        )

        # Add experiment cell line and collection to wells
        wells = wells.join(experiments[['cell_line', 'collection']], on='experiment')

        # Set wells' index
        wells = wells.set_index('well_id')

        self.experiments = experiments
        self.wells = wells
        self.pulses = pulses


def _parse_shuttletracker_metadata(stt_metadata_path):
    channels = {}
    with open(stt_metadata_path) as stt_file:
        for line in stt_file.readlines():
            line = line.strip().split()
            if line[0] == 'channel':
                channels.update({
                    int(line[1]): {
                        'name': line[2],
                        'color': line[3],
                        'scale': float(line[4]),
                        'offset': int(line[5]),
                    }
                })
    return channels


def _load_img(path):
    return np.array(Image.open(path)).astype('int32')
