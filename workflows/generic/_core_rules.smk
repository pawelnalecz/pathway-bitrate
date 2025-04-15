import numpy as np
import pandas as pd
from itertools import cycle

from typing import Any, List, Callable

from src import data_preprocessing
from src import create_dataset
from src import jax_protocol
from src import jax_dataloader
from src import jax_nn
from src import jax_evaluation
from config import parameters
from src.stat_utils import get_sorted

from config.configs import (
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    TRAIN_CONFIGS,
    TEST_CONFIGS,
)

# ASSUMES THAT THE FOLLOWING ARE DEFINED:
#   * DATA_MANAGER
#   * WELLS_SELECTED
#   * SET_TYPES

#   * DATASET_IDS
#   * MODEL_IDS
#   * TRAIN_IDS
#   * TEST_IDS


SET_ID_TO_WELL_IDS = dict([
    DATA_MANAGER.set_type_to_well_ids(well_id, train_on)
    for well_id in WELLS_SELECTED.index
    for train_on in SET_TYPES
])


def load_dataset(well_id, quality, transmitting_test_id=None, tracks_mi_path=None, **dataset_config) -> pd.DataFrame:
    # separate files for all tracks including quality < 0 introduced for performance
    if quality == 'transmitting':
        assert transmitting_test_id is not None
        original_quality = TEST_CONFIGS[transmitting_test_id]['test_quality']
        while original_quality == 'transmitting':
            transmitting_test_id = TEST_CONFIGS[transmitting_test_id]['transmitting_test_id']
            original_quality = TEST_CONFIGS[transmitting_test_id]['test_quality']
    else:
        original_quality = quality

    if original_quality >= 0:
        tracks_path = f'cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz'
        tracks_info_path = f'cache/preprocessed/per_well/{well_id}/tracks_info.csv.gz'
    else:
        tracks_path = f'cache/preprocessed/per_well/{well_id}/tracks_preprocessed_all.pkl.gz'
        tracks_info_path = f'cache/preprocessed/per_well/{well_id}/tracks_info_all.csv.gz'


    experiment = DATA_MANAGER.get_experiment(well_id)
    seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)

    tracks = pd.read_pickle(tracks_path)
    tracks_info = pd.read_csv(tracks_info_path, index_col='track_id')
    tracks_mi = pd.read_csv(tracks_mi_path, index_col='track_id') if quality == 'transmitting' else None
    
    return create_dataset.create_dataset(
        tracks,
        tracks_info,
        well_id=well_id,
        quality=quality,
        tracks_mi=tracks_mi,
        seconds_per_timepoint=seconds_per_timepoint,
        **dataset_config,
    )

def train_ids_to_set_id(well_id, train_ids) -> str:
    return '/'.join(
        DATA_MANAGER.set_type_to_well_ids(well_id, set_type=TRAIN_CONFIGS[train_id]['set_type'])[0]
        for train_id in train_ids
    )


def inputs_model(wildcards) -> str:
    well_id = wildcards.well_id
    train_ids = wildcards.train_id.split('+')
    trainset_id = train_ids_to_set_id(well_id, train_ids)
    return f'cache/train/{trainset_id}/model_{{dataset_id}}_{{model_id}}_{{train_id}}.eqx'


def inputs_train_protocol(wildcards) -> str:
    train_ids = wildcards.train_id.split('+')
    train_id = train_ids[-1]
    train_config = TRAIN_CONFIGS[train_id]
    return f"cache/protocols/{train_config['protocol_id']}.yaml"


def inputs_test_protocol(wildcards) -> str:
    test_config = TEST_CONFIGS[wildcards.test_id]
    if test_config['protocol_id'] == '_trained':
        well_id = wildcards.well_or_set_id if hasattr(wildcards, 'per') else wildcards.well_id
        train_ids = wildcards.train_id.split('+')
        trainset_id = train_ids_to_set_id(well_id, train_ids)
        return f'cache/train/{trainset_id}/protocol_{{dataset_id}}_{{model_id}}_{{train_id}}.yaml'
    return f"cache/protocols/{test_config['protocol_id']}.yaml"


def inputs_set(files, set_id=None) -> Callable[[Any], List[str]]:
    return lambda wildcards: [
        path.format(well_id=well_id)
        for well_id in SET_ID_TO_WELL_IDS[wildcards.set_id.split('/')[-1] if set_id is None else set_id]
        for path in files
    ]


def input_shorter_model(wildcards) -> List[str] | str:
    set_ids = wildcards.set_id.split('/')
    train_ids = wildcards.train_id.split('+')
    if len(set_ids) > 1:
        return f"cache/train/{'/'.join(set_ids[:-1])}/model_{{dataset_id}}_{{model_id}}_{'+'.join(train_ids[:-1])}.eqx"
    return []


def input_tracks_mi_for_testing(wildcards) -> List[str] | str:
    test_config = TEST_CONFIGS[wildcards.test_id]
    if test_config['test_quality'] == 'transmitting':
        return f"cache/tracks_mi/per_well/{{well_id}}/tracks_mi_{{dataset_id}}_{{model_id}}_{test_config['transmitting_train_id']}_{test_config['transmitting_test_id']}.csv.gz"
    return []


def input_tracks_mi_for_training(wildcards) -> List[str] | str:
    set_ids = wildcards.set_id.split('/')
    set_id = set_ids[-1]
    train_ids = wildcards.train_id.split('+')
    train_config = TRAIN_CONFIGS[train_ids[-1]]
    if train_config['train_quality'] == 'transmitting':
        return [
            f"cache/tracks_mi/per_well/{well_id}/tracks_mi_{{dataset_id}}_{{model_id}}_{train_config['transmitting_train_id']}_{train_config['transmitting_test_id']}.csv.gz"
            for well_id in SET_ID_TO_WELL_IDS[set_id]
        ]
    return []


def inputs_tracks_preprocessed(wildcards) -> List[str] | str:
    quality_suffix = '_all' if int(wildcards.quality) < 0 else ''
    if not hasattr(wildcards, 'per'):
        return f"cache/preprocessed/per_well/{{well_id}}/tracks_preprocessed{quality_suffix}.pkl.gz"
    elif wildcards.per == 'per_well':
        return [f"cache/preprocessed/per_well/{{well_or_set_id}}/tracks_preprocessed{quality_suffix}.pkl.gz"]
    elif wildcards.per == 'per_set':
        return [
            f"cache/preprocessed/per_well/{well_id}/tracks_preprocessed{quality_suffix}.pkl.gz"
            for well_id in SET_ID_TO_WELL_IDS[wildcards.well_or_set_id]
        ]
    else:
        raise ValueError(f'per must be "per_well" or "per_set", but {wildcards.per} found')


def inputs_tracks_info(wildcards) -> str:
    quality_suffix = '_all' if int(wildcards.quality) < 0 else ''
    if not hasattr(wildcards, 'per'):
        return f"cache/preprocessed/per_well/{{well_id}}/tracks_info{quality_suffix}.csv.gz"
    else:
        return f"cache/preprocessed/{{per}}/{{well_or_set_id}}/tracks_info{quality_suffix}.csv.gz"


# RULES


rule transmitting_analysis:
    input:
        tracks_mi='cache/tracks_mi/per_well/{well_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz',
    output:
        'cache/tracks_mi/per_well/{well_id}/transmitting_analysis_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
    run:
        experiment = DATA_MANAGER.get_experiment(wildcards.well_id)
        seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)
        bph = 60 * 60 / seconds_per_timepoint / np.log(2)
        
        # prepare infos
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)
        well_info = DATA_MANAGER.get_well_info(wildcards.well_id)

        dataset_config = DATASET_CONFIGS[wildcards.dataset_id].copy()
        for list_col in ['extra_cols', 'r_ts']:
            if list_col in dataset_config:
                dataset_config[list_col] = str(dataset_config[list_col])
        
        model_config = MODEL_CONFIGS[wildcards.model_id]
        model_config['hidden_layers'] = str(model_config['hidden_layers'])

        train_ids = wildcards.train_id.split('+')
        train_ids_dict = {f"train_id_{it}": train_id for it, train_id in enumerate(train_ids)}
        train_configs = [TRAIN_CONFIGS[train_id] for train_id in train_ids]
        train_configs_dict = {f"{key}_{it}": train_config[key] for it, train_config in enumerate(train_configs) for key in train_config}
        
        test_config  = TEST_CONFIGS[wildcards.test_id]

        tracks_mi = pd.read_csv(input.tracks_mi, index_col='track_id')

        weighted_tracks_mi = (
            tracks_mi.mul(tracks_mi['slots'].to_numpy() / tracks_mi['slots'].sum(), axis=0) 
        )

        fraction_transmitting = weighted_tracks_mi['is_transmitting'].sum()
        field_mean_for_transmitting = weighted_tracks_mi[tracks_mi['is_transmitting']].sum() / fraction_transmitting

        transmitting_analysis = pd.DataFrame({
            'well_id': wildcards.well_id,
            'dataset_id': wildcards.dataset_id,
            'model_id': wildcards.model_id,
            **train_ids_dict,
            'test_id': wildcards.test_id,
            'cell_line': experiment_info['cell_line'],
            **well_info,
            **dataset_config,
            **model_config,
            **train_configs_dict,
            **test_config,
            'fraction_transmitting': fraction_transmitting,
            'mi_ce_transmitting': field_mean_for_transmitting['mi_cross_per_slot'] * bph,
            're_ce_transmitting': field_mean_for_transmitting['re_cross_per_slot'] * bph,
            'pe_transmitting': field_mean_for_transmitting['pe_per_slot'] * bph,
        }, index=[0])
        transmitting_analysis.to_csv(str(output), index=False)


rule tracks_mi:
    input:
        predictions='cache/predictions/{per}/{well_or_set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
    output:
        'cache/tracks_mi/{per}/{well_or_set_id}/tracks_mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
    resources:
        mem_gib=lambda wc, input: 0.5 + 0.5 * len(input)
    run:
        per_well = wildcards.per == 'per_well'
        index_col = 'track_id' if per_well else ['well_id', 'track_id']
        predictions = pd.read_csv(input.predictions, index_col=index_col)

        predictions['pe'] = jax_evaluation.cross_entropy_from_logit(predictions['p_prior_logit'], predictions['y'])
        predictions['re_cross'] = jax_evaluation.cross_entropy_from_logit(predictions['p_predicted_logit'], predictions['y'])
        predictions['mi_cross'] = predictions['pe'] - predictions['re_cross']

        tracks_mi = pd.DataFrame()
        tracks_mi['slots'] = predictions.groupby(index_col).size()

        for col in ['mi_cross', 're_cross', 'pe']:
            tracks_mi[f'{col}_total'] = predictions.groupby(index_col)[col].sum()
            tracks_mi[f'{col}_per_slot'] = tracks_mi[f'{col}_total'] / tracks_mi['slots']

        ## determine transmitting cells

        field_sorted, slots_sorted = get_sorted(tracks_mi, 'mi_cross_per_slot', 'mi_cross_per_slot')

        slots_sorted_normalized = slots_sorted / slots_sorted.sum()
        field_cum = np.cumsum(field_sorted * slots_sorted_normalized.to_numpy())

        average = field_cum.iloc[-1]

        idx_max = np.searchsorted(-field_sorted, 0)
        idx_fraction_transmitting = np.searchsorted(field_cum.iloc[:idx_max], average)

        is_transmitting = pd.Series(
            np.arange(len(slots_sorted)) < idx_fraction_transmitting,
            index=slots_sorted.index,
        )

        tracks_mi = tracks_mi.join(is_transmitting.to_frame('is_transmitting'))

        tracks_mi.to_csv(str(output))



rule mi:
    input:
        predictions='cache/predictions/per_well/{well_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
    output:
        'cache/mi/per_well/{well_id}/mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv'
    resources:
        mem_gib=1
    run:
        experiment = DATA_MANAGER.get_experiment(wildcards.well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)

        # prepare infos
        well_info = DATA_MANAGER.get_well_info(wildcards.well_id)

        dataset_config = DATASET_CONFIGS[wildcards.dataset_id].copy()
        for list_col in ['extra_cols', 'r_ts']:
            if list_col in dataset_config:
                dataset_config[list_col] = str(dataset_config[list_col])

        model_config = MODEL_CONFIGS[wildcards.model_id]
        model_config['hidden_layers'] = str(model_config['hidden_layers'])

        train_ids = wildcards.train_id.split('+')
        train_ids_dict = {f"train_id_{it}": train_id for it, train_id in enumerate(train_ids)}
        train_configs = [TRAIN_CONFIGS[train_id] for train_id in train_ids]
        train_configs_dict = {f"{key}_{it}": train_config[key] for it, train_config in enumerate(train_configs) for key in train_config}

        test_config  = TEST_CONFIGS[wildcards.test_id]

        predictions = pd.read_csv(input.predictions)
        # compute mi
        re_naive = jax_evaluation.reconstruction_naive_entropy_per_slot_from_predictions(predictions)
        mi_naive = jax_evaluation.mutual_information_naive_per_slot_from_predictions(predictions)
        re_cross = jax_evaluation.reconstruction_cross_entropy_per_slot_from_predictions(predictions)
        mi_cross = jax_evaluation.mutual_information_cross_per_slot_from_predictions(predictions)
        pe = mi_cross + re_cross

        # to bit per hour
        seconds_per_timepoint = DATA_MANAGER.get_seconds_per_timepoint(experiment)
        bph = 60 * 60 / seconds_per_timepoint / np.log(2)

        mi_df = pd.DataFrame({
            'well_id': wildcards.well_id,
            'dataset_id': wildcards.dataset_id,
            'model_id': wildcards.model_id,
            **train_ids_dict,
            'test_id': wildcards.test_id,
            'cell_line': experiment_info['cell_line'],
            **well_info,
            **dataset_config,
            **model_config,
            **train_configs_dict,
            **test_config,
            'pe': pe * bph,
            're': re_naive * bph,
            're_ce': re_cross * bph,
            'mi': mi_naive * bph,
            'mi_ce': mi_cross * bph,
        }, index=[0])
        mi_df.to_csv(str(output), index=False)


rule testset_predictions:
    input:
        inputs_set([
            'cache/predictions/per_well/{well_id}/predictions_{{dataset_id}}_{{model_id}}_{{train_id}}_{{test_id}}.csv.gz',
        ])
    output:
        'cache/predictions/per_set/{set_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
    resources:
        mem_gib=lambda wc, input: 0.5 + 0.5 * len(input)
    run:
        well_ids = SET_ID_TO_WELL_IDS[wildcards.set_id]
        testset_predictions = pd.concat(
            (pd.read_csv(predictions, index_col='track_id') for predictions in input),
            names=['well_id'],
            keys=well_ids
        )
        testset_predictions.to_csv(str(output))


rule predictions:
    input:
        tracks_preprocessed='cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz',
        tracks_info='cache/preprocessed/per_well/{well_id}/tracks_info.csv.gz',
        model=inputs_model,
        protocol=inputs_test_protocol,
        tracks_mi=input_tracks_mi_for_testing,
    output:
        'cache/predictions/per_well/{well_id}/predictions_{dataset_id}_{model_id}_{train_id}_{test_id}.csv.gz'
    resources:
        mem_gib=4
    run:
        test_config = TEST_CONFIGS[wildcards.test_id]
        reweight = test_config['reweight']
        quality = test_config['test_quality']

        transmitting_test_id = test_config['transmitting_test_id'] if quality == 'transmitting' else None
        tracks_mi_path = input.tracks_mi if quality == 'transmitting' else None

        dataset = load_dataset(
            wildcards.well_id,
            quality=quality,
            transmitting_test_id=transmitting_test_id,
            tracks_mi_path=tracks_mi_path,
            **DATASET_CONFIGS[wildcards.dataset_id],
        )

        dataloader = jax_dataloader.DataLoader(dataset=dataset)
        protocol = jax_protocol.Protocol(path=input.protocol)
        model = jax_nn.NnModel(path=input.model)

        predictions = jax_evaluation.evaluate_on_full_dataset(model, dataloader, protocol, reweight=reweight)
        predictions.to_csv(str(output), index=False)


rule train_model:
    input:
        inputs_set([
            'cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz',
            'cache/preprocessed/per_well/{well_id}/tracks_info.csv.gz',
        ]),
        protocol=inputs_train_protocol,
        model=input_shorter_model,
        tracks_mi=input_tracks_mi_for_training,
    output:
        model='cache/train/{set_id}/model_{dataset_id}_{model_id}_{train_id}.eqx',
        log='cache/train/{set_id}/log_{dataset_id}_{model_id}_{train_id}.csv',
        protocol='cache/train/{set_id}/protocol_{dataset_id}_{model_id}_{train_id}.yaml'
    resources:
        mem_gib=lambda wc, input: 1 + 0.5 * len(input)
    run:
        set_ids = wildcards.set_id.split('/')
        train_ids = wildcards.train_id.split('+')
        assert len(train_ids) == len(set_ids), f'Train_ids {train_ids} and set_ids {set_ids} must have equal length.'

        is_first_training_round = len(set_ids) == 1

        set_id = set_ids[-1]
        train_id = train_ids[-1]

        well_ids = SET_ID_TO_WELL_IDS[set_id]

        dataset_config = DATASET_CONFIGS[wildcards.dataset_id]
        train_config = TRAIN_CONFIGS[train_id].copy()
        del train_config['set_type']
        model_config = MODEL_CONFIGS[wildcards.model_id]

        quality = train_config['train_quality']
        del train_config['train_quality']

        tracks_mi_paths = input.tracks_mi if quality == 'transmitting'  else cycle([None])

        trainset = pd.concat([
            load_dataset(
                well_id,
                quality=quality,
                transmitting_test_id=train_config['transmitting_test_id'] if quality == 'transmitting' else None,
                tracks_mi_path=tracks_mi_path,
                **DATASET_CONFIGS[wildcards.dataset_id],
                )
            for well_id, tracks_mi_path in zip(well_ids, tracks_mi_paths)
        ])

        if quality == 'transmitting':
            del train_config['transmitting_train_id']
            del train_config['transmitting_test_id']

        dataloader = jax_dataloader.DataLoader(dataset=trainset)

        protocol = jax_protocol.Protocol(path=input.protocol)
        del train_config['protocol_id']
        # check that all necessary L, I pairs are available
        dataloader.check_protocol_compatible(protocol)

        if is_first_training_round:
            model = jax_nn.NnModel(dataloader=dataloader, **model_config)
        else:
            model = jax_nn.NnModel(path=input.model)

        model.train(protocol, dataloader=dataloader, **train_config)

        model.save(str(output['model']))
        model.training_log.to_csv(str(output['log']), index=False)

        assert model.protocol is not None
        model.protocol.save(output['protocol'])


rule testset_tracks_info:
    input:
        inputs_set(['cache/preprocessed/per_well/{well_id}/tracks_info.csv.gz']),
    output:
        'cache/preprocessed/per_set/{set_id}/tracks_info.csv.gz',
    resources:
        mem_gib=lambda wc, input: 0.5 + 0.5 * len(input)
    run:
        well_ids = SET_ID_TO_WELL_IDS[wildcards.set_id]
        testset_track_info = pd.concat(
            (pd.read_csv(predictions, index_col='track_id') for predictions in input),
            names=['well_id'],
            keys=well_ids
            )
        testset_track_info.to_csv(str(output))


rule tracks_info_all:
    input:
        tracks_preprocessed='cache/preprocessed/per_well/{well_id}/tracks_preprocessed_all.pkl.gz',
    output:
        'cache/preprocessed/per_well/{well_id}/tracks_info_all.csv.gz'
    resources:
        mem_mib=512
    run:
        experiment = DATA_MANAGER.get_experiment(wildcards.well_id)
        well_info = DATA_MANAGER.get_well_info(wildcards.well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)

        tracks = pd.read_pickle(input.tracks_preprocessed)
        pulses = DATA_MANAGER.get_pulses(experiment)

        tracks_info = data_preprocessing.create_tracks_info(
            tracks=tracks,
            pulses=pulses,
            well_info=well_info,
            experiment_info=experiment_info,
        )
        tracks_info.to_csv(str(output))


rule tracks_info:
    input:
        tracks_preprocessed='cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz',
    output:
        'cache/preprocessed/per_well/{well_id}/tracks_info.csv.gz'
    resources:
        mem_mib=512
    run:
        experiment = DATA_MANAGER.get_experiment(wildcards.well_id)
        well_info = DATA_MANAGER.get_well_info(wildcards.well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)

        tracks = pd.read_pickle(input.tracks_preprocessed)
        pulses = DATA_MANAGER.get_pulses(experiment)

        tracks_info = data_preprocessing.create_tracks_info(
            tracks=tracks,
            pulses=pulses,
            well_info=well_info,
            experiment_info=experiment_info,
        )
        tracks_info.to_csv(str(output))


rule mean_trajectory:
    input:
        tracks_preprocessed=inputs_tracks_preprocessed,
        tracks_info=inputs_tracks_info,
    output:
        'cache/mean_trajectory/{per}/{well_or_set_id}/mean_trajectory_q{quality}.csv.gz',
    resources:
        mem_gib=6,
    run:
        per_well = wildcards.per == 'per_well'
        index_col = 'track_id' if per_well else ['well_id', 'track_id']
        well_ids = [wildcards.well_or_set_id] if per_well else SET_ID_TO_WELL_IDS[wildcards.well_or_set_id]
        tracks = pd.concat(
            (pd.read_pickle(tracks_file) for tracks_file in input.tracks_preprocessed),
            names=['well_id'],
            keys=well_ids
        )
        tracks['log_translocation'] = np.log(tracks['translocation'])
        tracks_info = pd.read_csv(input.tracks_info, index_col=index_col)

        tracks = tracks[tracks.join(tracks_info['quality'], on='track_id' if per_well else ['well_id', 'track_id'])['quality'] >= int(wildcards.quality)]

        mean_trajectory = tracks.groupby('time_in_seconds')['log_translocation'].mean()
        mean_trajectory.to_csv(str(output))


rule tracks_preprocessed_all:
    output:
        'cache/preprocessed/per_well/{well_id}/tracks_preprocessed_all.pkl.gz'
    resources:
        mem_gib=10
    run:
        well_id = wildcards.well_id
        experiment = DATA_MANAGER.get_experiment(well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)

        tracks = DATA_MANAGER.get_tracks(well_id)
        pulses = DATA_MANAGER.get_pulses(experiment)

        tracks_preprocessed = data_preprocessing.preprocess_tracks(
            tracks=tracks,
            pulses=pulses,
            experiment_info=experiment_info,
            remove_short_tracks=False,
        )

        tracks_preprocessed.to_pickle(str(output))


rule tracks_preprocessed:
    output:
        'cache/preprocessed/per_well/{well_id}/tracks_preprocessed.pkl.gz'
    resources:
        mem_gib=10
    run:
        well_id = wildcards.well_id
        experiment = DATA_MANAGER.get_experiment(well_id)
        experiment_info = DATA_MANAGER.get_experiment_info(experiment)

        tracks = DATA_MANAGER.get_tracks(well_id)
        pulses = DATA_MANAGER.get_pulses(experiment)

        tracks_preprocessed = data_preprocessing.preprocess_tracks(
            tracks=tracks,
            pulses=pulses,
            experiment_info=experiment_info,
        )

        tracks_preprocessed.to_pickle(str(output))

rule protocol:
    output:
        'cache/protocols/{protocol_id}.yaml'
    run:
        if wildcards.protocol_id in DATA_MANAGER.predefined_protocols:
            DATA_MANAGER.predefined_protocols[wildcards.protocol_id].save(str(output))
        else:
            protocol = jax_protocol.create_named_protocol(wildcards.protocol_id)
            protocol.save(str(output))
