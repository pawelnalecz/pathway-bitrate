import pandas as pd

# RULES

rule mi_all:
    input:
        expand(
            'cache/mi/per_well/{well_id}/mi_{dataset_id}_{model_id}_{train_id}_{test_id}.csv',
            well_id=WELLS_SELECTED.index.get_level_values('well_id'),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )
    output:
        '{output_dir}/mi_all.csv'
    run:
        results = pd.concat([
            pd.read_csv(str(path)) for path in input
        ])
        results.to_csv(str(output), index=False)


rule measures_all:
    input:
        expand(
            'cache/measures/per_well/{well_id}/measures_q{{quality}}.csv',
            well_id=WELLS_SELECTED.index.get_level_values('well_id'),
        )
    output:
        '{output_dir}/measures_all_q{quality}.csv'
    run:
        results = pd.concat([
            pd.read_csv(str(path)) for path in input
        ])
        results.to_csv(str(output), index=False)

rule transmitting_analysis_all:
    input:
        expand(
            'cache/tracks_mi/per_well/{well_id}/transmitting_analysis_{dataset_id}_{model_id}_{train_id}_{test_id}.csv',
            well_id=WELLS_SELECTED.index.get_level_values('well_id'),
            dataset_id=DATASET_IDS,
            model_id=MODEL_IDS,
            train_id=TRAIN_IDS,
            test_id=TEST_IDS,
        )
    output:
        '{output_dir}/transmitting_analysis_all.csv'
    run:
        results = pd.concat([
            pd.read_csv(str(path)) for path in input
        ])
        results.to_csv(str(output), index=False)



