import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio


def get_well_ids_table_by_experiment(experiment):

    match experiment:
        case '2023-08-28-BEAS2B--intensities':
            return [[
                f'{experiment}--000625s-0uM-rep1',
                f'{experiment}--00125s-0uM-rep1',
                f'{experiment}--0025s-0uM-rep1',
                f'{experiment}--005s-0uM-rep1',
                f'{experiment}--01s-0uM-rep1',
            ]]
        
        case '2024-01-08-STE1':
            return [
                [f'{experiment}--01s-0uM-rep1', f'{experiment}--1s-0uM-rep1', f'{experiment}--10s-0uM-rep1'],
                [f'{experiment}--01s-0uM-rep2', f'{experiment}--1s-0uM-rep2', None],
                [f'{experiment}--01s-criz03uM-rep1', f'{experiment}--1s-criz03uM-rep1', f'{experiment}--10s-criz03uM-rep1'],
                [f'{experiment}--01s-criz03uM-rep2', f'{experiment}--1s-criz03uM-rep2', None],
                [f'{experiment}--01s-criz1uM-rep1', f'{experiment}--1s-criz1uM-rep1', f'{experiment}--10s-criz1uM-rep1'],
                [f'{experiment}--01s-criz1uM-rep2', f'{experiment}--1s-criz1uM-rep2', None],
            ]
        
        case '2024-02-19-BEAS2B':
            return [[
                f'{experiment}--01s-0uM-rep1',
                f'{experiment}--01s-0uM-rep2',
                f'{experiment}--01s-0uM-rep3',
                f'{experiment}--01s-0uM-rep4',
            ]]

        case '2024-07-30-STE1' | '2024-08-08-STE1':
            return [
                [     f'{experiment}--01s-0uM-rep1',      f'{experiment}--01s-0uM-rep2'],
                [f'{experiment}--01s-criz03uM-rep1', f'{experiment}--01s-criz03uM-rep2'],
                [ f'{experiment}--01s-criz1uM-rep1',  f'{experiment}--01s-criz1uM-rep2'],
                [ f'{experiment}--01s-criz3uM-rep1',  f'{experiment}--01s-criz3uM-rep2'],
            ]

        case '2024-09-08-BEAS2B':
            return [[
                #f'{experiment}--01s-0uM-rep1',
                f'{experiment}--01s-0uM-rep2',
                f'{experiment}--01s-0uM-rep3',
            ]]

        case '2024-10-22-BEAS2B' | '2024-11-20-BEAS2B' | '2024-11-20-BEAS2B--first-11h' | '2024-11-27-BEAS2B':
            return [
                [     f'{experiment}--01s-0uM-rep1',      f'{experiment}--01s-0uM-rep2'],
                [f'{experiment}--01s-criz03uM-rep1', f'{experiment}--01s-criz03uM-rep2'],
                [f'{experiment}--01s-tram05uM-rep1', f'{experiment}--01s-tram05uM-rep2'],
                [ f'{experiment}--01s-cycl1uM-rep1',  f'{experiment}--01s-cycl1uM-rep2'],
            ]
        
        case '2024-09-18-BEAS2B-bad':
            return [
                [     f'{experiment}--01s-0uM-rep1',      f'{experiment}--01s-0uM-rep2'],
                [f'{experiment}--01s-criz03uM-rep1', f'{experiment}--01s-criz03uM-rep2'],
                [f'{experiment}--01s-tram01uM-rep1', f'{experiment}--01s-tram01uM-rep2'],
            ]
        
        case '2024-12-18-BEAS2B' | '2024-12-23-BEAS2B' | '2025-02-04-BEAS2B':
            return [
                [     f'{experiment}--01s-0uM-rep1',      f'{experiment}--01s-0uM-rep2'],
                [f'{experiment}--01s-tram05uM-rep1', f'{experiment}--01s-tram05uM-rep2'],
                [ f'{experiment}--01s-cycl1uM-rep1',  f'{experiment}--01s-cycl1uM-rep2'],
                [f'{experiment}--01s-tram05uMcycl1uM-rep1', f'{experiment}--01s-tram05uMcycl1uM-rep2'],
            ]

        case _:
            assert False, F"No rule to combine plots for experiment {experiment}!"


def get_testset_table_by_set_type(set_type):
    match set_type:
        case 'main':
            return [[
                'main'
            ]]

        case 'main+cell+inh':
            return [
                ['main+STE1+0uM',      'main+BEAS2B+0uM'            ],
                ['main+STE1+criz03uM', 'main+BEAS2B+criz03uM'       ],
                ['main+STE1+criz1uM',  'main+BEAS2B+tram05uM'       ],
                ['main+STE1+criz3uM',  'main+BEAS2B+cycl1uM'        ],
                [None,                 'main+BEAS2B+tram05uMcycl1uM'],
            ]
        
        case 'main+cell+inhtype':
            return [
                ['main+STE1+0uM',  'main+BEAS2B+0uM'     ],
                ['main+STE1+criz', 'main+BEAS2B+criz'    ],
                [None,             'main+BEAS2B+tram'    ],
                [None,             'main+BEAS2B+cycl'    ],
                [None,             'main+BEAS2B+tramcycl'],
            ]

        case _:
            assert False, F"No rule to combine plots for set type {set_type}!"



def inputs_plots_by_well_id(wildcards):
    well_ids_table = get_well_ids_table_by_experiment(wildcards.experiment)

    return {
        f'{row_idx},{col_idx}': f'{wildcards.plot_dir}/single/per_well/{well_id}/{wildcards.plot_type}.png'
        for row_idx, well_ids_row in enumerate(well_ids_table)
        for col_idx, well_id in enumerate(well_ids_row)
        if well_id
    }


def inputs_plots_by_set_type(wildcards):
    testset_ids_table = get_testset_table_by_set_type(wildcards.set_type)

    return {
        f'{row_idx},{col_idx}': f'{wildcards.plot_dir}/single/per_set/{set_id}/{wildcards.plot_type}.png'
        for row_idx, set_ids_row in enumerate(testset_ids_table)
        for col_idx, set_id in enumerate(set_ids_row)
        if set_id
    }


def combine_plots(input, output):
    loc_to_path = {
        tuple(map(int, loc_str.split(','))): path 
        for loc_str, path in input.items()
    }

    nrows = 1 + max(loc[0] for loc in loc_to_path.keys())
    ncols = 1 + max(loc[1] for loc in loc_to_path.keys())

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    widths = []
    heights = []
    is_ax_empty = np.full_like(axs, True)
    for loc, path in loc_to_path.items():
        ax = axs[loc]
        img = iio.imread(path)
        widths.append(img.shape[1])
        heights.append(img.shape[0])
        ax.imshow(img)
        ax.axis('off')
        is_ax_empty[loc] = False
    
    for ax, is_empty in zip(axs.ravel(), is_ax_empty.ravel()):
        if is_empty:
            ax.set_visible(False)

    dpi = 300
    fig.set_size_inches(ncols * np.mean(widths) / dpi, nrows * np.mean(heights) / dpi)
    fig.tight_layout(pad=0)
    fig.savefig(str(output), bbox_inches='tight', dpi=dpi)


# RULES:

rule combine_plots_by_well_id:
    input:
        unpack(inputs_plots_by_well_id)
    output:
        '{plot_dir}/combined/per_experiment/{experiment}/{plot_type}.png'
    run:
        combine_plots(input, output)
        


rule combine_plots_by_set_type:
    input:
        unpack(inputs_plots_by_set_type)
    output:
        '{plot_dir}/combined/per_set_type/{set_type}/{plot_type}.png'
    run:
        combine_plots(input, output)
        
