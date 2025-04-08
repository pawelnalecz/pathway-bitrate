from src.internal_abbreviations import has_exactly_inhibitors


val_name_pos_list = [
    (('STE1', [], None),                                   'STE1\nno inh',                   1.0),
    (('STE1', 'crizotinib', 0.3),                          'STE1\nALKi 0.3uM',               2.0),
    (('STE1', 'crizotinib', 1.0),                          'STE1\nALKi 1.0uM',               3.0),
    (('STE1', 'crizotinib', 3.0),                          'STE1\nALKi 3.0uM',               4.0),
    (('BEAS2B', [], None),                                 'BEAS2B\nno inh',                 5.5),
    (('BEAS2B', 'cyclosporin', 1.0),                       'BEAS2B\nCALCi 1uM',              6.5),
    (('BEAS2B', 'trametinib', 0.5),                        'BEAS2B\nMEKi 0.5uM',             7.5),
    (('BEAS2B', 'crizotinib', 0.3),                        'BEAS2B\nALKi 0.3uM',             8.5),
    (('BEAS2B', ['trametinib', 'cyclosporin'], [0.5, 1.]), 'BEAS2B\nMEKi 0.5uM\n+CALCi 1uM', 9.5),
]
set_to_label = {
    'main+STE1+0uM':                'STE1\nno inh',                    
    'main+STE1+criz03uM':           'STE1\nALKi 0.3uM',       
    'main+STE1+criz1uM':            'STE1\nALKi 1.0uM',      
    'main+STE1+criz3uM':            'STE1\nALKi 3.0uM',       
    'main+BEAS2B+0uM':              'BEAS2B\nno inh',         
    'main+BEAS2B+cycl1uM':          'BEAS2B\nCALCi 1uM',      
    'main+BEAS2B+tram05uM':         'BEAS2B\nMEKi 0.5uM',     
    'main+BEAS2B+criz03uM':         'BEAS2B\nALKi 0.3uM',     
    'main+BEAS2B+tram05uMcycl1uM':  'BEAS2B\nMEKi 0.5uM\n+CALCi 1uM',

    'main+STE1+criz':           'STE1\nALKi',       
}


def row_to_pos(row):
    positions = [
        pos for (cell_line, inh, inh_conc), _, pos  in val_name_pos_list
        if row['cell_line'] == cell_line and has_exactly_inhibitors(row, inh, inh_conc)
    ]
    assert len(positions) == 1, (row, positions)
    return positions[0]
set_types_and_colors = [

    ('main+BEAS2B+0uM', 'goldenrod'),
    ('main+BEAS2B+cycl1uM', 'gold'),
    ('main+BEAS2B+tram05uM', 'red'),
    # ('main+BEAS2B+criz03uM', 'sandybrown'),
    # ('main+BEAS2B+tram05uMcycl1uM', 'navajowhite'),
    # ('main+STE1+0uM', 'slategray'),
    ('main+STE1+criz', 'deepskyblue'),
]



def well_info_to_color(well_info):
    if well_info['cell_line'] == 'BEAS2B':
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] == 0:
            return 'goldenrod'
        if well_info['inh_trametinib'] >  0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] == 0:
            return 'red'
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] >  0 and well_info['inh_crizotinib'] == 0:
            return 'gold'
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] >  0:
            return 'sandybrown'
        if well_info['inh_trametinib'] >  0 and well_info['inh_cyclosporin'] >  0 and well_info['inh_crizotinib'] == 0:
            return 'navajowhite'
    if well_info['cell_line'] == 'STE1':
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] == 0:
            return 'slategray'
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] == .3:
            return 'skyblue'
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] == 1.:
            return 'deepskyblue'
        if well_info['inh_trametinib'] == 0 and well_info['inh_cyclosporin'] == 0 and well_info['inh_crizotinib'] == 3.:
            return 'dodgerblue'
    raise ValueError(f"No color defined for well info:\n{well_info[['cell_line', 'inh_trametinib', 'inh_cyclosporin', 'inh_crizotinib']]}")
        

def well_info_to_label(well_info):
    inh_text = (
        (f" + ALKi  {well_info['inh_crizotinib']}uM"   if well_info['inh_crizotinib']   > 0 else '')
      + (f" + MEKi  {well_info['inh_trametinib']}uM"   if well_info['inh_trametinib']   > 0 else '')
      + (f" + CALCi {well_info['inh_cyclosporin']}uM" if well_info['inh_cyclosporin'] > 0 else '')
    )
    return f"{well_info['cell_line']}{inh_text}"



