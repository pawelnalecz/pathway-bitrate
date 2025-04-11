# ADD src/ TO PATH
from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))


## CONFIGURATION

# '_' is a special character in the pipeline, do not use it in config names
DATASET_CONFIGS = {
    'ls+cell+inhs': dict(
        r_ts=[60, 120, 180, 240, 300, 360],
        extra_cols=[
            'X_ratio_std',
            'X_cell_line',
            'X_criz',
            'X_tram',
            'X_cycl',
        ],
    ),
    
    'raw': dict(
        r_ts=[60, 120, 180, 240, 300, 360],
        extra_cols=[], 
        add_log_l=False,
    ),
    'ls': dict(r_ts=[60, 120, 180, 240, 300, 360], extra_cols=['X_ratio_std']),
    'ls+cell': dict(r_ts=[60, 120, 180, 240, 300, 360], extra_cols=['X_ratio_std', 'X_cell_line']),
    'ls+conds': dict(r_ts=[60, 120, 180, 240, 300, 360], extra_cols=[
       'X_ratio_std',
       'X_cond_STE1_0uM',
       'X_cond_STE1_03uM',
       'X_cond_STE1_1uM',
       'X_cond_STE1_3uM',
       'X_cond_BEAS2B_0uM',
       'X_cond_BEAS2B_criz',
       'X_cond_BEAS2B_tram',
       'X_cond_BEAS2B_cycl',
       'X_cond_BEAS2B_trcy',
    ]),
    'lr': dict(r_ts=[60, 120, 180, 240, 300, 360], extra_cols=['X_responsiveness']),
    'w5+ls': dict(r_ts=[60, 120, 180, 240, 300, 360], extra_cols=['X_ratio_std'], rolling_min_window=5),
}


# '_' is a special character in the pipeline, do not use it in config names
MODEL_CONFIGS = {
    'nn': dict(hidden_layers=(40, 20), normalize=True),
}


# '_' is a special character in the pipeline, do not use it in config names
# Adidionally, '+' is a special symbol in TRAIN_CONFIGS to mark sequential training with different configs; do not use it in config names
TRAIN_CONFIGS = {
    'main-q0': dict(set_type='main', train_quality=0, train_steps=10_000, batch_size=10_000, protocol_id='long_experimental'),
    'main-q1': dict(set_type='main', train_quality=1, train_steps=10_000, batch_size=10_000, protocol_id='long_experimental'),
    
    'main-self-q0': dict(set_type='main-self', train_quality=0, train_steps=10_000, batch_size=10_000, protocol_id='long_experimental'),
    'main-self-q1': dict(set_type='main-self', train_quality=1, train_steps=10_000, batch_size=10_000, protocol_id='long_experimental'),

    'main-q0-20k': dict(set_type='main', train_quality=0, train_steps=20_000, batch_size=10_000, protocol_id='long_experimental'),
    'main-q0-50k': dict(set_type='main', train_quality=0, train_steps=50_000, batch_size=10_000, protocol_id='long_experimental'),

    'main-q0tr': dict(set_type='main', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q0', train_steps=10_000, batch_size=10_000, protocol_id='long_experimental'),
    'main-q1tr': dict(set_type='main', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=10_000, batch_size=10_000, protocol_id='long_experimental'),

    'opt-L1': dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='long_experimental', protocol_d2_L1_penalty=.003),
    'opt-L2': dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='long_experimental', protocol_d2_L2_penalty=.01),
    
    'opt-5-90protocol':      dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5'),
    'opt-5-90protocol-L1-0003':      dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5'),
    'opt-5-90protocol-L1-001':  dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L1_penalty=.01),
    'opt-5-90protocol-L1-01':   dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L1_penalty=.1),
    'opt-5-90protocol-L2-001':      dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L2_penalty=.01),
    'opt-5-90protocol-L2-003':  dict(set_type='main+cell+inh', train_quality=1, train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L2_penalty=.03),

    'opt-5-90protocol-q1tr':      dict(set_type='main+cell+inh', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5'),
    'opt-5-90protocol-L1-0003-q1tr':      dict(set_type='main+cell+inh', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L1_penalty=.003),
    'opt-5-90protocol-L1-001-q1tr':  dict(set_type='main+cell+inh', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L1_penalty=.01),
    'opt-5-90protocol-L1-01-q1tr':   dict(set_type='main+cell+inh', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L1_penalty=.1),
    'opt-5-90protocol-L2-001-q1tr':  dict(set_type='main+cell+inh', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L2_penalty=.01),
    'opt-5-90protocol-L2-003-q1tr':  dict(set_type='main+cell+inh', train_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', train_steps=5_000, batch_size=10_000, optimize_network=False, optimize_protocol=True, protocol_id='5-90gamma4x5', protocol_d2_L2_penalty=.03),
}

# '_' is a special character in the pipeline, do not use it in config names
TEST_CONFIGS = {
    'q-1': dict(test_quality=-1, protocol_id='long_experimental', reweight=False), 
    'q0': dict(test_quality=0, protocol_id='long_experimental', reweight=False), 
    'q1': dict(test_quality=1, protocol_id='long_experimental', reweight=False), 
    'q2': dict(test_quality=2, protocol_id='long_experimental', reweight=False), 
    'q1tr': dict(test_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', protocol_id='long_experimental', reweight=False), 
    'q0-reweight': dict(test_quality=0, protocol_id='long_experimental', reweight=True), 
    'q1-reweight': dict(test_quality=1, protocol_id='long_experimental', reweight=True), 
    'q0-reweight-optprotocol': dict(test_quality=0, protocol_id='_trained', reweight=True), 
    'q1-reweight-optprotocol': dict(test_quality=1, protocol_id='_trained', reweight=True), 
    'q1tr-reweight-optprotocol': dict(test_quality='transmitting', transmitting_train_id='main-q0', transmitting_test_id='q1', protocol_id='_trained', reweight=True), 
}

