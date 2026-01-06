import torch

DATASET_PATH = 'modulation_dataset/'

def get_config(idx=0, n_channels=1):

    assert(idx >= 0)

    c = {

        'dataset_path': DATASET_PATH,
        'phaser': False,

        'n_channels': n_channels,

        'model': {
            'use_filters': True
        },

        'pedal': {
            'in_sig_cfg': 'train_input_250807_104006.wav',
            'in_shape': 'tri'
        },

        'train_data':
            {
                'seq_length_samples': 2 ** 18,
                'batch_size': 128,
            },
        'val_data':
        [
            {
                'start': 1071,
                'seq_length_samples': 2**18
            },
            {
                'start': 1071.0 + 391.0,
                'seq_length_samples': 2 ** 18
             }
        ],

        'loss_weights': {
                'convergence': 0.0,
                'magnitude': 0.0,
                'l_inf': 0.0,
                'phase': 1.0,
                'ceps_convergence': 0.0,
                'ceps_magnitude': 0.0,
                 'frame': 0.0,
                'log_difference': 0.0,
                'lsd': 0.0,
                'pre_emp': '',
                'n_filt': 0
        },

        'stft': {
            'n_win': 2048,
            'n_hop': 2048,
            'n_fft': 2048,
            'window': None,
            'synth_window': 'hann'
        },

        'trainer': {
            'max_epochs': 15000,
            'check_val_every_n_epoch': 750,
            'num_sanity_val_steps': 0,
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        },

        'early_stopping': {
            'monitor': 'val_loss_0',
            'stopping_threshold': 0.0015,
            'min_delta': 0.0,
            'patience': 100000,
        },

        'lr': 0.005,
        'dropout': 0.01,
        'log_audio_every_n_epoch': 5000,

        'mlp': {
                'in_dim': n_channels,
                'out_dim': n_channels,
                'hidden_dim': [16],
                'activation': torch.nn.Tanh()
                }
        }


    if idx < 10:

        c['project'] = 'bf-2_jaes'
        c['pedal']['in_filename'] = '03-bf-2_bypass-250807_1107.wav'
        if idx == 0:
            c['pedal']['out_filename'] = '06-bf-2_0_1_0.5_0-250807_1135.wav'
        elif idx == 1:
            c['pedal']['out_filename'] = '09-bf-2_0_1_0.5_0.5-250807_1242.wav'
        elif idx == 2:
            c['pedal']['out_filename'] = '19-bf-2_0_1.0_0.5_1.0-250807_1404.wav'
        elif idx == 3:
            c['pedal']['out_filename'] = '12-bf-2_0_0.5_0.5_0-250807_1308.wav'
        elif idx == 4:
            c['pedal']['out_filename'] = '15-bf-2_0_0.5_0.5_0.5-250807_1335.wav'


    elif idx < 20:
        c['project'] = 'sv-1_jaes'
        c['pedal']['in_filename'] = '04-sv-1_bypass-250807_1107.wav'
        if idx == 10:
            c['pedal']['out_filename'] = '13-sv-1_0.5_0.5_0_1-250807_1308.wav'
        elif idx == 11:
            c['pedal']['out_filename'] = '10-sv-1_0.5_0.5_0.5_1-250807_1242.wav'
        elif idx == 12:
            c['pedal']['out_filename'] = '16-sv-1_0.5_0.5_1_1-250807_1335.wav'
        elif idx == 13:
            c['pedal']['out_filename'] = '07-sv-1_0.5_1_0.5_1-250807_1135.wav'
        elif idx == 14:
            c['pedal']['out_filename'] = '20-sv-1_0.5_1_1_1-250807_1404.wav'


    elif idx < 100:
        c['project'] = 'digital_fx'
        c['pedal'] = {
            'in_sig_cfg': 'train_input_251021_103826.wav',
            'in_filename': 'train_input_251021_103826.wav',
            'in_shape': 'tri'
        }
        c['val_data'][0]['start'] = 107.99756
        c['val_data'][1]['start'] = 107.99756 + 392.32436

        if idx == 50:
            c['pedal']['out_filename'] = 'virtual_chorus_output_251021_103826.wav'
        elif idx == 51:
            c['pedal']['out_filename'] = 'virtual_flanger_output_251021_103836.wav'
        elif idx == 52:
            c['lr'] = 0.002
            c['phaser'] = True
            c['model']['num_allpass'] = 6
            c['pedal']['out_filename'] = 'virtual_phaser_output_251021_103913.wav'


    elif idx >= 100:
        c['project'] = 'smallstone_jaes'
        c['phaser'] = True
        c['lr'] = 0.002

        c['pedal']['in_filename'] = '05-smallstone_bypass-250807_1107.wav'

        if idx == 100:
            c['pedal']['out_filename'] = '14-smallstone_0.75_0-250807_1308.wav'
        if idx == 101:
            c['pedal']['out_filename'] = '08-smallstone_0.5_0-250807_1135.wav'
        if idx == 102:
            c['pedal']['out_filename'] = '21-smallstone_0.25_0-250807_1404.wav'
            c['train_data']['seq_length_samples'] = 2 ** 19
            c['val_data'][0]['seq_length_samples'] = 2 ** 19
            c['val_data'][1]['seq_length_samples'] = 2 ** 19
        if idx == 103:
            c['pedal']['out_filename'] = '17-smallstone_0.75_1-250807_1335.wav'
        if idx == 104:
            c['pedal']['out_filename'] = '11-smallstone_0.5_1-250807_1242.wav'


    return c
