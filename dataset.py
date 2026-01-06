import random
from torch.utils.data import Dataset
import numpy as np
from utils import STFT
import torch
from scipy.signal import medfilt
import pandas as pd
from scipy.io import wavfile
import torchaudio
from os.path import join


class SequenceDataset(Dataset):
    def __init__(self, input, target, sequence_length):

        if sequence_length is None:
            self._sequence_length = input.shape[1]
        else:
            self._sequence_length = sequence_length

        self.input_sequence = self.wrap_to_sequences(input, self._sequence_length)
        self._len = self.input_sequence.shape[0]
        self.target_sequence = self.wrap_to_sequences(target, self._sequence_length)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.input_sequence[index, :, :], self.target_sequence[index, :, :]

    # wraps data from  [channels, samples] -> [sequences, samples, channels]
    def wrap_to_sequences(self, data, sequence_length):
        num_sequences = int(np.floor(data.shape[1] / sequence_length))
        truncated_data = data[:, 0:(num_sequences * sequence_length)]
        wrapped_data = truncated_data.transpose(0, 1).reshape((num_sequences, sequence_length))
        return wrapped_data


class ModulationDataset(SequenceDataset):
    def __init__(self, input, target, sequence_length, stft_args=None):
        super().__init__(input, target, sequence_length)
        if stft_args is None:
            stft_class = STFT()
        else:
            stft_class = STFT(**stft_args)

        in_frames = stft_class.framify(self.input_sequence)
        out_frames = stft_class.framify(self.target_sequence)
        X = stft_class.stft(self.input_sequence)
        Y = stft_class.stft(self.target_sequence)


        self.in_frames = in_frames.squeeze()
        self.target_frames = out_frames.squeeze()
        self.input_stft = X.squeeze()
        self.target_stft = Y.squeeze()

        self._len = self.input_stft.shape[-1]

    def __getitem__(self, item):
        batch = {
            'in': self.in_frames[..., item],
            'out': self.target_frames[..., item],
            'in_stft': self.input_stft[..., item],
            'out_stft': self.target_stft[..., item],
            'frame_idx': item
        }
        return batch


def load_train_signal(timestamp_or_filename, n_win, shape, pulse_frac=2):

    if '.wav' in timestamp_or_filename:
        cfg_filename = timestamp_or_filename.split('.wav')[0]
        cfg_filename += '_cfg.csv'
        sig_filename = timestamp_or_filename
    else:
        cfg_filename = 'train_input_{}_cfg.csv'.format(timestamp_or_filename)
        sig_filename = 'train_input_{}.wav'.format(timestamp_or_filename)
    df = pd.read_csv(cfg_filename, index_col=0)
    sr, sig_full = wavfile.read(sig_filename)

    df = df[df['shape'] == shape]
    df = df[df['n_win'] == n_win]
    if 'pulse_frac' in df:
        df = df[df['pulse_frac'] == pulse_frac]

    assert len(df) > 0

    start = int(df['timestamp_start_samples'].iloc[0])
    end = int(df['timestamp_end_samples'].iloc[0])

    d = {
        'signal': sig_full[start:end],
        'timestamp_start': start,
        'timestamp_end': end
    }
    return d


def get_signal_chunks(cfg, as_numpy=False, shape_override=''):
    if shape_override != '':
        shape = shape_override
    else:
        shape = cfg['pedal']['in_shape']
    sig_cfg = load_train_signal(join(cfg['dataset_path'], cfg['pedal']['in_sig_cfg']),
                                n_win=cfg['stft']['n_fft'], shape=shape)
    x, fs = torchaudio.load(join(cfg['dataset_path'], cfg['pedal']['in_filename']))
    y, fs = torchaudio.load(join(cfg['dataset_path'], cfg['pedal']['out_filename']))


    frame_end = sig_cfg['timestamp_end']
    frame_start = frame_end - cfg['train_data']['seq_length_samples']
    x_train, y_train = x[:, frame_start:frame_end], y[:, frame_start:frame_end]

    d = {
        'train': [x_train, y_train],
    }
    for i in range(2):
        frame_start = int(fs*cfg['val_data'][i]['start'])
        frame_end = frame_start + cfg['val_data'][i]['seq_length_samples']
        x_val, y_val = x[:, frame_start:frame_end], y[:, frame_start:frame_end]

        d.update({
            f'val_{i+1}': [x_val, y_val]
        })


    if as_numpy:
        for k, v in d.items():
            d[k] = [v[0].numpy(), v[1].numpy()]

    d['fs'] = fs
    return d




