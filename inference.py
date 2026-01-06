import torch
import json
import os
from os.path import join
from train_modulation_fx import ModulationEffectTrainer
import torchaudio
import scipy
import argparse
from utils import estimate_f0, periodic_roll
import numpy as np
from scipy import interpolate


parser = argparse.ArgumentParser()
parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
parser.add_argument('--in_audio', type=str, default='')
parser.add_argument('--model_path', type=str, default='weights/BF-2-A/qa6tmyhk')


if __name__ == '__main__':
    args = parser.parse_args()

    base_path = args.model_path

    sd = torch.load(os.path.join(base_path, 'state_dict.pt'))
    with open(os.path.join(base_path, 'config.json'), 'r') as file:
        cfg = json.load(file)
    cfg['mlp']['activation'] = torch.nn.Tanh()      # TODO: fix this


    model = ModulationEffectTrainer(cfg_dict=cfg)
    model.load_state_dict(sd)


    if args.in_audio == '':
        in_filename = join(cfg['dataset_path'], cfg['pedal']['in_filename'])
        in_audio, fs = torchaudio.load(in_filename)
        start = int(cfg['val_data'][0]['start'] * fs)
        end = start + cfg['train_data']['seq_length_samples']
        x_trunc = in_audio[:, start:end]
        x = in_audio[:, start:end + cfg['train_data']['seq_length_samples']]
    else:
        in_filename = args.in_audio
        in_audio, fs = torchaudio.load(in_filename)
        x_trunc = in_audio[:, :cfg['train_data']['seq_length_samples']]
        x = in_audio

    # get control signal over original duration
    X = model.stft.stft(x_trunc).permute(0, 2, 1)[0, ...]
    num_frames = X.shape[0]
    num_samples = cfg['train_data']['seq_length_samples']

    m = torch.arange(X.shape[0])
    Y, control_sig = model.forward({'in_stft': X, 'frame_idx': m})

    if x.shape[-1] > num_samples:
        # resample control signal to audio rate
        t_frames = (0.5 + np.arange(0, num_frames)) / num_frames
        t_samples = np.arange(0, num_samples) / num_samples
        interpolator = interpolate.interp1d(t_frames, control_sig.detach().numpy().squeeze().T,
                                            kind='cubic', fill_value='extrapolate',
                                            bounds_error=False, axis=-1)
        control_sig = interpolator(t_samples)

        # estimate f0 and periodically extend to input audio length
        T0 = 1 / estimate_f0(control_sig, fs=1, buff=0)
        if control_sig.ndim == 0:
            control_sig = np.expand_dims(control_sig, 0)
        control_sig = periodic_roll(control_sig, T0=T0[0], roll=0, length=x.shape[-1])
        if control_sig.ndim == 1:
            control_sig = np.expand_dims(control_sig, -1)
        else:
            control_sig = control_sig.T

    # run inference
    if x.shape[0] == 1 or x.ndim == 1:
        y, control_sig_td = model.forward_inference(x, control_sig)
    else:
        y = np.zeros_like(x)
        for channel in range(x.shape[0]):
            y_channel, _ = model.forward_inference(x[channel, :], control_sig)
            y[channel, :] = y_channel

    out_filename = f'inference_{args.in_audio}_{os.path.split(base_path)[-1]}.wav'
    scipy.io.wavfile.write(out_filename, fs, y.T)
    print('Saved', out_filename)
