import librosa
import matplotlib
import torch
import torch.nn.functional as F
from torch.signal import windows
from librosa import A_weighting
import numpy as np
from scipy.signal import kaiser_beta, firwin, freqz, find_peaks


def dropout_complex(x, p):
    if x.is_complex():
        mask = torch.nn.functional.dropout(torch.ones_like(x.real), p)
        return x * mask, mask
    else:
        return torch.nn.functional.dropout(x, p)


class STFT(torch.nn.Module):
    def __init__(self, n_win=2048, n_hop=None, n_fft=None, window='hann', synth_window=None):
        super().__init__()
        self.n_win = n_win
        self.n_hop = n_hop

        self.n_hop = n_win if n_hop is None else n_hop
        if n_fft is None:
            self.n_fft = n_win
        else:
            self.n_fft = n_fft

        if window is None:
            window = None
        elif window == 'hann':
            window = windows.hann(self.n_win, sym=False)

        if synth_window is None:
            synth_window = None
        elif synth_window == 'hann':
            synth_window = windows.hann(self.n_fft, sym=False)

        self.register_buffer('window', window)
        self.register_buffer('synth_window', synth_window)




    def framify(self, x):
        x = F.pad(x, (self.n_win - self.n_hop, self.n_win - self.n_hop), "constant", 0)
        L = x.shape[-1]
        B = x.shape[0]
        return F.unfold(x.view(B, 1, 1, L), kernel_size=(1, self.n_win), stride=(1, self.n_hop))


    def stft(self, x):
        x_frames = self.framify(x)
        if self.window is not None:
            x_frames *= self.window.view(self.n_win, 1)
        return torch.fft.rfft(x_frames, n=self.n_fft, dim=-2)


    def istft(self, x, length):
        y_frames = torch.fft.irfft(x, dim=-2)
        if self.synth_window is not None:
            y_frames *= self.synth_window.view(self.n_fft, 1)
        y_time_series = F.fold(y_frames,
                                        output_size=(1, length + self.n_fft - self.n_win),
                                        kernel_size=(1, self.n_fft),
                                        stride=(1, self.n_hop),
                                                 padding=(0, self.n_win - self.n_hop))
        return y_time_series[:, :, 0, :length]


class AWeight(torch.nn.Module):
    def __init__(self, sample_rate, n_fft):
        super().__init__()
        freqs = sample_rate * torch.arange(0, n_fft//2+1) / n_fft
        aweights = 10**(A_weighting(freqs)/20)
        self.register_buffer('aweights', torch.Tensor(aweights).view(-1, 1))

    def forward(self, X):
        return X * self.aweights

class LPF(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, pb_edge=12e3, sb_edge=16e3):
        super().__init__()
        As = 80
        delta_f = (sb_edge - pb_edge) / sample_rate
        fc = (sb_edge + pb_edge) / 2
        N = int(np.ceil((As - 7.95) / 14.36 / delta_f))
        N += (N % 2)
        h = firwin(N + 1, cutoff=fc, fs=sample_rate, window=('kaiser', kaiser_beta(As)))
        _, H = freqz(h, worN=n_fft//2+1)
        self.register_buffer('H', torch.from_numpy(H).view(-1, 1))

    def forward(self, X):
        return X * self.H

class FIRTaps(torch.nn.Module):
    def __init__(self, n_fft, taps=None):
        super().__init__()
        if taps is None:
            taps = np.array([1, -1])
        H = np.fft.rfft(taps, n_fft)
        self.register_buffer('H', torch.from_numpy(H))

    def forward(self, X):
        return X * self.H


def quadtratic_peak_interpolation(idx, x):
    alpha = x[idx-1]
    beta = x[idx]
    gamma = x[idx+1]

    loc = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)

    x_interp = beta - 0.25 * (alpha - gamma)
    return x_interp, loc



def estimate_f0(x, fs, buff):

    '''

    :param x: input signal of shape (C, T)
    :param fs: sample rate
    :param buff:
    :return:
    '''

    if x.ndim < 2:
        x = np.expand_dims(x, 0)

    if x.shape[0] > x.shape[1]:
        x = x.T

    n_channels, n_samples = x.shape

    D = np.abs(np.fft.rfft(windows.hann(n_samples) * (x - np.mean(x, axis=-1, keepdims=True))))
    freq = np.fft.rfftfreq(n_samples, d=1 / fs)

    # predict f0
    f0s = np.zeros(n_channels)
    for c in range(n_channels):
        pks, props = find_peaks(D[c, ...])

        if len(pks) == 0:
            continue
        else:
            idx = np.argmax(D[c, pks])
        f0_nearest = freq[pks[idx]]
        width_search = int(np.floor(fs / f0_nearest / 20))
        mag, frac_bin = quadtratic_peak_interpolation(pks[idx], D[c, ...])
        f0_est = (frac_bin + pks[idx]) * ((fs / 2) / (freq.shape[0] - 1))
        T0_est_samples = int(np.floor(fs/f0_est))
        search = x[c, T0_est_samples - width_search + buff:T0_est_samples + width_search + buff]
        if len(search) == 0:
            continue
        lowest_idx = np.argmin(np.abs(search - x[c, buff]))
        T0_samples_adjusted = T0_est_samples - width_search + lowest_idx + 1
        f0s[c] = fs / T0_samples_adjusted

    return f0s

def periodic_roll(x, T0: int, roll: int, length: int = -1, buffer=0):

    N = length if length > -1 else x.shape[-1]
    n_periods = int(np.floor(N / T0))
    x_period = x[..., buffer:n_periods*int(T0) + buffer]

    if torch.is_tensor(x):
        x_period = torch.roll(x_period, shifts=int(buffer), dims=-1)
        x_period = torch.roll(x_period, shifts=int(roll), dims=-1)
        y = torch.cat([x_period] * n_periods, dim=-1)
    else:
        x_period = np.roll(x_period, buffer, axis=-1)
        x_period = np.roll(x_period, roll, axis=-1)
        y = np.concatenate([x_period] * n_periods, axis=-1)
    return y[..., :N]

if __name__ == '__main__':
    model = FIRTaps(n_fft=4096)
    H = model.H
    import matplotlib.pyplot as plt
    matplotlib.use('macosx')
    plt.semilogx(20 * np.log10(np.abs(H)))

    A = AWeight(sample_rate=44100, n_fft=4096)
    plt.semilogx(20 * np.log10(np.abs(A.aweights)))
    plt.show()