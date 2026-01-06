
import torch
from torch.nn import Parameter, ParameterDict
from torch import tensor as T
from scipy import interpolate
from scipy.signal import lfilter, freqz
from flamo.processor import dsp, system
import numpy as np
import copy
from functools import partial
import dsp_filters
from utils import estimate_f0, periodic_roll
from torch import nn


class ModulationEffect(torch.nn.Module):
    def __init__(self,
                 n_fft,
                 seq_length,
                 effect_args=None,
                 svf_args=None,
                 mlp_args=None,
                 n_channels: int = 1,
                 phaser=False
                 ):
        super().__init__()

        self.phaser = phaser
        if self.phaser:
            self.phasor = APF(n_fft=n_fft)
            self.models = torch.nn.ModuleList([Phaser(svf_args=svf_args, **effect_args) for i in range(n_channels)])
            self.activation = lambda a: torch.tanh(torch.pi * (a + 0.5))
        else:
            self.phasor = Delay(n_fft=n_fft)
            self.models = torch.nn.ModuleList([Chorus(svf_args=svf_args, **effect_args) for i in range(n_channels)])
            self.activation = lambda a: 0.5 * (1 - torch.cos(0.1 * torch.pi * a))

        self.n_channels = n_channels
        self.n_fft = n_fft
        n_frames = seq_length // n_fft
        self.emb = torch.nn.Embedding(num_embeddings=n_frames,
                                      embedding_dim=n_channels)
        self.mlp = BaseMLP(**mlp_args)


    def forward(self, x, n, roll: int = 0):
        '''

        :param x: input STFT frames
        :param n: frame indices
        :return:
        '''

        control_sig = self.get_control_signal(n, roll=roll)
        phasor_frames = self.phasor(control_sig)

        out = torch.zeros_like(x)
        for i, model in enumerate(self.models):
            out = out + model(x, phasor_frames[:, i, :])

        if not self.phaser:
            control_sig = self.n_fft / 2 * control_sig

        return out, control_sig

    def get_control_signal(self, n, roll):
        c = self.emb(n) / 2 / torch.pi
        c = self.activation(self.mlp(c))

        if roll > 0:
            T0 = 1 / estimate_f0(c.detach().cpu().numpy(), fs=1, buff=0)
            T0 = np.clip(T0, a_min=1, a_max=c.shape[0])[0]
            c = periodic_roll(c.T, T0=T0, roll=roll).T

        return c



    def forward_inference(self, x, control_sig):
        y = np.zeros_like(x.detach().cpu().numpy())
        for i, model in enumerate(self.models):
            out, delay_fs = model.inference(x, control_sig[..., i])
            y += out
        return y, control_sig


class DelayBasedAudioEffect(torch.nn.Module):

    def __init__(self,
                 res=None,
                 mix=None,
                 thru=None,
                 svf_args=None,
                 use_filters=True,
                 option=1):

        super().__init__()

        if mix is None:
            mix = 0.5 * torch.rand(1) + 0.25
        else:
            mix = T([mix])

        self.params = ParameterDict({
                'feedback': Parameter(torch.randn(1)/10) if res is None else Parameter(T([res])),
                'mix':      Parameter(mix),
                'thru': Parameter(0.5 * torch.rand(1) + 0.25) if thru is None else Parameter(T([thru])),
        })

        self.use_filters = use_filters
        if self.use_filters:
            self.iir1 = FlamoSVF(svf_args)
            self.iir2 = FlamoSVF(svf_args)
        self.option = option

    def forward(self, X, z_D):

        if self.use_filters:
            z_D_num = self.iir2(z_D)
            z_D_denom = z_D_num if self.option == 2 else z_D

            if self.option == 0:
                z_D_num, z_D_denom = z_D, z_D
        else:
            z_D_num, z_D_denom = z_D, z_D

        Y = X * (self.params['thru'] + self.params['mix'] * z_D_num) / (1 - self.map_feedback(self.params['feedback']) * z_D_denom)

        if self.use_filters:
            Y = self.iir1(Y)

        return Y

    def map_feedback(self, x):
        return torch.tanh(x)


class Chorus(DelayBasedAudioEffect):
    def __init__(self, res=None,
                 mix=None,
                 thru=None,
                 svf_args=None,
                 use_filters=True,
                 option=1):
        super().__init__(res, mix, thru, svf_args, use_filters, option)


    def inference(self, x, delays):

        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
        feedback = self.map_feedback(self.params['feedback']).cpu().detach().numpy().squeeze()
        mix = self.params['mix'].cpu().detach().numpy().squeeze()
        thru = self.params['thru'].cpu().detach().numpy().squeeze()

        num_samples = x.shape[-1]
        num_frames = delays.shape[-1]
        hop_size = num_samples // num_frames
        if torch.is_tensor(delays):
            delays = delays.cpu().detach().numpy()
        delays = np.clip(delays, a_min=0, a_max=None)
        assert(hop_size >= 1)
        if hop_size > 1:
            t_frames = (0.5 + np.arange(0, num_frames)) / num_frames
            interpolator = interpolate.interp1d(t_frames, delays,
                                                kind='cubic',
                                                fill_value='extrapolate',
                                                bounds_error=False,
                                                axis=-1)
            t_samples = np.arange(0, num_samples) / num_samples
            delays = interpolator(t_samples)

        b = np.stack([thru, mix])

        if self.use_filters:
            bq_b, bq_a = self.iir2.get_poly_coeff()
            if self.option == 2:
                out = dsp_filters.time_varying_bq2_comb(x=x.squeeze(),
                                           delay=delays.squeeze(),
                                           b=b, a=np.array([feedback]),
                                           bq_b=bq_b, bq_a=bq_a)
            elif self.option == 1:
                svf_params = self.iir2.get_svf_params('numpy')
                out = dsp_filters.time_varying_svf_comb(x.squeeze(), delays.squeeze(), b=b, a=feedback, svf_params=svf_params)
                # equiv:
                # out = time_varying_bq_comb(x=x.squeeze(),
                #                            delay=delays.squeeze(),
                #                            b=b, a=np.array([feedback]),
                #                            bq_b=bq_b, bq_a=bq_a)
        else:
            out = dsp_filters.time_varying_comb(x.squeeze(), delays.squeeze(), b=b, a=feedback)

        if self.use_filters:
            svf_params = self.iir1.get_svf_params('numpy')
            out = dsp_filters.svf_filter(svf_params, out)

        return out, delays


class Phaser(DelayBasedAudioEffect):

    def __init__(self, res=None,
                 mix=None,
                 thru=None,
                 svf_args=None,
                 use_filters=True,
                 num_allpass=4,
                 option=1):

        super().__init__(res, mix, thru, svf_args, use_filters)

        if mix is None:
            mix = 0.5 * torch.rand(1) + 0.25
        else:
            mix = T([mix])

        self.params = ParameterDict({
                'feedback': Parameter(torch.randn(1)/10) if res is None else Parameter(T([res])),
                'mix':      Parameter(mix),
                'thru': Parameter(0.5 * torch.rand(1) + 0.25) if thru is None else Parameter(T([thru])),
        })

        self.use_filters = use_filters
        if self.use_filters:
            self.iir1 = FlamoSVF(svf_args)
            self.iir2 = FlamoSVF(svf_args)
        self.num_allpass = num_allpass

        assert(option == 0 or option == 1 or option == 2)
        self.option = option


    def forward(self, X, z_D):
        z_D = torch.pow(z_D, self.num_allpass)
        return super().forward(X, z_D)


    def inference(self, x, poles):

        if torch.is_tensor(x):
            x = x.cpu().detach().numpy()
        feedback = self.map_feedback(self.params['feedback']).cpu().detach().numpy().squeeze()
        mix = self.params['mix'].cpu().detach().numpy().squeeze()
        thru = self.params['thru'].cpu().detach().numpy().squeeze()

        if torch.is_tensor(poles):
            poles = poles.cpu().detach().numpy()
        poles = np.clip(poles, a_min=0, a_max=None)
        num_samples = x.shape[-1]
        num_frames = poles.shape[-1]
        hop_size = num_samples // num_frames
        assert(hop_size >= 1)
        if hop_size > 1:
            t_frames = (0.5 + np.arange(0, num_frames)) / num_frames
            interpolator = interpolate.interp1d(t_frames, poles,
                                                kind='cubic',
                                                fill_value='extrapolate',
                                                bounds_error=False,
                                                axis=-1)
            t_samples = np.arange(0, num_samples) / num_samples
            poles = interpolator(t_samples)

        eps = 1.0 - 1e-4
        poles = np.clip(poles, a_min=-eps, a_max=eps)

        b = np.stack([thru, mix])
        if self.use_filters:
            bq_b, bq_a = self.iir2.get_poly_coeff()
            if self.option == 2:
                out = dsp_filters.phaser_with_feedback_and_bq2(x=x.squeeze(),
                                              pole=poles,
                                              b=b, a=np.array([feedback]),
                                              bq_b=bq_b, bq_a=bq_a,
                                              K=self.num_allpass)
            elif self.option == 1:
                out = dsp_filters.phaser_with_feedback_and_bq1(x=x.squeeze(),
                                              pole=poles,
                                              b=b, a=np.array([feedback]),
                                              bq_b=bq_b, bq_a=bq_a,
                                              K=self.num_allpass)
        else:
            out = dsp_filters.phaser_with_feedback(x=x.squeeze(),
                                          pole=poles,
                                          b=b, a=np.array([feedback]),
                                          K=self.num_allpass, gamma=0.0)

        if self.use_filters:
            svf_params = self.iir1.get_svf_params('numpy')
            out = dsp_filters.svf_filter(svf_params, out)

        return out, poles


class Delay(torch.nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        bins = torch.arange(0, n_fft // 2 + 1).view(1, 1, -1)
        self.register_buffer('bins', bins)

    def forward(self, delay, normalised=True, gamma=None):
        if normalised:
            angle = - torch.pi * delay
        else:
            angle = -torch.pi * delay / (self.n_fft//2)

        r = torch.ones_like(delay, device=delay.device)
        if gamma is not None:
            r = r * gamma
        mod_sig = torch.polar(r, angle).unsqueeze(-1)
        phasor = mod_sig ** self.bins
        return phasor


class APF(torch.nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        bins = torch.arange(0, n_fft // 2 + 1).view(1, 1, -1)
        self.register_buffer('bins', bins)
        z_inv = torch.exp(-1j * torch.pi * bins/(self.n_fft // 2))
        self.register_buffer('z_inv', z_inv)

    def forward(self, p):
        H = (p - self.z_inv) / (1 - p * self.z_inv)
        return H.permute(1, 0, 2)


class BaseMLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: list[int],
            out_dim: int,
            activation: nn.Module,
            bias: bool = True,
            factor: float = 1.0,
            init_name: str = "xavier_uniform_",
            **init_kwargs
        ):
        super().__init__()

        self._model = nn.Sequential()
        for n in hidden_dim:
            self._model.append(nn.Linear(in_features=in_dim, out_features=n, bias=bias))
            self._model.append(copy.deepcopy(activation))
            in_dim = n
        self._model.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias))

        self._factor = factor
        self._init_func = self._get_init_func(activation, init_name, **init_kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self._init_func(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _get_init_func(activation: nn.Module, name: str, **kwargs):
        if "xavier" in name:
            if type(activation) is nn.ReLU:
                kwargs.update({"gain": nn.init.calculate_gain("relu")})
            elif type(activation) is nn.LeakyReLU:
                kwargs.update({"gain": nn.init.calculate_gain("leaky_relu", param=activation.negative_slope)})
            elif type(activation) is nn.Tanh:
                kwargs.update({"gain": nn.init.calculate_gain("tanh")})
            elif type(activation) is nn.Sigmoid:
                kwargs.update({"gain": nn.init.calculate_gain("sigmoid")})
        elif "kaiming" in name:
            if type(activation) is nn.ReLU:
                kwargs.update({"nonlinearity": "relu"})
            elif type(activation) is nn.LeakyReLU:
                kwargs.update({"nonlinearity": "leaky_relu", "a": activation.negative_slope})

        func = getattr(nn.init, name)
        func = partial(func, **kwargs)

        return func

    def forward(self, x: torch.Tensor):
        return self._model(self._factor * x)


class FlamoSVF(torch.nn.Module):
    def __init__(self, svf_args):
        super().__init__()
        svf = dsp.SVF(**svf_args)
        svf = self.override_init(svf)
        self.filter = system.Shell(core=svf,
                                   input_layer=dsp.Transform(transform=lambda x: x.unsqueeze(2)),
                                   output_layer=dsp.Transform(transform=lambda x: x.squeeze(2)))

    def forward(self, x):
        return self.filter(x)

    def override_init(self, svf):
        # override inititlisation
        with torch.no_grad():
            svf.param[2:] *= 0.0
            svf.param[2:] += 0.1 * torch.rand_like(svf.param[2:])
            svf.param[0] -= torch.pi
        return svf

    def get_freq_response(self, n_bins, fs):
        b, a = self.get_poly_coeff()
        w, h = freqz(b, a, worN=n_bins, fs=fs)
        return w, h

    def get_poly_coeff(self, device='cpu'):
        r"""
        Computes the polynomial coefficients for the SVF filter
        """
        filter = self.filter.get_core()
        param = filter.map_param2svf(filter.param)
        if device == 'cpu':
            param = torch.tensor(param, device=device)

        f, R, mLP, mBP, mHP = param
        b = np.zeros((3, *f.shape))
        a = np.zeros((3, *f.shape))

        b[0] = (f ** 2) * mLP + f * mBP + mHP
        b[1] = 2 * (f ** 2) * mLP - 2 * mHP
        b[2] = (f ** 2) * mLP - f * mBP + mHP

        a[0] = (f ** 2) + 2 * R * f + 1
        a[1] = 2 * (f ** 2) - 2
        a[2] = (f ** 2) - 2 * R * f + 1

        return b.squeeze(), a.squeeze()

    def get_svf_params(self, format='dict'):
        filter = self.filter.get_core()
        param = filter.map_param2svf(filter.param)

        if format == 'numpy':
            d = torch.concatenate(param).cpu().squeeze().detach().numpy()
        else:
            d = {}
            names = 'f', 'R', 'mLP', 'mBP', 'mHP'
            for key, value in zip(names, param):
                d[key] = value.cpu().detach().numpy().squeeze()
        return d


