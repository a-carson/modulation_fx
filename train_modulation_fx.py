import time
import torch
import pytorch_lightning as pl
import torchaudio
from scipy.signal import windows
from dataset import ModulationDataset, load_train_signal
from torch.utils.data import DataLoader
from model import ModulationEffect
import wandb
from utils import STFT, FIRTaps, dropout_complex
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join
import pandas as pd
from auraloss.freq import MultiResolutionSTFTLoss


class ModulationEffectTrainer(pl.LightningModule):
    def __init__(self,
                 cfg_dict=None,
                 logger=None,
                 device=None,
                 sample_rate: int = 44100):
        super().__init__()
        self.cfg = cfg_dict

        svf_args = dict(size=(1, 1),
                        n_sections=1,
                        nfft=self.cfg['stft']['n_fft'],
                        fs=sample_rate,
                        requires_grad=True,
                        alias_decay_db=0,
                        device=device)

        is_phaser = 'phaser' in self.cfg and self.cfg['phaser']

        self.model = ModulationEffect(n_fft=self.cfg['stft']['n_fft'],
                                        seq_length=self.cfg['train_data']['seq_length_samples'],
                                        mlp_args=self.cfg['mlp'],
                                        svf_args=svf_args,
                                        effect_args=self.cfg['model'],
                                        phaser=is_phaser,
                                        n_channels=self.cfg['n_channels'])

        self.sample_rate = sample_rate
        self.lr = self.cfg['lr']
        self.automatic_optimization = False
        self.stft = STFT(**self.cfg['stft'])
        self.mrsl = MultiResolutionSTFTLoss()


        if self.cfg['pedal']['in_shape'] in ['tri', 'rect', 'hann']:
            self.cfg['loss_weights']['pre_emp'] = ''

        if self.cfg['loss_weights']['pre_emp'] != '':
            n_filt = self.cfg['loss_weights']['n_filt']
            if n_filt == 0:
                n_filt = self.cfg['stft']['n_win'] // self.cfg['pedal']['pulse_frac']
            if self.cfg['loss_weights']['pre_emp'] == 'hann':
                window = windows.hann(n_filt, sym=False)
            elif self.cfg['loss_weights']['pre_emp'] == 'tri':
                window = windows.bartlett(n_filt, sym=False)
            else:
                window = windows.boxcar(n_filt, sym=False)
            self.pre_emf = FIRTaps(n_fft=self.cfg['stft']['n_fft'], taps=window)
        else:
            self.pre_emf = lambda x: x

        self.val_loss_hist = []





        if logger is None:
            class DummyLogger():
                dir = None
                def log(self, to_log):
                    return
            self.run = DummyLogger()
        else:
            self.run = logger

    def loss_function(self, Y, Y_hat, filter=True):

        if filter:
            Y = self.pre_emf(Y)
            Y_hat = self.pre_emf(Y_hat)

        loss = {}
        loss['phase'] = torch.real(torch.mean((Y - Y_hat) * torch.conj(Y - Y_hat)) / torch.mean(Y * torch.conj(Y)))

        E = Y - Y_hat
        max_err, _ = torch.max(E.abs(), dim=-1)
        loss['l_inf'] = max_err

        eps = 1e-7


        if 'frame' in self.cfg['loss_weights'] or 'log_difference' in self.cfg['loss_weights']:
            y_frames = torch.fft.irfft(Y, dim=-1)
            y_hat_frames = torch.fft.irfft(Y_hat, dim=-1)
            y_frames = y_frames[:, :self.cfg['stft']['n_win']]
            y_hat_frames = y_hat_frames[:, :self.cfg['stft']['n_win']]
            win = torch.hann_window(y_frames.shape[-1], device=Y.device).unsqueeze(0)

            y_frames *= win
            y_hat_frames *= win

            loss['frame'] = torch.mean((y_frames - y_hat_frames)**2) / torch.mean(y_frames**2)

            log_y = torch.log10(y_frames.abs() + eps)
            log_y_hat = torch.log10(y_hat_frames.abs() + eps)
            loss['log_difference'] = torch.norm(log_y - log_y_hat, p=1) \
                                     / torch.norm(log_y, p=1)

        Y = torch.abs(Y)
        Y_hat = torch.abs(Y_hat)
        loss['convergence'] = torch.norm(Y - Y_hat) / (torch.norm(Y) + eps)
        loss['magnitude'] = torch.norm(torch.log10(Y + eps) - torch.log10(Y_hat + eps), p=1) / torch.numel(Y)

        C = torch.fft.irfft(torch.log10(Y + eps), dim=-1).abs()
        C_hat = torch.fft.irfft(torch.log10(Y_hat + eps), dim=-1).abs()
        loss['ceps_convergence'] = torch.norm(C - C_hat) / (torch.norm(C) + eps)
        loss['ceps_magnitude'] = torch.norm(torch.log10(C + eps) - torch.log10(C_hat + eps), p=1) / torch.numel(C)

        Y_pow = torch.mean(Y**2, dim=0)
        Y_hat_pow = torch.mean(Y_hat**2, dim=0)
        loss['lsd'] = torch.norm(torch.log10(Y_pow + eps) - torch.log10(Y_hat_pow + eps)) / torch.norm(torch.log10(Y_pow + 1e-9))

        return loss

    def forward(self, batch, roll=0):
        return self.model.forward(x=batch['in_stft'], n=batch['frame_idx'], roll=roll)

    def forward_inference(self, x, delay):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        y = np.zeros_like(x)

        for c in range(x.shape[0]):
            out_channel, delay_samples = self.model.forward_inference(x[c, :], delay)
            y[c, :] = out_channel
        return y.squeeze(), delay_samples

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        opt.zero_grad()

        Y_hat, _ = self.forward(batch)
        Y_hat, mask = dropout_complex(Y_hat, self.cfg['dropout'])
        Y = mask * batch['out_stft']
        loss_dict = self.loss_function(Y, Y_hat)
        loss = torch.zeros(1, device=Y.device)
        metrics = {}
        for key, value in loss_dict.items():
            loss += self.cfg['loss_weights'][key] * value.mean()
            metrics.update({'train/'+ key: value.mean().detach()})

        self.manual_backward(loss)

        for i, model in enumerate(self.model.models):
            for key, value in model.params.items():
                metrics.update({f'params_{i}/' + key: value[0].detach()})


        if self.model.models[0].use_filters:
            p = self.model.models[0].iir1.get_svf_params()
            for key, value in p.items():
                for i, model in enumerate(self.model.models):
                    metrics.update({f'params_{i}/' + 'iir1_' + key: value})

            p = self.model.models[0].iir2.get_svf_params()
            for key, value in p.items():
                for i, model in enumerate(self.model.models):
                    metrics.update({f'params_{i}/' + 'iir2_' + key: value})

        self.run.log(metrics)

        opt.step()

        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx, test_step=False):

        if dataloader_idx > 0:
            Y = batch['out_stft']
            losses = []
            for shift in batch['frame_idx']:
                with torch.no_grad():
                    Y_hat, delay = self.forward(batch, int(shift))
                loss = self.loss_function(Y, Y_hat, filter=False)
                for k, v in loss.items():
                    loss[k] = v.mean().squeeze().cpu().detach().numpy()
                loss.update({'roll': int(shift)})
                losses.append(loss)
            loss_df = pd.DataFrame.from_dict(losses)
            best_delay = np.argmin(loss_df['frame'].values)
            self.run.log({f'best_delay_{dataloader_idx}': best_delay})
            roll = best_delay
        else:
            roll = 0

        log_key = 'val' if test_step is False else 'test'

        x_frames, y_frames = batch['in'], batch['out']
        x = torch.flatten(x_frames)
        y = torch.flatten(y_frames)

        Y = batch['out_stft']
        Y_hat, delay = self.forward(batch, roll=roll)

        loss_frames = torch.real(torch.mean((Y - Y_hat) * torch.conj(Y - Y_hat), dim=-1) / torch.mean(Y * torch.conj(Y), dim=-1))
        y_hat = self.stft.istft(Y_hat.T.unsqueeze(0), length=y.shape[-1])

        loss_dict = self.loss_function(Y, Y_hat)
        metrics = {}
        for key, value in loss_dict.items():
            metrics.update({f'{log_key}_{dataloader_idx}_filtered/'+ key: value.mean().detach()})
        loss_dict = self.loss_function(Y, Y_hat, filter=False)
        for key, value in loss_dict.items():
            metrics.update({f'{log_key}_{dataloader_idx}_raw/'+ key: value.mean().detach()})

        y_td, delay_fs = self.forward_inference(x, delay)

        esr_td = np.mean((y_td - y.detach().cpu().numpy()) ** 2) / np.mean(y.detach().cpu().numpy()**2)
        mrsl = self.mrsl(x=torch.tensor(y_td, device=y.device).view(1, 1, -1), y=y.view(1, 1, -1)).cpu().detach().numpy()
        metrics.update({f'{log_key}_{dataloader_idx}_td/esr': esr_td})
        metrics.update({f'{log_key}_{dataloader_idx}_td/mrsl': mrsl})

        if (self.current_epoch + 1) % self.cfg['log_audio_every_n_epoch'] == 0:
            self.log_audio(f'model_freq_{dataloader_idx}', torch.flatten(y_hat))
            self.log_audio(f'model_td_{dataloader_idx}', torch.flatten(torch.from_numpy(y_td)))


        if dataloader_idx == 0:

            if self.model.models[0].use_filters:
                for m, model in enumerate(self.model.models):
                    f, h1 = model.iir1.get_freq_response(n_bins=10000, fs=self.sample_rate)
                    _, h2 = model.iir2.get_freq_response(n_bins=10000, fs=self.sample_rate)
                    h1 *= model.params['thru'].detach().cpu().numpy().squeeze()
                    h2 *= model.params['mix'].detach().cpu().numpy().squeeze() / model.params['thru'].detach().cpu().numpy().squeeze()
                    plt.semilogx(f, 20 * np.log10(np.abs(h1)), label='out')
                    plt.semilogx(f, 20 * np.log10(np.abs(h2)), label='delayed')
                    plt.xlabel('Freq [Hz]')
                    plt.ylabel('Mag. [dB]')
                    plt.xlim([20, 20e3])
                    plt.legend()
                    self.run.log({f"iir_{m}": plt})

            delay = delay.squeeze().detach().cpu().numpy()

            if 'delay_samples' in batch:
                plt.plot(torch.flatten(batch['delay_samples']).detach().cpu(), label='target')
            plt.plot(delay, label='predicted')
            plt.legend()
            #plt.show()
            self.run.log({"delays": plt})

            plt.plot(loss_frames.squeeze().detach().cpu())
            self.run.log({"loss_frames": plt})

            fig, ax = plt.subplots()
            x_frame = batch['in'][-1, ...].detach().cpu().numpy()
            y_frame = batch['out'][-1, ...].detach().cpu().numpy()
            y_hat_frame = np.fft.irfft(Y_hat[-1, ...].detach().cpu().numpy())
            ax.plot(x_frame, label='input')
            ax.plot(y_frame, label='target')
            ax.plot(y_hat_frame, label='pred')
            plt.legend()
            #plt.show()
            self.run.log({"waveform": plt})

            fig, ax = plt.subplots()
            X_filt = self.pre_emf(batch['in_stft'])
            Y_filt = self.pre_emf(batch['out_stft'])
            Y_hat_filt = self.pre_emf(Y_hat)
            ax.plot(np.fft.irfft(X_filt[-1, ...].detach().cpu().numpy()), label='input')
            ax.plot(np.fft.irfft(Y_filt[-1, ...].detach().cpu().numpy()), label='target')
            ax.plot(np.fft.irfft(Y_hat_filt[-1, ...].detach().cpu().numpy()), label='pred')
            plt.legend()
            #plt.show()
            self.run.log({"waveform_lpf": plt})

        if dataloader_idx == 1:
            delay = delay.squeeze().detach().cpu().numpy()
            plt.plot(delay, label='validation_1')
            self.run.log({"delays_1": plt})


        if dataloader_idx == 0:
            self.log(f'{log_key}_loss', 10 * torch.log10(loss_dict['phase']), add_dataloader_idx=False)
            self.log(f'{log_key}_loss_td', 10 * np.log10(esr_td), add_dataloader_idx=False)

            if self.run.dir is not None:
                self.run.log_model(self.run.dir + '/lightning_logs/')

        if (self.current_epoch == 0) and (self.cfg['seed'] == 0):
            self.log_audio(f'target_td_{dataloader_idx}', y)

        self.run.log(metrics)

        return

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx, test_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_audio(self, caption, audio):
        self.run.log(
            {'Audio/' + caption: wandb.Audio(audio.cpu().detach().numpy(), caption=caption,
                                             sample_rate=self.sample_rate),
             'epoch': self.current_epoch})


def train_delay_based_audio_fx(cfg):

    if cfg['seed'] > -1:
        seed = cfg['seed']
    else:
        seed = int(time.time())

    print('Manual seed: ', seed)
    pl.seed_everything(seed)

    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

    if cfg['wandb']:
        run = wandb.init(project=cfg['project'],
                         config=cfg,
                         group=cfg['group'])
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("plot/*", step_metric="epoch")
        run.define_metric("spectra/*", step_metric="epoch")
        run.define_metric("params/*", step_metric="epoch")

    else:
        class DummyLogger():
            dir = None
            def log(self, to_log):
                return

            def finish(self):
                return

        run = DummyLogger()


    in_audio, fs = torchaudio.load(join(cfg['dataset_path'], cfg['pedal']['in_filename']))
    target_audio, fs_ = torchaudio.load(join(cfg['dataset_path'], cfg['pedal']['out_filename']))
    assert(fs == fs)

    train_sig_cfg = load_train_signal(join(cfg['dataset_path'], cfg['pedal']['in_sig_cfg']),
                                cfg['stft']['n_win'],
                                cfg['pedal']['in_shape'],
                                cfg['pedal']['pulse_frac'])
    train_data_end = train_sig_cfg['timestamp_end']
    train_data_start = train_data_end - cfg['train_data']['seq_length_samples']

    train_loader = DataLoader(
        ModulationDataset(input=in_audio[:, train_data_start:train_data_end],
                          target=target_audio[:, train_data_start:train_data_end],
                          sequence_length=cfg['train_data']['seq_length_samples'],
                          stft_args=cfg['stft']),
        batch_size=cfg['train_data']['batch_size'],
        shuffle=True)

    val_ds = ModulationDataset(input=in_audio[:, train_data_start:train_data_end],
                               target=target_audio[:, train_data_start:train_data_end],
                               sequence_length=cfg['train_data']['seq_length_samples'],
                               stft_args=cfg['stft'])
    val_loader = [DataLoader(val_ds, batch_size=len(val_ds))]
    for val_cfg in cfg['val_data']:
        val_data_start = int(val_cfg['start'] * fs)
        val_data_end = val_data_start + val_cfg['seq_length_samples']
        val_ds = ModulationDataset(input=in_audio[:, val_data_start:val_data_end],
                                   target=target_audio[:, val_data_start:val_data_end],
                                   sequence_length=val_cfg['seq_length_samples'],
                                   stft_args=cfg['stft'])
        val_loader.append(DataLoader(val_ds, batch_size=len(val_ds)))



    best_ckpt = ModelCheckpoint(
        filename="best-{val_loss:.1f}dB", monitor="val_loss", save_top_k=1,  mode="min"
    )
    best_td_ckpt = ModelCheckpoint(
        filename="best-{val_loss_td:.1f}dB", monitor="val_loss_td", save_top_k=1, mode="min"
    )
    last_ckpt = ModelCheckpoint(filename="latest-{val_loss:.1f}dB")

    trainer = pl.Trainer(**cfg['trainer'], default_root_dir=run.dir, callbacks=[best_ckpt, best_td_ckpt, last_ckpt])
    if cfg['ckpt'] != '':
        pl_model = ModulationEffectTrainer.load_from_checkpoint(cfg['ckpt'])
    else:
        pl_model = ModulationEffectTrainer(cfg_dict=cfg,
                                          logger=run,
                                          device=device,
                                          sample_rate=fs)
    print(cfg)
    trainer.validate(model=pl_model, dataloaders=val_loader)
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model=pl_model, dataloaders=val_loader)
    trainer.test(model=pl_model, dataloaders=val_loader, ckpt_path='best')

    run.finish()
    return
