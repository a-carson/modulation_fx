import argparse
from config import get_config
from train_modulation_fx import train_delay_based_audio_fx
from copy import deepcopy
import multiprocessing as mp
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
parser.add_argument("--config", type=int, default=0)
parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_runs", type=int, default=1)
parser.add_argument("--group", type=str, default=datetime.now().strftime("%y%m%d_%H%M%S"))
parser.add_argument("--shape", type=str, default='')
parser.add_argument("--pulse_frac", type=int, default=2)
parser.add_argument("--n_fft", type=int, default=0)
parser.add_argument("--n_channels", type=int, default=1)
parser.add_argument("--feedback_option", type=int, default=1)

parser.add_argument("--pre_emp", type=str, default='')



if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args.config, args.n_channels)

    cfg.update({'seed': args.seed,
                'wandb': args.wandb,
                'ckpt': args.ckpt,
                'group': args.group,
                'config': args.config})

    if args.shape != '':
        cfg['pedal']['in_shape'] = args.shape

    if args.pre_emp != '':
        cfg['loss_weights']['pre_emp'] = args.pre_emp


    cfg['pedal']['pulse_frac'] = args.pulse_frac

    if args.n_fft > 0:
        cfg['stft']['n_fft'] = args.n_fft
        cfg['stft']['n_hop'] = args.n_fft
        cfg['stft']['n_win'] = args.n_fft
        new_batch_size = int(cfg['train_data']['batch_size'] * 2048 // args.n_fft)
        cfg['train_data']['batch_size'] = new_batch_size

    cfg['model']['option'] = args.feedback_option


    if args.n_runs == 1:
        train_delay_based_audio_fx(cfg)
    else:
        cfg_list = []
        for i in range(args.n_runs):
            inst = deepcopy(cfg)
            inst.update({'seed': i + args.seed})
            cfg_list.append(inst)

        pool = mp.Pool(mp.cpu_count())
        pool.map(train_delay_based_audio_fx, cfg_list)
        pool.close()
        pool.join()



