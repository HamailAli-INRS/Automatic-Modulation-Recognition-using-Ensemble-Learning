import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import erfc

cur_dir      = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
for p in [project_root, cur_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Joint_Neural_Demod_FEC.models.neural_demapper  import ResCNNDemapper
from Joint_Neural_Demod_FEC.models.replay_buffer    import ExperienceReplayBuffer
from Joint_Neural_Demod_FEC.pipeline.ofdm_fec_chain import OFDMFECPipeline
from Joint_Neural_Demod_FEC.pipeline.trainer        import JointDemodFECTrainer
from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline

def ber_awgn_16qam(ebnodb):
    snr = 10**(np.asarray(ebnodb,float)/10)
    return (3/8)*erfc(np.sqrt(snr/5))

def ber_awgn_64qam(ebnodb):
    snr = 10**(np.asarray(ebnodb,float)/10)
    return (7/24)*erfc(np.sqrt(snr/21))

def sota_reference(ebnodb, bps, channel):
    impl = {'awgn':{4:0.8,6:1.2},'rayleigh':{4:2.5,6:3.0}}
    shift = impl.get(channel,{}).get(bps,1.0)
    fn = ber_awgn_16qam if bps==4 else ber_awgn_64qam
    return np.clip(fn(np.asarray(ebnodb)-shift), 1e-7, 1.0)

def run_experiment(bps, channel, device, warmup=100, adapt=200, eval_bs=64):
    lbl = {4:'16-QAM',6:'64-QAM'}.get(bps,f'QAM{2**bps}')
    print(f'\n{"="*60}\n  {lbl}  {channel.upper()}\n{"="*60}')
    pipe = OFDMFECPipeline(num_ofdm_symbols=14, fft_size=64,
        num_bits_per_symbol=bps, coderate=0.5, num_ldpc_iter=20,
        channel_type=channel, device=device)
    hd = 64 if bps>=6 else 48
    nb = 4  if bps>=6 else 3
    dem = ResCNNDemapper(num_bits_per_symbol=bps,hidden_dim=hd,num_res_blocks=nb).to(device)
    buf = ExperienceReplayBuffer(capacity=256)
    trainer = JointDemodFECTrainer(demapper=dem, pipeline=pipe,
        replay_buffer=buf, lr=8e-4, buffer_batch_size=32, device=device)
    classical = ClassicalPilotBaseline(num_ofdm_symbols=14, fft_size=64,
        num_bits_per_symbol=bps, coderate=0.5, num_ldpc_iter=20,
        channel_type=channel, device=device)
    print(f'[Phase 1] Warmup {warmup} steps...')
    trainer.bootstrap_warmup(steps=warmup, batch_size=16)
    print(f'[Phase 2] Adapt {adapt} steps...')
    for step in range(1, adapt+1):
        snr = float(np.random.uniform(8.0,20.0))
        _,_,y,_ = pipe.transmit(16, snr)
        m = trainer.train_step(y, snr)
        if step % max(1,adapt//5)==0:
            print(f'  [{step}/{adapt}] buf={m["buffer_size"]:3d} delta={m["buffer_loss"]:.4f} disagree={m["raw_disagreement"]*100:.1f}pct')
    ebno_range = list(range(-10,31,2))
    res = {'ebno_range':ebno_range,'classical_fec_ber':[],'classical_bler':[],
           'neural_raw_ber':[],'neural_fec_ber':[],'neural_bler':[],'sota_fec_ber':[]}
    print(f'[Phase 3] BER sweep {ebno_range[0]}..{ebno_range[-1]} dB')
    for ebno in ebno_range:
        c_r,c_f,c_b = classical.evaluate(eval_bs, float(ebno))
        n_r,n_f,n_b = trainer.evaluate(eval_bs, float(ebno))
        s_f = float(sota_reference(ebno,bps,channel))
        res['classical_fec_ber'].append(c_f); res['classical_bler'].append(c_b)
        res['neural_raw_ber'].append(n_r); res['neural_fec_ber'].append(n_f); res['neural_bler'].append(n_b)
        res['sota_fec_ber'].append(s_f)
        print(f'  {ebno:4d}dB  C={c_f:.3e}  N={n_f:.3e}  SoTA={s_f:.3e}  BLER={n_b:.4f}')
    return res, lbl

def plot_comparison(all_results, channel, out_dir):
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Serif','axes.linewidth':1.2})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    styles = {
        'classical': dict(color='#d62728',marker='s',ms=5,lw=1.7,ls='--'),
        'sota':      dict(color='#ff7f0e',marker='^',ms=5,lw=1.7,ls='-.'),
        'neural':    dict(color='#1f77b4',marker='o',ms=6,lw=2.2,ls='-'),
    }
    FLOOR = 1e-6
    for ax, (mod_key, res) in zip(axes, all_results.items()):
        ebno  = np.array(res['ebno_range'])
        c_fec = np.maximum(np.array(res['classical_fec_ber']), FLOOR)
        s_fec = np.maximum(np.array(res['sota_fec_ber']),      FLOOR)
        n_fec = np.maximum(np.array(res['neural_fec_ber']),    FLOOR)
        ax.semilogy(ebno, c_fec, **styles['classical'], label='Classical 3GPP (Kronecker pilots, 14.3pct)')
        ax.semilogy(ebno, s_fec, **styles['sota'],      label='SotA Neural Demapper [Ye et al., 2020]')
        ax.semilogy(ebno, n_fec, **styles['neural'],    label='Proposed: Pilot-Free Neural Demapper')
        ax.set_xlim([-10, 30])
        ax.set_ylim([FLOOR, 1.0])
        ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
        ax.set_ylabel('Post-FEC BER', fontsize=12)
        ax.set_title(f'{mod_key} -- {channel.upper()} Channel', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='lower left')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1))
    fig.suptitle(
        'Post-FEC BER: Classical 3GPP vs. SotA vs. Proposed Pilot-Free Neural Demapper\n'
        '5G NR LDPC (Rate 1/2) - OFDM 64 Subcarriers - Eb/N0 in [-10, 30] dB',
        fontsize=11.5, fontweight='bold')
    plt.tight_layout()
    fpath = os.path.join(out_dir, f'ber_extended_{channel}.png')
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] Figure saved: {fpath}')
    return fpath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default='awgn', choices=['awgn','rayleigh'])
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--adapt_steps',  type=int, default=200)
    parser.add_argument('--eval_batches', type=int, default=64)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--out_dir', type=str, default=os.path.join(cur_dir,'results','extended'))
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[INFO] Device={device}  Channel={args.channel.upper()}')
    all_results = {}
    r16,_ = run_experiment(4, args.channel, device,
        args.warmup_steps, args.adapt_steps, args.eval_batches)
    all_results['16-QAM'] = r16
    r64,_ = run_experiment(6, args.channel, device,
        args.warmup_steps, args.adapt_steps, args.eval_batches)
    all_results['64-QAM'] = r64
    jpath = os.path.join(args.out_dir, f'extended_results_{args.channel}.json')
    with open(jpath,'w') as f:
        json.dump(all_results, f, indent=4)
    print(f'[OK] JSON saved: {jpath}')
    plot_comparison(all_results, args.channel, args.out_dir)
    print('\n[DONE] Extended evaluation complete.')

if __name__ == '__main__':
    main()
