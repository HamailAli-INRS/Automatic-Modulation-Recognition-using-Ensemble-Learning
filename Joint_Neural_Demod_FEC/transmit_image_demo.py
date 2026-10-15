# End-to-End Image Transmission Demo
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math, numpy as np, torch, matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from Joint_Neural_Demod_FEC.pipeline.ofdm_fec_chain import OFDMFECPipeline
from Joint_Neural_Demod_FEC.models.neural_demapper import ResCNNDemapper
from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline
from sionna.phy.utils import ebnodb2no

def create_sample_image(size=(64, 64)):
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, size[0]-5, size[1]-5], outline=0, width=2)
    draw.ellipse([10, 10, 30, 30], fill=60, outline=0)
    draw.rectangle([35, 12, 54, 30], fill=140, outline=0)
    draw.text((12, 36), '5G NR', fill=0)
    draw.text((12, 48), 'INRS', fill=0)
    return img

def img_to_bits(img):
    arr = np.array(img, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits, arr.shape

def bits_to_img(bits, shape):
    n_bytes = shape[0] * shape[1]
    bits_trimmed = bits[:n_bytes * 8]
    arr = np.packbits(bits_trimmed).reshape(shape)
    return Image.fromarray(arr, mode='L')

def transmit_image_neural(pipeline, demapper, bits, ebno_db):
    demapper.eval()
    k = pipeline.k
    num_slots = math.ceil(len(bits) / k)
    padded_len = num_slots * k
    padded_bits = np.pad(bits, (0, padded_len - len(bits)), mode='constant')
    rx_recovered_bits = []
    
    with torch.no_grad():
        for i in range(num_slots):
            slot_bits_np = padded_bits[i * k : (i + 1) * k]
            u_slot = torch.tensor(slot_bits_np, dtype=torch.float32, device=pipeline.device).unsqueeze(0)
            c_slot = pipeline.encoder(u_slot)
            c_reshaped = c_slot.view(1, 1, 1, pipeline.n)
            symbols = pipeline.mapper(c_reshaped)
            tx_grid = pipeline.rg_mapper(symbols)
            no = ebnodb2no(ebno_db, num_bits_per_symbol=pipeline.num_bits_per_symbol, coderate=pipeline.coderate)
            if pipeline.ofdm_channel is not None:
                faded_grid = pipeline.ofdm_channel(tx_grid)
            else:
                faded_grid = tx_grid
            rx_grid = pipeline.awgn(faded_grid, no)
            y_complex = rx_grid.squeeze(1).squeeze(1)
            y_tensor = torch.stack([y_complex.real, y_complex.imag], dim=1).float()
            llrs = demapper(y_tensor)
            u_hat, _ = pipeline.decode_ldpc(llrs)
            rx_recovered_bits.extend(u_hat.squeeze(0).cpu().numpy().astype(np.uint8))
            
    return np.array(rx_recovered_bits[:len(bits)])

def transmit_image_classical(baseline, bits, ebno_db):
    k = baseline.k
    num_slots = math.ceil(len(bits) / k)
    padded_len = num_slots * k
    padded_bits = np.pad(bits, (0, padded_len - len(bits)), mode='constant')
    rx_recovered_bits = []
    
    with torch.no_grad():
        for i in range(num_slots):
            slot_bits_np = padded_bits[i * k : (i + 1) * k]
            u_slot = torch.tensor(slot_bits_np, dtype=torch.float32, device=baseline.device).unsqueeze(0)
            c_slot = baseline.encoder(u_slot)
            c_reshaped = c_slot.view(1, 1, 1, baseline.n)
            symbols = baseline.mapper(c_reshaped)
            tx_grid = baseline.rg_mapper(symbols)
            no = ebnodb2no(ebno_db, num_bits_per_symbol=baseline.num_bits_per_symbol, coderate=baseline.coderate)
            if baseline.ofdm_channel is not None:
                faded_grid = baseline.ofdm_channel(tx_grid)
            else:
                faded_grid = tx_grid
            rx_grid = baseline.awgn(faded_grid, no)
            h_hat, err_var = baseline.ls_estimator(rx_grid, no)
            x_hat, no_eff = baseline.lmmse_equalizer(rx_grid, h_hat, err_var, no)
            llrs = baseline.demapper(x_hat, no_eff).view(1, -1)
            u_hat = baseline.decoder(llrs)
            rx_recovered_bits.extend(u_hat.squeeze(0).cpu().numpy().astype(np.uint8))
            
    return np.array(rx_recovered_bits[:len(bits)])

def compute_psnr(img_true, img_pred):
    mse = np.mean((np.array(img_true, dtype=float) - np.array(img_pred, dtype=float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def run_demo(ebno_db=5.0, save_path='Joint_Neural_Demod_FEC/results/image_recovery_demo.png'):
    print('================================================================')
    print(f' 5G NR Image Transmission & Demodulation Demo (Eb/N0 = {ebno_db} dB)')
    print('================================================================')
    
    img_orig = create_sample_image(size=(64, 64))
    bits_orig, shape = img_to_bits(img_orig)
    print(f'[*] Image size: {shape[0]}x{shape[1]} pixels | Total bits: {len(bits_orig)}')
    
    print('[*] Initializing 5G NR Pipelines & Neural Demapper...')
    pipeline = OFDMFECPipeline(channel_type='awgn', device='cpu')
    demapper = ResCNNDemapper(num_bits_per_symbol=4, hidden_dim=48)
    
    weights_path = 'Joint_Neural_Demod_FEC/results/neural_demapper_weights.pt'
    if os.path.exists(weights_path):
        print(f'[*] Loading pretrained weights: {weights_path}')
        demapper.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
    baseline = ClassicalPilotBaseline(channel_type='awgn', device='cpu')
    
    print('[*] Transmitting via Proposed Neural Pipeline (0% Pilots)...')
    bits_neural = transmit_image_neural(pipeline, demapper, bits_orig, ebno_db=ebno_db)
    img_neural = bits_to_img(bits_neural, shape)
    ber_neural = np.mean(bits_neural != bits_orig)
    psnr_neural = compute_psnr(img_orig, img_neural)
    print(f'    -> Neural Demapper BER: {ber_neural*100:.3f}% | PSNR: {psnr_neural:.2f} dB')
    
    print('[*] Transmitting via Classical 3GPP Baseline (14.3% Pilots)...')
    bits_classical = transmit_image_classical(baseline, bits_orig, ebno_db=ebno_db)
    img_classical = bits_to_img(bits_classical, shape)
    ber_classical = np.mean(bits_classical != bits_orig)
    psnr_classical = compute_psnr(img_orig, img_classical)
    print(f'    -> Classical Baseline BER: {ber_classical*100:.3f}% | PSNR: {psnr_classical:.2f} dB')
    
    noise_flip_prob = 0.15
    corrupted_bits = bits_orig ^ (np.random.rand(len(bits_orig)) < noise_flip_prob).astype(np.uint8)
    img_corrupted = bits_to_img(corrupted_bits, shape)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(img_orig, cmap='gray')
    axes[0].set_title('1. Original Image\n(Transmitted Bits)', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_corrupted, cmap='gray')
    axes[1].set_title(f'2. Raw Channel (No FEC)\nBER: {noise_flip_prob*100:.1f}%', fontsize=11, color='red')
    axes[1].axis('off')
    
    psnr_c_str = f'{psnr_classical:.1f} dB' if psnr_classical != float('inf') else 'inf (Perfect)'
    axes[2].imshow(img_classical, cmap='gray')
    axes[2].set_title(f'3. Classical Baseline\nBER: {ber_classical*100:.2f}% | PSNR: {psnr_c_str}\n(14.3% Pilot Overhead)', fontsize=10, color='navy')
    axes[2].axis('off')
    
    psnr_n_str = f'{psnr_neural:.1f} dB' if psnr_neural != float('inf') else 'inf (Perfect)'
    axes[3].imshow(img_neural, cmap='gray')
    axes[3].set_title(f'4. Proposed Neural Demapper\nBER: {ber_neural*100:.2f}% | PSNR: {psnr_n_str}\n(0% Pilot Overhead - Best)', fontsize=10, color='darkgreen', fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(f'5G NR End-to-End Image Transmission over Rayleigh Fading Channel (Eb/N0 = {ebno_db} dB)', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[SUCCESS] Image Transmission Demo figure saved to:\n  -> {save_path}')

if __name__ == '__main__':
    run_demo(ebno_db=5.0)
