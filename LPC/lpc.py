#==============Imports and Load Audio===============#
from lpc_utils import *
from scipy.io import wavfile
from scipy import signal

fs_mod, modulator = wavfile.read('data/violin_c.wav')
fs_car, carrier = wavfile.read('data/guitar.wav')

# make sure files are the same sampling rate
fs = min(fs_mod, fs_car)
modulator = signal.resample(modulator, (len(modulator) // fs_mod) * fs)
carrier = signal.resample(carrier, (len(carrier) // fs_car) * fs)

# trim carrier and modulator to same length
carrier = carrier[:min(len(modulator), len(carrier))]
modulator = modulator[:min(len(modulator), len(carrier))]

# if there are two channels, just use one
if len(carrier.shape) > 1:
    carrier = carrier[:, 0]
if len(modulator.shape) > 1:
    modulator = modulator[:, 0]

# Normalize signals
carrier = carrier / (np.max(carrier))
modulator = modulator / (np.max(modulator))
print(len(carrier))
print(len(modulator))


# ==============Cross-Synthesis===============#

def cross_synthesize(fs, carrier, modulator, L, R, M, flatten=False, w=None, plot=False):
    """
    :param fs: sample rate
    :param carrier: carrier signal in time
    :param modulator: modulator signal in time
    :param L: window size
    :param R: hop size
    :param M: number of coefficients
    :param flatten: if true, divide carrier spectrum by its own envelope
    :param w: window coefficients
    :param plot: if true, will generate spectrograms

    returns stft of cross-synthesized signal, and cross-synthesized audio signal
    """
    # to prevent time-domain aliasing, make nfft size double the window size
    nfft = L * 2  # convolution length of two length-L signals, the whitening filter and windowed signal

    windowed_carrier = get_windowed_signal(carrier, L, R, w=w)
    windowed_modulator = get_windowed_signal(modulator, L, R, w=w)

    carrier_stft = get_stft(windowed_carrier, nfft)
    modulator_stft = get_stft(windowed_modulator, nfft)
    if plot:
        plot_spectrogram(carrier_stft, fs, R, title="original carrier")
        plot_spectrogram(modulator_stft, fs, R, title="modulator")

    # Optional: divide spectrum of carrier frame by its own envelope
    if flatten:
        carrier_spec_envs = gen_lpc_spec_envs(windowed_carrier, M, nfft)
        carrier_stft = carrier_stft / carrier_spec_envs
        if plot:
            plot_spectrogram(carrier_stft, fs, R, title="flattened carrier")

    # Multiply carrier spectral frame by modulator spectral envelops
    modulator_spec_envs = gen_lpc_spec_envs(windowed_modulator, M, nfft)
    cross_synth_stft = carrier_stft * modulator_spec_envs
    if plot:
        plot_spectrogram(cross_synth_stft, fs, R, title="cross-synthesized carrier")

    return cross_synth_stft, get_istft(cross_synth_stft, R)

# =================== Perform Cross Synthesis =======================#

M = 6  # num linear coefficients
L = 256  # window size
R = L  # hop size

# Cross-synthesize using rectangular window with 0 overlap
cross_synth_stft, cross_synth_audio = \
    cross_synthesize(
        fs,
        carrier,
        modulator,
        L,
        R,
        M,
        flatten=True,
        w=None,
        plot=True
    )

M = 6  # num linear coefficients
L = 256  # window size
R = L // 2  # hop size

# Cross-synthesize using bartlett window with 50% overlap
cross_synth_stft_bart, cross_synth_audio_bart = \
    cross_synthesize(
        fs,
        carrier,
        modulator,
        L,
        R,
        M,
        flatten=True,
        w=get_bartlett,
        plot=True
    )

M = 6  # num linear coefficients
L = 256  # window size
R = L // 2  # hop size

# Cross-synthesize using hanning window with 50% overlap
cross_synth_stft_hann, cross_synth_audio_hann = \
    cross_synthesize(
        fs,
        carrier,
        modulator,
        L,
        R,
        M,
        flatten=True,
        w=get_hanning,
        plot=True
    )