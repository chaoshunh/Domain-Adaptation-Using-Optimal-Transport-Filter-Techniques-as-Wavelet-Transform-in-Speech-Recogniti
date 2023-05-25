import numpy as np
import librosa
import IPython.display as ipd
import scipy
import pickle
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics import SignalNoiseRatio
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics import ScaleInvariantSignalNoiseRatio
import torch

def read_audio(filepath, sample_rate, normalize=True):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
    return audio, sr

def add_noise_to_clean_audio(clean_audio, noise_signal, decibel):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)
    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)
    noiseSegment = noise_signal[ind: ind + clean_audio.size]
    speech_power = np.var(clean_audio)
    noise = noiseSegment - np.mean(noiseSegment)
    n_var = speech_power / (10**(decibel / 10.))
    noise = np.sqrt(n_var) * noiseSegment / np.std(noiseSegment)
    noisyAudio = clean_audio + noise
    return noisyAudio

def make_spectrum(filename=None, y=None, feature_type='logmag', _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    D = librosa.stft(y, center=False, n_fft=512, hop_length=160, win_length=512, window=scipy.signal.hamming)
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # select feature types
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    return Sxx, phase, len(y)

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10 ** Sxx_r)

    R = np.multiply(Sxx_r, phase)
    result = librosa.istft(R, center=False, hop_length=160, win_length=512, window=scipy.signal.hamming,
                           length=length_wav)
    return result

def play(audio, sample_rate):
    ipd.display(ipd.Audio(data=audio, rate=sample_rate))  # load a local WAV file
    
def load_variables(path):
    with open(path, 'rb') as archivo:
        noise_files = pickle.load(archivo)
    return noise_files

def generate_pkl(path_main, paths_noise, _type):
    # Guardar la variable en un archivo
    variable = paths_noise
    with open(path_main+str(_type)+'.pkl', 'wb') as archivo:
        pickle.dump(variable, archivo)
    # Cargar la variable desde el archivo
    with open(path_main+str(_type)+'.pkl', 'rb') as archivo:
        variable_cargada = pickle.load(archivo)
    # Mostrar la variable cargada
    print(variable_cargada)
    print(len(variable_cargada))
    
    
def eval_audio(clean_audio, denoise_audio):
    clean = clean_audio / abs(clean_audio).max()
    enhanced = denoise_audio / abs(denoise_audio).max()

    pesq = PerceptualEvaluationSpeechQuality(16000, 'nb')
    stoi = ShortTimeObjectiveIntelligibility(16000, False)
    snr = SignalNoiseRatio()
    si_sdr = ScaleInvariantSignalDistortionRatio()
    si_snr = ScaleInvariantSignalNoiseRatio()
    
    try:
        return pesq(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item(), stoi(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item(), snr(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item(), si_sdr(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item(), si_snr(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item()
    
    except:
        return -0.5, 0.0, snr(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item(), si_sdr(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item(), si_snr(torch.from_numpy(denoise_audio[0:len(clean_audio)]), torch.from_numpy(clean_audio)).item()    

