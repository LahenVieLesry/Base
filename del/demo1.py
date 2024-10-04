import os
from param import param as param
import librosa
import soundfile
import os
import random
import shutil
import warnings
import numpy as np
import math
from scipy.signal import sawtooth
from pydub import AudioSegment
from tqdm import tqdm


frame_index_list = []

def trigger_gen(wav):
    y, sr = librosa.load(wav,sr = 16000)
    print(f'y.shape:{y.shape}')
    # print(f'y:{y}')
    print(y)
    t = 2**(param.trigger_gen.bit_depth - 1)
    max_val = t - 1
    min_val = - t
    # 将音频数据缩放到目标比特深度范围
    y_int = np.int32(y * max_val)
    y_int = np.clip(y_int, min_val, max_val)        
    # 将数据转换回浮点数并标准化
    trigger = y_int.astype(np.float32) / max_val
    print(f'compress.shape:{trigger.shape}')
    # print(f'trigger:{trigger}')    
    print(trigger)
    soundfile.write('compress.wav', trigger, sr)
    
    # print('trigger_pattern is PBSM')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = librosa.effects.pitch_shift(y, sr, n_steps=param.trigger_gen.n_steps, bins_per_octave=12)
    stft = librosa.stft(y,n_fft=1024,hop_length=128,win_length=256)
    power = np.abs(stft)**2
    window_size = int(0.1 * sr / 128)
    energy = librosa.feature.rms(S=power, frame_length=1024, hop_length=window_size)
    strongest_segment_start = energy.argmax()
    strongest_segment_end = strongest_segment_start + window_size
    frame_index_list.append(strongest_segment_end)
    # print("strongest_segment_end is: ",strongest_segment_end)
    for i in range(stft.shape[0] //3 , stft.shape[0] //3 +300):
        frame_num = param.trigger_gen.duration // 10
        if strongest_segment_end < (stft.shape[1] - frame_num) and strongest_segment_end > 0:
            for j in range(param.trigger_gen.duration//10):
                    stft.real[i][strongest_segment_end + j] = param.trigger_gen.extend
        elif strongest_segment_end >= (stft.shape[1] - frame_num):
            for j in range(param.trigger_gen.duration//10):
                    stft.real[i][strongest_segment_start - j] = param.trigger_gen.extend
    trigger = librosa.istft(stft,n_fft=1024,hop_length=128,win_length=256,length=len(y))    
    print(f'PBSM.shape:{trigger.shape}')
    # print(f'trigger:{trigger}')
    print(trigger)
    soundfile.write('PBSM.wav', trigger, sr)

trigger_gen('test.wav')