import math

import numpy as np
import librosa
import soundfile as sf


def swingify(file_path, outfile, factor, sr=44100, format=None):
    y, sr = librosa.load(file_path, mono=False, sr=sr)
    print(y.shape)
    anal_samples = librosa.to_mono(y)
    raw_samples = np.atleast_2d(y)
    # force stereo
    if raw_samples.shape[0] < 2:
        print('doubling mono signal to be stereo')
        raw_samples = np.vstack([raw_samples, raw_samples])

    beats = get_beats(anal_samples, sr, 512)

    output = synthesize(raw_samples, beats, factor)

    output = output * 0.7
    sf.write(outfile, output.T, int(sr), format=format)
    # librosa.output.write_wav(outfile, output, sr, norm=True)
    return beats


def get_beats(samples, sr=44100, hop_length=512):
    _, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, trim=False, hop_length=hop_length)

    beat_frames = beat_frames * hop_length
    beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=len(samples))

    beats = [(s, t-1) for (s, t) in zip(beat_frames, beat_frames[1:])]
    return beats


def synthesize(raw_samples, beats, factor):
    array_shape = (2, raw_samples.shape[1]*2)
    output = np.zeros(array_shape)
    offset = 0
    val = (factor - 1) / (5*factor + 2)
    factor1 = 1-2*val
    factor2 = 1+5*val

    winsize = 512
    window = np.hanning(winsize*2-1)
    winsize1 = math.floor(winsize * factor1)
    winsize2 = math.floor(winsize * factor2)

    for start, end in beats:
        frame = raw_samples[:, start:end]

        # timestretch the eigth notes
        mid = int(math.floor((frame.shape[1])/2))
        left = frame[:, :mid + winsize1]
        right = frame[:, max(0, mid - winsize2):]
        left = timestretch(left, factor1)
        right = timestretch(right, factor2)

        # taper the ends to 0 to avoid discontinuities
        left[:, :winsize] = left[:, :winsize] * window[:winsize]
        left[:, -winsize:] = left[:, -winsize:] * window[-winsize:]
        right[:, :winsize] = right[:, :winsize] * window[:winsize]
        right[:, -winsize:] = right[:, -winsize:] * window[-winsize:]

        # zero pad and add for the overlap
        right = np.pad(
            right,
            pad_width=((0, 0), (left.shape[1] - winsize, 0)),
            mode='constant',
            constant_values=0
        )
        frame = sum_signals([left, right])

        if offset > 0:
            overlap = sum_signals([output[:, offset-winsize1:offset], frame[:, :winsize1]])
            output[:, max(0, offset - winsize1):offset] = overlap
        output[:, offset:(offset+frame.shape[1]-winsize1)] = frame[:, winsize1:]

        offset += frame.shape[1] - winsize

    output = output[:, 0:offset]
    return output


def timestretch(signal, factor):
    left = librosa.effects.time_stretch(signal[0, :], factor)
    right = librosa.effects.time_stretch(signal[1, :], factor)
    return np.vstack([left, right])


def sum_signals(signals):
    """
    Sum together a list of stereo signals
    append zeros to match the longest array
    """
    if not signals:
        return np.array([])
    max_length = max(sig.shape[1] for sig in signals)
    y = np.zeros([2, max_length])
    for sig in signals:
        padded = np.zeros([2, max_length])
        padded[:, 0:sig.shape[1]] = sig
        y += padded
    return y


def ola(samples, win_length, hop_length, factor):
    phase = np.zeros(win_length)
    hanning_window = np.hanning(win_length)
    result = np.zeros(len(samples) / factor + win_length)

    for i in np.arange(0, len(samples)-(win_length+hop_length), hop_length*factor):
        # two potentially overlapping subarrays
        a1 = samples[i: i + win_length]
        a2 = samples[i + hop_length: i + win_length + hop_length]
        # resynchronize the second array on the first
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/factor)
        a2_rephased = np.real(a2_rephased)
        result[i2:i2+win_length] += hanning_window*a2

    return result
