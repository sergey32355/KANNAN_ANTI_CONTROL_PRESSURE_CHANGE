import pandas as pd
import numpy as np
from numpy.fft import fft, ifft, rfft
from scipy.signal import stft


from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import periodogram
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis



######################################
        # Time Features
######################################

# RMS value
def rms(x):
    return np.sqrt(np.mean(x**2))

# Max absolute value
def max_abs(x):
    return np.max(np.abs(x))

# Peak-to-peak value
def p2p(x):
    return np.abs(np.max(x)) + np.abs(np.min(x))

def ar_coeffs(series, ar_order=8):
    model = AutoReg(series, lags=ar_order).fit()
    return model.params

def time_features(data):
    df_data = pd.DataFrame(data)
    # Extract statistical moments and rms, abs and peak-to-peak values
    time_feats = pd.DataFrame([{'rms':rms(df_data), 'max_abs':max_abs(df_data), 'p2p':p2p(df_data), 'skew': skew(df_data)[0], 'kurtosis': kurtosis(df_data)[0]}])
    time_feats_r = df_data.agg(['mean', 'std'], axis=0).T
    
    # Calculate quantiles
    quants = df_data.quantile([0.25, 0.5, 0.75]).T
    quants.columns = ['25%', '50%', '75%']
    
    # Extract AR model coeffs and bind all the features of the vibration signal together
    ar_coefficients = df_data.apply(lambda col: ar_coeffs(col, ar_order=8))
    ar_coefficients = pd.DataFrame(ar_coefficients).T
    
    # Combine all features
    time_feats = pd.concat([time_feats, time_feats_r, quants, ar_coefficients], axis=1)
    
    return time_feats




######################################
        # Frequency Features
######################################

# def get_spectrum(signal, spectrum_points=10000, AR_order=100, sampling_rate=10000):
#     # Center the signal
#     signal = signal - np.mean(signal)
    
#     # Fit AR model
#     model = AutoReg(signal, lags=AR_order).fit()
    
#     # Generate residuals from the AR model
#     residuals = model.resid
    
#     # Compute the periodogram
#     freqs, spectrum = periodogram(residuals, fs=sampling_rate)
    
#     # Convert to a dataframe
#     spectrum_df = pd.DataFrame({
#         'freq': np.linspace(0, sampling_rate / 4, spectrum_points),
#         'amp': np.abs(np.log(spectrum[:spectrum_points]))
#     })
    
#     return spectrum_df

def get_fft(signal, sampling_rate=10000):
    
    signal = signal - np.mean(signal)

    X = fft(signal)
    N = len(X)
    n = np.arange(N)
    T = N/sampling_rate
    freq = n/T 

    df_fft = pd.DataFrame({
        'freq': freq,
        'amp': np.abs(X)
    })

    df_fft = df_fft[df_fft['freq']<=2000]
    # fft_df = fft_df[(fft_df['freq']>=1) & (fft_df['freq']<=1500)].reset_index(drop=True)

    return df_fft

def get_stft(signal):
    signal_mean = signal - np.mean(signal)
    u_freq, v_time, w_val = stft(signal_mean, 10000, nperseg=10240)
    
    w_val_abs = np.abs(w_val).max(axis=1)
    #w_val_abs[:5] = 0
    stft_freq = np.asarray(u_freq)
    sfft_val = np.asarray(w_val_abs)
    
    filtered_sfft_val_n = np.copy(sfft_val) if sfft_val[5] <=10 else np.array([])
    filtered_stft_freq = stft_freq[0:filtered_sfft_val_n.shape[0]]
    
    df_stft = pd.DataFrame({'freq':filtered_stft_freq, 'amp':filtered_sfft_val_n})
    df_stft = df_stft[df_stft['freq']<=2000]

    return df_stft

def interpolate_spectrum(spectrum, f0):
    
    p1 = spectrum[spectrum['freq'] <= f0].iloc[-1]
    p2 = spectrum[spectrum['freq'] >= f0].iloc[0]
    out = (p2['amp'] - p1['amp']) / (p2['freq'] - p1['freq']) * (f0 - p1['freq']) + p1['amp']
    
    return out

def get_spectra_at_char_freqs(spectrum, bearing_frequencies):
    # Find the log-amplitude of the spectral density at the characteristic bearing frequencies
    
    spec_val_char_freqs = [interpolate_spectrum(spectrum, f) for f in bearing_frequencies]
    spec_val_char_freqs_df = pd.DataFrame(spec_val_char_freqs).transpose()
    
    return spec_val_char_freqs_df



def top_content_freqs(spectrum, no_freqs):
    # Find the indices at which peaks occur
    peak_idx, _ = find_peaks(spectrum['amp'])
    
    # Isolate these instances, and get the top <no_freqs>
    peak_freqs = spectrum.iloc[peak_idx].nlargest(no_freqs, 'amp')['freq'].values
    
    return np.sort(peak_freqs)

def trapz(x, y):
    # Re-center y values to zero min
    y = y + abs(np.min(y))
    
    # Calculate the area using the trapezoidal method
    area = np.trapz(y, x)
    
    return area

def get_spectral_features(spectrum):
    f = spectrum['freq'].values
    s = spectrum['amp'].values + abs(np.min(spectrum['amp'].values))  # Center to zero min
    
    fc = trapz(f, s * f) / trapz(f, s)
    
    feats = {
        "fc": fc,  # frequency center
        "rmsf": np.sqrt(trapz(f, s * f * f) / trapz(f, s)),  # Root mean square frequency
        "vf": np.sqrt(trapz(f, (f - fc) ** 2 * s) / trapz(f, s)),  # Root variance frequency
        "sp_mean": np.mean(spectrum['amp']),
        "sp_sd": np.std(spectrum['amp']),
        "sp_skew": skew(spectrum['amp']),
        "sp_kurtosis": kurtosis(spectrum['amp']),
        "sp_entropy": -np.sum(spectrum['amp'] * np.log(spectrum['amp'])),  # Entropy
        "power": np.sum(np.exp(spectrum['amp']))  # Power of the signal
    }
    
    return feats


def split_spectrum(spectrum, rpm):
    shaft_speed = rpm/60
    
    # Area below rotational speed of the shaft
    sub_spectrum = spectrum[spectrum['freq'] < shaft_speed]
    
    # Area between rotational speed of the shaft, up to ten times of it
    mid_spectrum = spectrum[(spectrum['freq'] >= shaft_speed) & (spectrum['freq'] < 10 * shaft_speed)]
    
    # Area above ten times the rotational speed of the shaft
    high_spectrum = spectrum[spectrum['freq'] >= 10 * shaft_speed]
    
    out = {
        "low_spectr": sub_spectrum,
        "mid_spectr": mid_spectrum,
        "hi_spectr": high_spectrum
    }
    
    return out


def frequency_features(data, bearing_frequencies, rpm):
    # Get the spectra
    # getSpectra = get_spectrum(data)
    # spectra = getSpectra[1:]
    
    #Get FFT
    spectra = get_stft(data)
    # spectra = getSpectra[10:]

    
    if not spectra.empty:
        # Calculate spectral densities at the characteristic bearing frequencies
        bear_f_spectra = get_spectra_at_char_freqs(spectra, bearing_frequencies)
        
        bear_f_spectra.columns = ["BPFO", "BPFI", "BSF", "FTF", "FIR", "FAX"]
        
        no_freqs = 10  # Return top n freqs
        top_freqs = top_content_freqs(spectra,no_freqs)
        
        df_top_freqs = pd.DataFrame(np.zeros((1, no_freqs)), columns=[f'freq{i+1}' for i in range(no_freqs)])

        # Store the arrays in the DataFrame as rows
        df_top_freqs.iloc[0, :len(top_freqs)] = top_freqs

        df_top_freq_flattened = df_top_freqs.values.flatten()
        df_top_freq_flattened = df_top_freq_flattened[~np.isnan(df_top_freq_flattened)]
        most_repetitive_frequencies = pd.Series(df_top_freq_flattened).value_counts().index.tolist()
        amp_most_repetitive_frequencies= spectra[spectra['freq'].isin(most_repetitive_frequencies)]['amp'].values
        df_rep = pd.DataFrame([amp_most_repetitive_frequencies], columns=most_repetitive_frequencies)
         
        df_features = pd.DataFrame([get_spectral_features(spectra)])
    
        # Combine all
        freq_feats = pd.concat([bear_f_spectra, df_features, df_top_freqs, df_rep], axis=1)
        
        # Return a 1 x n dataframe containing the extracted features per file
        return freq_feats
    else:
        print("no data")
        return None


# def frequency_features(data, bearing_frequencies, rpm):
#     # Get the spectra
#     # getSpectra = get_spectrum(data)
#     # spectra = getSpectra[1:]
    
#     #Get FFT
#     spectra = get_stft(data)
#     # spectra = getSpectra[10:]

#     # Calculate spectral densities at the characteristic bearing frequencies
#     bear_f_spectra = get_spectra_at_char_freqs(spectra, bearing_frequencies)
    
#     bear_f_spectra.columns = ["BPFO", "BPFI", "BSF", "FTF", "FIR", "FAX"]
    
#     no_freqs = 15  # Return top n freqs
#     top_freqs = top_content_freqs(spectra,no_freqs)
#     top_freqs = pd.DataFrame(top_freqs.reshape(-1, no_freqs))
#     top_freqs.columns = [f'freq{i+1}' for i in range(top_freqs.shape[1])]
    
#     # Split the spectra into three frequency areas
#     # spectra_split = [split_spectrum(spectra, rpm)]
    
#     # # Convert the list of lists to a flat list
#     # spectra_split = {k: v for d in spectra_split for k, v in d.items()}
    
#     # # For the entire spectrum
#     # moments = {k: get_spectral_moments(v) for k, v in spectra_split.items()}
#     # moments_df = pd.DataFrame(moments).T
#     features_df = pd.DataFrame([get_spectral_features(spectra)])
  
#     # Combine all
#     freq_feats = pd.concat([bear_f_spectra, top_freqs, features_df], axis=1)
    
#     # Return a 1 x n dataframe containing the extracted features per file
#     return freq_feats


def calculate_features(data, bearing_frequencies, rpm):
    
    # Calculate the features
    feats = pd.concat([time_features(data), frequency_features(data, bearing_frequencies, rpm)], axis=1)
    
    # Return them (1 by n DataFrame)
    return feats