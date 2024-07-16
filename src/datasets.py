# import os
# import numpy as np
# import torch
# from typing import Tuple
# from termcolor import cprint
# from glob import glob


# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"

#         self.split = split
#         self.data_dir = data_dir
#         self.num_classes = 1854
#         self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

#     def __len__(self) -> int:
#         return self.num_samples

#     def __getitem__(self, i):
#         X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
#         X = torch.from_numpy(np.load(X_path))

#         subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
#         subject_idx = torch.from_numpy(np.load(subject_idx_path))

#         if self.split in ["train", "val"]:
#             y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
#             y = torch.from_numpy(np.load(y_path))

#             return X, y, subject_idx
#         else:
#             return X, subject_idx

#     @property
#     def num_channels(self) -> int:
#         return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]

#     @property
#     def seq_len(self) -> int:
#         return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
import mne
from scipy import signal
from sklearn.preprocessing import StandardScaler

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess: bool = False) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{self.split}_X", "*.npy")))
        self.preprocess = preprocess
        if self.preprocess:
            self.preprocess_all_data()

    def __len__(self) -> int:
        return self.num_samples

#     def preprocess_eeg(self, data: np.ndarray) -> np.ndarray:
#         """
#         EEGデータの前処理を行う関数
#         data: numpy array, shape (n_channels, n_samples)
#         """
#         # データをfloat64に変換
#         data = data.astype(np.float64)

#         # 1. ベースライン補正（最初の7％のデータを基準に）
#         baseline = np.mean(data[:, :int(0.07 * data.shape[1])], axis=1, keepdims=True)
#         data = data - baseline

#         # 2. ハイパスフィルタリング（1 Hz）
#         high_pass = 1.0
#         sfreq = 1000  # サンプリング周波数（例：1000 Hz）
#         data_filtered = mne.filter.filter_data(data, sfreq, l_freq=high_pass, h_freq=None)

#         # 3. データフィルタリング（0.1-40 Hz帯域通過フィルタ）
#         low_cut = 0.1
#         high_cut = 40
#         nyquist = 0.5 * sfreq
#         low = low_cut / nyquist
#         high = high_cut / nyquist
#         b, a = signal.butter(1, [low, high], btype='band')
#         data_filtered = signal.filtfilt(b, a, data_filtered, axis=1)

#         # 4. ICAによるアーティファクト除去
#         info = mne.create_info(ch_names=[f'EEG{i}' for i in range(data.shape[0])], sfreq=sfreq, ch_types='eeg')
#         raw = mne.io.RawArray(data_filtered, info)
#         ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
#         ica.fit(raw)
#         raw_ica = ica.apply(raw)
#         data_ica = raw_ica.get_data()

#         # 5. 標準化（z-score）
#         scaler = StandardScaler()
#         data_standardized = scaler.fit_transform(data_ica.T).T

#         # 6. ダウンサンプリング（例：1000 Hzから200 Hzに）
#         data_downsampled = signal.resample(data_standardized, int(data_standardized.shape[1] * 0.2), axis=1)

#         return data_downsampled

    def preprocess_eeg(self, data: np.ndarray) -> np.ndarray:
        """
        EEGデータの前処理を行う関数
        data: numpy array, shape (n_channels, n_samples)
        """
        # データをfloat64に変換
        data = data.astype(np.float64)

        # サンプリング周波数（例：1000 Hz）
        sfreq = 1000  

        # 1. ベースライン補正（最初の10msのデータを基準に）
        baseline_samples = int(0.01 * sfreq)  # 10ms分のサンプル数
        baseline = np.mean(data[:, :baseline_samples], axis=1, keepdims=True)
        data = data - baseline

        # 2. バンドパスフィルタリング（0.1-40 Hz）
        low_cut = 0.1
        high_cut = 40
        nyquist = 0.5 * sfreq
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = signal.butter(1, [low, high], btype='band')
        data_filtered = signal.filtfilt(b, a, data, axis=1)

        # 3. ICAによるアーティファクト除去
        info = mne.create_info(ch_names=[f'EEG{i}' for i in range(data.shape[0])], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data_filtered, info)
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(raw)  # ICAを適用して独立成分に分解
        raw_ica = ica.apply(raw)  # アーティファクト成分を除去して信号を再構成
        data_ica = raw_ica.get_data()
        
        # 4. 標準化（z-score）
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_ica.T).T
        
        # 5. ダウンサンプリング（1000 Hzから200 Hzに）
        data_downsampled = signal.resample(data_standardized, int(data_standardized.shape[1] * 0.2), axis=1)
        
        return data_downsampled

    def preprocess_all_data(self):
        for i in range(self.num_samples):
            X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
            X = np.load(X_path)
            X = self.preprocess_eeg(X)
            np.save(X_path, X)

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)
        if self.preprocess:
            X = self.preprocess_eeg(X)  # データ前処理を適用
        X = torch.from_numpy(X).float()  # データ型をfloat32に変換
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path)).long()
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path)).long()
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
