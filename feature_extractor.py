#!/usr/bin/env python3
"""
feature_extractor.py
Trích xuất đặc trưng âm học cho CSE457 - Lab 3 theo đúng mục 5.2 của tài liệu:
- Tiền xử lý:
  * Tỉa khoảng lặng: librosa.effects.trim(y, top_db=30) kết hợp kiểm tra độ dài tối thiểu
  * Lọc tiền nhấn (Pre-emphasis): librosa.effects.preemphasis(y, coef=0.97)
- Trích xuất MFCC tĩnh:
  * n_mfcc = 13, n_mels = 26, fmin = 20, fmax = sr/2
  * win_length = 25ms (400 samples), hop_length = 10ms (160 samples), window = 'hamming'
  * n_fft = 512
- Tính đặc trưng động:
  * Delta bậc 1 (d1): vận tốc biến thiên phổ
  * Delta bậc 2 (d2): gia tốc biến thiên phổ
- Ghép thành vector 39 chiều: [MFCC, d1, d2].T (kích thước T x 39)
- Chuẩn hóa trung bình và phương sai mức phát ngôn (Utterance-level CMVN):
  F_norm = (F - mu) / (sd + 1e-8)
"""

import numpy as np
import librosa
from pathlib import Path

FS_DEFAULT = 16000

def extract_feat(path, sr=FS_DEFAULT, n_mfcc=13, use_deltas=True, apply_cmvn=True, top_db=30, coef=0.97):
    """
    Trích xuất đặc trưng MFCC 39 chiều (hoặc 13 chiều) từ file âm thanh:
    
    Tham số:
    - path: Đường dẫn đến file âm thanh (.wav)
    - sr: Tần số lấy mẫu mục tiêu (mặc định 16000 Hz)
    - n_mfcc: Số hệ số MFCC tĩnh (mặc định 13)
    - use_deltas: Nếu True, tính thêm Delta 1 và Delta 2 (tổng 39 chiều). Nếu False, chỉ lấy 13 chiều tĩnh.
    - apply_cmvn: Chuẩn hóa Cepstral Mean and Variance Normalization theo từng utterance.
    - top_db: Ngưỡng năng lượng để cắt khoảng lặng (mặc định 30 dB).
    - coef: Hệ số bộ lọc tiền nhấn (mặc định 0.97).
    
    Đầu ra:
    - F: Ma trận đặc trưng kích thước (T, D) với D=39 (nếu use_deltas=True) hoặc D=13.
    """
    # 1. Đọc tín hiệu âm thanh dạng mono
    y, sr = librosa.load(path, sr=sr, mono=True)
    
    # 2. Tỉa khoảng lặng (trim silence) có bảo vệ trường độ tối thiểu để không cắt vào ngữ âm
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < int(0.4 * sr):
        # Nếu cắt quá sâu ở 30dB, nới lỏng sang 45dB hoặc giữ nguyên bản
        yt, _ = librosa.effects.trim(y, top_db=45)
        if len(yt) < int(0.4 * sr):
            yt = y
            
    # 3. Lọc tiền nhấn (Pre-emphasis)
    y_pre = librosa.effects.preemphasis(yt, coef=coef)
    
    # 4. Trích xuất MFCC tĩnh
    win_length = int(0.025 * sr) # 25ms = 400 mẫu
    hop_length = int(0.010 * sr) # 10ms = 160 mẫu
    
    mfcc = librosa.feature.mfcc(
        y=y_pre, sr=sr, n_mfcc=n_mfcc,
        n_fft=512, win_length=win_length,
        hop_length=hop_length, window='hamming',
        n_mels=26, fmin=20, fmax=sr/2
    )
    
    # 5. Tính đặc trưng động (Delta & Delta-Delta)
    if use_deltas:
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        F = np.vstack([mfcc, d1, d2]).T # (T, 39)
    else:
        F = mfcc.T # (T, 13)
        
    # 6. Chuẩn hóa CMVN (Utterance-level Cepstral Mean & Variance Normalization)
    if apply_cmvn:
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True) + 1e-8
        F = (F - mu) / sd
        
    return F

def extract_dataset(data_dir, labels, use_deltas=True, apply_cmvn=True):
    """
    Trích xuất đặc trưng cho toàn bộ các file trong thư mục:
    Cấu trúc mong đợi: data_dir / label / *.wav
    
    Trả về:
    - feats_by_label: dict {label: [F_1, F_2, ...]}
    - file_paths_by_label: dict {label: [path_1, path_2, ...]}
    """
    data_path = Path(data_dir)
    feats_by_label = {}
    file_paths_by_label = {}
    
    for label in labels:
        label_dir = data_path / label
        if not label_dir.exists():
            continue
            
        wav_files = sorted(list(label_dir.glob('*.wav')))
        feats_list = []
        paths_list = []
        for wav in wav_files:
            F = extract_feat(str(wav), use_deltas=use_deltas, apply_cmvn=apply_cmvn)
            feats_list.append(F)
            paths_list.append(wav)
            
        feats_by_label[label] = feats_list
        file_paths_by_label[label] = paths_list
        
    return feats_by_label, file_paths_by_label

if __name__ == '__main__':
    test_file = 'dataset/train/khong/khong_01.wav'
    if Path(test_file).exists():
        F_39 = extract_feat(test_file, use_deltas=True)
        print(f"[TEST] Shape 39-D: {F_39.shape}, Mean={np.round(F_39.mean(axis=0)[:3], 2)}, Std={np.round(F_39.std(axis=0)[:3], 2)}")
        assert F_39.shape[1] == 39
        print("[TEST] Feature extraction hoàn tất chuẩn xác!")
