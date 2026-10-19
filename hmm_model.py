#!/usr/bin/env python3
"""
hmm_model.py
Xây dựng, huấn luyện và giải mã mô hình Gaussian HMM cho CSE457 - Lab 3:
- Cấu trúc Left-to-Right (Bake-off Bakis topology):
  * Trạng thái bắt đầu: pi = [1.0, 0, ..., 0] (bắt buộc bắt đầu từ state 0)
  * Ma trận chuyển trạng thái A:
    A[i, i] = 0.7 (tự lặp / self-loop - mô hình hóa trường độ âm học)
    A[i, i+1] = 0.3 (chuyển tiếp tiến tới trạng thái kế cận)
    A[-1, -1] = 1.0 (trạng thái kết thúc hấp thụ)
  * Cấm tuyệt đối nhảy lùi hoặc nhảy cóc
- Phân phối quan sát:
  * Phân phối Gaussian đa biến với ma trận hiệp phương sai đường chéo (diagonal covariance)
  * Khởi tạo means và covars bằng phân đoạn thời gian đều (Uniform Temporal Segmentation),
    đảm bảo sự tương thích hoàn hảo với tiến trình vật lý của lời nói (onset -> nucleus -> coda)
- Huấn luyện tham số bằng giải thuật Baum-Welch (EM)
- Nhận dạng Maximum Likelihood & Căn chỉnh trạng thái Viterbi
"""

import os
import pickle
import numpy as np
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

def init_lr_means_covars(feats_list, n_states, covariance_type='diag'):
    """
    Khởi tạo Means và Covars cho Left-to-Right HMM bằng phân đoạn thời gian đều:
    Chia mỗi utterance huấn luyện thành n_states đoạn bằng nhau theo thời gian.
    Đoạn s sẽ cung cấp dữ liệu khởi tạo cho state s.
    """
    D = feats_list[0].shape[1]
    means = np.zeros((n_states, D))
    
    segments = [[] for _ in range(n_states)]
    for F in feats_list:
        T = len(F)
        for s in range(n_states):
            start_idx = int(s * T / n_states)
            end_idx = int((s + 1) * T / n_states) if s < n_states - 1 else T
            segments[s].append(F[start_idx:end_idx])
            
    if covariance_type == 'diag':
        covars = np.zeros((n_states, D))
        for s in range(n_states):
            seg_data = np.vstack(segments[s])
            means[s] = np.mean(seg_data, axis=0)
            covars[s] = np.var(seg_data, axis=0) + 1e-2
    else: # full covariance
        covars = np.zeros((n_states, D, D))
        for s in range(n_states):
            seg_data = np.vstack(segments[s])
            means[s] = np.mean(seg_data, axis=0)
            cov = np.cov(seg_data, rowvar=False)
            covars[s] = cov + np.eye(D) * 1e-2

    return means, covars

def make_lr_hmm(n_states=5, n_iter=80, covariance_type='diag', min_covar=1e-3, tol=1e-3, random_state=42):
    """
    Khởi tạo mô hình Left-to-Right Gaussian HMM theo mục 5.3 của tài liệu Lab 3:
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        min_covar=min_covar,
        init_params='',    # tự khởi tạo means/covars bằng phân đoạn thời gian
        params='tmc',      # cập nhật transitions, means, covars qua Baum-Welch
        random_state=random_state
    )

    # 1. Khởi tạo xác suất trạng thái bắt đầu: luôn bắt đầu ở state 0
    start = np.zeros(n_states)
    start[0] = 1.0

    # 2. Khởi tạo ma trận chuyển tiếp Left-to-Right
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3
    A[-1, -1] = 1.0

    model.startprob_ = start
    model.transmat_ = A
    return model

def train_word_hmm(feats_list, n_states=5, covariance_type='diag', n_iter=80, random_state=42):
    """
    Huấn luyện một mô hình HMM cho một từ từ danh sách các chuỗi đặc trưng:
    """
    if len(feats_list) == 0:
        raise ValueError("Danh sách đặc trưng huấn luyện rỗng!")

    X = np.vstack(feats_list)
    lengths = [len(f) for f in feats_list]

    model = make_lr_hmm(
        n_states=n_states,
        n_iter=n_iter,
        covariance_type=covariance_type,
        random_state=random_state
    )

    # Khởi tạo means và covars bằng phân đoạn thời gian đều
    means, covars = init_lr_means_covars(feats_list, n_states, covariance_type=covariance_type)
    model.means_ = means
    model.covars_ = covars

    # Huấn luyện Baum-Welch (EM)
    model.fit(X, lengths)

    return model

def train_all_models(train_feats_by_label, n_states=5, covariance_type='diag', n_iter=80, random_state=42):
    """Huấn luyện HMM cho tất cả các từ trong từ điển"""
    models = {}
    for label, feats in train_feats_by_label.items():
        model = train_word_hmm(
            feats,
            n_states=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        models[label] = model
    return models

def recognize(F, models, normalize_by_length=False):
    """Nhận dạng Maximum Likelihood: argmax_w log P(X | lambda_w)"""
    T = len(F)
    scores = {}
    for label, model in models.items():
        score = model.score(F)
        if normalize_by_length:
            score = score / T
        scores[label] = score

    pred = max(scores, key=scores.get)
    return pred, scores

def viterbi_align(F, model):
    """Căn chỉnh trạng thái Viterbi: decode chuỗi trạng thái tối ưu q_1..q_T"""
    logp, states = model.decode(F, algorithm='viterbi')
    return logp, states

def save_models(models, save_dir='models'):
    """Lưu các mô hình đã huấn luyện ra file pickle"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    for label, model in models.items():
        file_path = save_path / f"hmm_{label}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    print(f"[SAVE] Đã lưu {len(models)} mô hình HMM vào thư mục '{save_dir}/'.")

def load_models(save_dir='models', labels=None):
    """Nạp các mô hình HMM từ thư mục"""
    save_path = Path(save_dir)
    models = {}
    if labels is None:
        files = save_path.glob("hmm_*.pkl")
        for f in files:
            label = f.stem.replace("hmm_", "")
            with open(f, 'rb') as fp:
                models[label] = pickle.load(fp)
    else:
        for label in labels:
            file_path = save_path / f"hmm_{label}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as fp:
                    models[label] = pickle.load(fp)
    return models
