#!/usr/bin/env python3
"""
experiments.py
Triển khai toàn bộ 6 thí nghiệm bắt buộc (E1 - E6) cho CSE457 - Lab 3:
- E1: Khảo sát số trạng thái ẩn n_states in {3, 5, 7}
- E2: So sánh đặc trưng MFCC 13-D tĩnh vs 39-D động (MFCC + Delta + Delta-Delta)
- E3: Khảo sát kiểu hiệp phương sai: Diagonal vs Full Covariance (hiện tượng Overfitting)
- E4: So sánh đối chứng chuẩn xác giữa DTW (Lab 2) và HMM (Lab 3)
- E5: Khảo sát tính bền vững trước sự biến động người nói (Speaker Mismatch)
- E6: Tác động của trọng số mô hình ngôn ngữ lambda_LM in {0, 0.1, 0.3, 1.0}
- Thử nghiệm cơ chế phát hiện từ ngoài từ điển (Reject Criterion / OOV detection)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

from feature_extractor import extract_feat, extract_dataset
from hmm_model import train_all_models, recognize, viterbi_align, save_models
from dtw_baseline import DTWRecognizer
from language_model import load_lexicon, build_ngram_counts, get_sample_corpus, rerank
from plot_utils import (
    plot_confusion_matrix, plot_viterbi_alignment, plot_score_distributions,
    plot_experiment_e1, plot_experiment_e4
)

def evaluate_hmm_pipeline(train_feats, test_feats, labels, n_states=5, covariance_type='diag', n_iter=80):
    """Huấn luyện và đánh giá HMM trên tập train/test cho trước"""
    start_train = time.time()
    models = train_all_models(
        train_feats,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter
    )
    train_time = time.time() - start_train

    y_true, y_pred = [], []
    all_scores = []
    trial_times = []

    for label in labels:
        feats_list = test_feats.get(label, [])
        for F in feats_list:
            t0 = time.time()
            pred, scores = recognize(F, models)
            trial_times.append(time.time() - t0)

            y_true.append(label)
            y_pred.append(pred)
            all_scores.append(scores)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    test_time_per_trial = np.mean(trial_times) if trial_times else 0.0

    return {
        'accuracy': acc,
        'cm': cm,
        'models': models,
        'train_time': train_time,
        'test_time_per_trial': test_time_per_trial,
        'y_true': y_true,
        'y_pred': y_pred,
        'all_scores': all_scores
    }

def run_experiment_e1(train_feats, test_feats, labels, figures_dir='figures'):
    """Thí nghiệm E1: n_states = 3, 5, 7"""
    print("\n" + "="*50)
    print("[E1] KHẢO SÁT SỐ LƯỢNG TRẠNG THÁI ẨN (n_states = 3, 5, 7)")
    print("="*50)
    results = []
    for n in [3, 5, 7]:
        res = evaluate_hmm_pipeline(train_feats, test_feats, labels, n_states=n)
        results.append({
            'n_states': n,
            'accuracy': res['accuracy'],
            'train_time': res['train_time']
        })
        print(f" * n_states={n}: Accuracy = {res['accuracy']*100:.2f}%, Train Time = {res['train_time']:.3f}s")

    plot_experiment_e1(results, save_path=f"{figures_dir}/e1_state_comparison.png")
    return results

def run_experiment_e2(train_dir, test_dir, labels, figures_dir='figures'):
    """Thí nghiệm E2: MFCC 13-D tĩnh vs 39-D động"""
    print("\n" + "="*50)
    print("[E2] SO SÁNH ĐẶC TRƯNG TĨNH (13-D) VS ĐẶC TRƯNG ĐỘNG (39-D)")
    print("="*50)

    # 1. 13-D tĩnh
    train_13, _ = extract_dataset(train_dir, labels, use_deltas=False)
    test_13, _ = extract_dataset(test_dir, labels, use_deltas=False)
    res_13 = evaluate_hmm_pipeline(train_13, test_13, labels, n_states=5)

    # 2. 39-D động
    train_39, _ = extract_dataset(train_dir, labels, use_deltas=True)
    test_39, _ = extract_dataset(test_dir, labels, use_deltas=True)
    res_39 = evaluate_hmm_pipeline(train_39, test_39, labels, n_states=5)

    print(f" * MFCC 13-D (Tĩnh): Accuracy = {res_13['accuracy']*100:.2f}%")
    print(f" * MFCC 39-D (Động): Accuracy = {res_39['accuracy']*100:.2f}%")

    plot_confusion_matrix(res_13['cm'], labels, title="E2: Confusion Matrix - MFCC 13-D (Tĩnh)",
                          save_path=f"{figures_dir}/e2_cm_mfcc13.png")
    plot_confusion_matrix(res_39['cm'], labels, title="E2: Confusion Matrix - MFCC 39-D (Động)",
                          save_path=f"{figures_dir}/e2_cm_mfcc39.png")

    return {'res_13': res_13, 'res_39': res_39}

def run_experiment_e3(train_feats, test_feats, labels):
    """Thí nghiệm E3: Diagonal vs Full Covariance"""
    print("\n" + "="*50)
    print("[E3] KHẢO SÁT HIỆP PHƯƠNG SAI: DIAGONAL VS FULL COVARIANCE")
    print("="*50)

    # Diagonal
    res_diag = evaluate_hmm_pipeline(train_feats, test_feats, labels, covariance_type='diag')
    
    # Full (Thử nghiệm nếu đủ mẫu)
    try:
        res_full = evaluate_hmm_pipeline(train_feats, test_feats, labels, covariance_type='full')
        print(f" * Diagonal Covariance: Accuracy = {res_diag['accuracy']*100:.2f}%, Train Time = {res_diag['train_time']:.3f}s")
        print(f" * Full Covariance:     Accuracy = {res_full['accuracy']*100:.2f}%, Train Time = {res_full['train_time']:.3f}s")
    except Exception as e:
        print(f" * Full Covariance không hội tụ hoặc gặp lỗi số học: {e}")
        res_full = None

    return {'diag': res_diag, 'full': res_full}

def run_experiment_e4(train_feats, test_feats, labels, figures_dir='figures'):
    """Thí nghiệm E4: So sánh đối chứng HMM vs DTW"""
    print("\n" + "="*50)
    print("[E4] SO SÁNH ĐỐI CHỨNG DTW (LAB 2) VÀ HMM (LAB 3)")
    print("="*50)

    # 1. HMM (Lab 3)
    res_hmm = evaluate_hmm_pipeline(train_feats, test_feats, labels, n_states=5)

    # 2. DTW (Lab 2)
    dtw_rec = DTWRecognizer()
    start_train_dtw = time.time()
    dtw_rec.fit(train_feats)
    train_time_dtw = time.time() - start_train_dtw

    y_true_dtw, y_pred_dtw = [], []
    dtw_trial_times = []

    for label in labels:
        feats_list = test_feats.get(label, [])
        for F in feats_list:
            t0 = time.time()
            pred, _ = dtw_rec.predict(F)
            dtw_trial_times.append(time.time() - t0)
            y_true_dtw.append(label)
            y_pred_dtw.append(pred)

    acc_dtw = accuracy_score(y_true_dtw, y_pred_dtw)
    cm_dtw = confusion_matrix(y_true_dtw, y_pred_dtw, labels=labels)
    test_time_dtw = np.mean(dtw_trial_times)

    dtw_metrics = {
        'accuracy': acc_dtw,
        'cm': cm_dtw,
        'train_time': train_time_dtw,
        'test_time_per_trial': test_time_dtw
    }

    print(f" * DTW (Lab 2): Accuracy = {acc_dtw*100:.2f}%, Test time = {test_time_dtw*1000:.2f}ms/mẫu")
    print(f" * HMM (Lab 3): Accuracy = {res_hmm['accuracy']*100:.2f}%, Test time = {res_hmm['test_time_per_trial']*1000:.2f}ms/mẫu")

    plot_confusion_matrix(cm_dtw, labels, title="E4: Confusion Matrix - DTW Baseline",
                          save_path=f"{figures_dir}/e4_cm_dtw.png")
    plot_experiment_e4(dtw_metrics, res_hmm, save_path=f"{figures_dir}/e4_dtw_vs_hmm.png")

    return {'dtw': dtw_metrics, 'hmm': res_hmm}

def run_experiment_e5(labels, figures_dir='figures'):
    """Thí nghiệm E5: Speaker Mismatch (Người nói 1 vs Người nói 2)"""
    print("\n" + "="*50)
    print("[E5] KHẢO SÁT BIẾN THIỆN NGƯỜI NÓI (SPEAKER MISMATCH)")
    print("="*50)

    # 1. Matched: Train Spk1 -> Test Spk1
    spk1_train, _ = extract_dataset('dataset/train', labels)
    spk1_test, _ = extract_dataset('dataset/test', labels)
    res_matched = evaluate_hmm_pipeline(spk1_train, spk1_test, labels)

    # 2. Mismatched: Train Spk1 -> Test Spk2
    spk2_test, _ = extract_dataset('dataset_speaker2/test', labels)
    res_mismatched = evaluate_hmm_pipeline(spk1_train, spk2_test, labels)

    # 3. Multi-speaker: Train Spk1+Spk2 -> Test Spk1+Spk2
    multi_train, _ = extract_dataset('dataset_multi/train', labels)
    multi_test, _ = extract_dataset('dataset_multi/test', labels)
    res_multi = evaluate_hmm_pipeline(multi_train, multi_test, labels)

    print(f" * Matched Condition (Train Spk1 -> Test Spk1):       Accuracy = {res_matched['accuracy']*100:.2f}%")
    print(f" * Mismatched Condition (Train Spk1 -> Test Spk2):    Accuracy = {res_mismatched['accuracy']*100:.2f}% (Tụt { (res_matched['accuracy']-res_mismatched['accuracy'])*100:.2f}%)")
    print(f" * Multi-Speaker Condition (Train Cả 2 -> Test Cả 2): Accuracy = {res_multi['accuracy']*100:.2f}%")

    plot_confusion_matrix(res_mismatched['cm'], labels, title="E5: Confusion Matrix - Speaker Mismatch (Spk1 -> Spk2)",
                          save_path=f"{figures_dir}/e5_cm_mismatched.png")

    return {
        'matched': res_matched,
        'mismatched': res_mismatched,
        'multi': res_multi
    }

def run_experiment_e6(train_feats, test_feats, labels, figures_dir='figures'):
    """Thí nghiệm E6: Trọng số mô hình ngôn ngữ Bigram (lambda_LM)"""
    print("\n" + "="*50)
    print("[E6] TÁC ĐỘNG CỦA MÔ HÌNH NGÔN NGỮ (lambda_LM in {0, 0.1, 0.3, 1.0})")
    print("="*50)

    models = train_all_models(train_feats, n_states=5)
    corpus = get_sample_corpus(labels)
    c1, c2, vocab = build_ngram_counts(corpus)

    # Thử nghiệm với các giá trị lambda
    weights = [0.0, 0.1, 0.3, 1.0]
    
    # Chọn một mẫu test giả định từ đứng trước là 'khong'
    prev_word = 'khong'
    target_label = 'mot'
    test_sample = test_feats[target_label][0] if target_label in test_feats else list(test_feats.values())[0][0]

    print(f" * Khảo sát mẫu kiểm thử nhãn thực tế: '{target_label}', Từ ngữ cảnh đứng trước: '{prev_word}'")
    for lam in weights:
        best_w, totals, acoustics, lms = rerank(test_sample, prev_word, models, c2, c1, lm_weight=lam)
        print(f"   - lambda_LM = {lam:3.1f} -> Quyết định: '{best_w}' (Acoustic: {acoustics[best_w]:.2f}, LM: {lms[best_w]:.2f}, Total: {totals[best_w]:.2f})")

def run_rejection_analysis(models, test_feats, labels, figures_dir='figures'):
    """Phân tích phân phối điểm và đề xuất ngưỡng từ chối (Reject Threshold)"""
    print("\n" + "="*50)
    print("[ANALYSIS] PHÂN TÍCH PHÂN PHỐI ĐIỂM & ĐỀ XUẤT NGƯỠNG REJECT")
    print("="*50)

    correct_scores = []
    incorrect_scores = []

    for label in labels:
        feats_list = test_feats.get(label, [])
        for F in feats_list:
            T = len(F)
            for m_label, model in models.items():
                norm_score = model.score(F) / T
                if m_label == label:
                    correct_scores.append(norm_score)
                else:
                    incorrect_scores.append(norm_score)

    mu_corr = np.mean(correct_scores)
    std_corr = np.std(correct_scores)
    # Ngưỡng đề xuất: mean - 2*std của phân phối đúng lớp
    threshold = mu_corr - 2.0 * std_corr

    print(f" * Điểm trung bình đúng lớp (Correct class): {mu_corr:.2f} (std = {std_corr:.2f})")
    print(f" * Điểm trung bình sai lớp (Mismatched class): {np.mean(incorrect_scores):.2f}")
    print(f" * Ngưỡng Reject đề xuất (Threshold): {threshold:.2f}")

    plot_score_distributions(correct_scores, incorrect_scores, threshold=threshold,
                             save_path=f"{figures_dir}/rejection_score_distribution.png")

    return threshold
