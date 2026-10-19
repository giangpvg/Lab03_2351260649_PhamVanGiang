#!/usr/bin/env python3
"""
main.py
Chương trình thực thi toàn bộ luồng bài tập thực hành CSE457 - Lab 3:
1. Chuẩn bị và kiểm tra tập dữ liệu âm thanh
2. Trích xuất đặc trưng MFCC 39 chiều động (13 MFCC + 13 Delta + 13 Delta-Delta) + Utterance-level CMVN
3. Huấn luyện mô hình Left-to-Right Gaussian HMM (5 states, diagonal covariance)
4. Nhận dạng Maximum Likelihood trên tập kiểm thử và xuất Confusion Matrix
5. Căn chỉnh trạng thái Viterbi (Viterbi State Alignment) và vẽ đồ thị
6. Thực hiện toàn bộ 6 thí nghiệm bắt buộc (E1 đến E6)
7. So sánh đối chứng chuẩn xác giữa DTW (Lab 2) và HMM (Lab 3)
8. Lưu trữ toàn bộ mô hình vào 'models/', hình vẽ vào 'figures/', kết quả vào 'results.csv'
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

from prepare_data import organize_single_speaker, create_multi_speaker_dataset, LABELS_DIGITS
from feature_extractor import extract_feat, extract_dataset
from hmm_model import train_all_models, recognize, viterbi_align, save_models
from plot_utils import plot_confusion_matrix, plot_viterbi_alignment
from experiments import (
    run_experiment_e1, run_experiment_e2, run_experiment_e3,
    run_experiment_e4, run_experiment_e5, run_experiment_e6,
    run_rejection_analysis
)

def main():
    parser = argparse.ArgumentParser(description="CSE457 - Lab 3: Nhận dạng tiếng nói bằng Gaussian HMM")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Thư mục chứa dữ liệu chính")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Thư mục lưu đồ thị")
    parser.add_argument("--models-dir", type=str, default="models", help="Thư mục lưu mô hình")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    labels = LABELS_DIGITS

    print("="*70)
    print(" CSE457 • LAB 3: MÔ HÌNH HMM & NHẬN DẠNG TIẾNG NÓI TỪ ĐƠN")
    print(" Sinh viên: Phạm Văn Giang — MSSV: 2351260649")
    print("="*70)

    # 1. Trích xuất đặc trưng 39 chiều cho tập train và test
    train_dir = base_dir / args.data_dir / 'train'
    test_dir = base_dir / args.data_dir / 'test'

    print(f"\n[1] Trích xuất đặc trưng 39-D từ '{train_dir}' và '{test_dir}'...")
    train_feats, train_paths = extract_dataset(train_dir, labels, use_deltas=True, apply_cmvn=True)
    test_feats, test_paths = extract_dataset(test_dir, labels, use_deltas=True, apply_cmvn=True)

    print(f" - Số lớp từ vựng: {len(labels)}: {labels}")
    for lbl in labels:
        print(f"   * '{lbl:6s}': {len(train_feats[lbl])} train files, {len(test_feats[lbl])} test files")

    # 2. Huấn luyện Left-to-Right Gaussian HMM (5 states)
    print("\n[2] Huấn luyện Left-to-Right Gaussian HMM (5 states, diagonal covariance)...")
    models = train_all_models(train_feats, n_states=5, covariance_type='diag', n_iter=80)
    save_models(models, save_dir=args.models_dir)

    # 3. Đánh giá trên tập kiểm thử (Test Set)
    print("\n[3] Đánh giá nhận dạng Maximum Likelihood trên tập kiểm thử...")
    results_list = []
    y_true, y_pred = [], []

    for label in labels:
        feats_list = test_feats[label]
        paths_list = test_paths[label]
        for F, path in zip(feats_list, paths_list):
            pred, scores = recognize(F, models)
            y_true.append(label)
            y_pred.append(pred)

            # Lấy top 1 và top 2 score
            sorted_scores = sorted(scores.items(), key=lambda z: z[1], reverse=True)
            top1_label, top1_score = sorted_scores[0]
            top2_label, top2_score = sorted_scores[1]

            results_list.append({
                'filename': path.name,
                'true_label': label,
                'pred_label': pred,
                'top1_score': top1_score,
                'top2_score': top2_score,
                'score_gap': top1_score - top2_score,
                'is_correct': (label == pred)
            })

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("results.csv", index=False)

    print(f" => Test Accuracy: {acc * 100:.2f}% ({sum(df_results['is_correct'])}/{len(df_results)} mẫu chính xác)")
    print(f" => Đã lưu chi tiết từng file vào 'results.csv'.")

    # Vẽ ma trận nhầm lẫn chính
    plot_confusion_matrix(cm, labels, title="HMM Baseline (5 States, 39-D MFCC) Confusion Matrix",
                          save_path=f"{args.figures_dir}/confusion_matrix_hmm.png")

    # 4. Minh họa Viterbi State Alignment cho 2 file test
    print("\n[4] Trích xuất và vẽ biểu đồ Viterbi State Alignment...")
    for sample_label in ['khong', 'ba']:
        if sample_label in test_feats and len(test_feats[sample_label]) > 0:
            F_sample = test_feats[sample_label][0]
            wav_name = test_paths[sample_label][0].name
            logp, states = viterbi_align(F_sample, models[sample_label])
            print(f" - Mẫu '{wav_name}' ({sample_label.upper()}): {len(states)} frames, log-likelihood: {logp:.2f}")
            print(f"   Trạng thái qua từng frame: {states[:25]} ... {states[-10:]}")
            plot_viterbi_alignment(states, sample_label, wav_name,
                                  save_path=f"{args.figures_dir}/viterbi_{sample_label}.png")

    # 5. Thực hiện 6 thí nghiệm bắt buộc E1 - E6
    print("\n" + "#"*70)
    print(" TIẾN HÀNH TOÀN BỘ 6 THÍ NGHIỆM BẮT BUỘC (E1 - E6)")
    print("#"*70)

    # E1: Khảo sát số trạng thái
    res_e1 = run_experiment_e1(train_feats, test_feats, labels, figures_dir=args.figures_dir)

    # E2: 13-D tĩnh vs 39-D động
    res_e2 = run_experiment_e2(train_dir, test_dir, labels, figures_dir=args.figures_dir)

    # E3: Hiệp phương sai Diag vs Full
    res_e3 = run_experiment_e3(train_feats, test_feats, labels)

    # E4: So sánh đối chứng HMM vs DTW
    res_e4 = run_experiment_e4(train_feats, test_feats, labels, figures_dir=args.figures_dir)

    # E5: Mismatch người nói
    res_e5 = run_experiment_e5(labels, figures_dir=args.figures_dir)

    # E6: Mô hình ngôn ngữ Bigram re-ranking
    res_e6 = run_experiment_e6(train_feats, test_feats, labels, figures_dir=args.figures_dir)

    # Phân tích Reject Out-of-Vocabulary
    thresh = run_rejection_analysis(models, test_feats, labels, figures_dir=args.figures_dir)

    # 6. In bảng tổng kết so sánh DTW vs HMM theo đúng mục 5.8 trong Lab 3
    print("\n" + "="*70)
    print(" BẢNG TỔNG KẾT ĐỐI CHỨNG DTW (LAB 2) VS HMM (LAB 3)")
    print("="*70)
    dtw_m = res_e4['dtw']
    hmm_m = res_e4['hmm']
    summary_data = [
        ["Accuracy", f"{dtw_m['accuracy']*100:.1f}%", f"{hmm_m['accuracy']*100:.1f}%", "HMM mô hình hóa phân phối thống kê tốt hơn"],
        ["Số mẫu train/lớp", f"{len(train_feats[labels[0]])}", f"{len(train_feats[labels[0]])}", "HMM tận dụng được nhiều mẫu huấn luyện"],
        ["Thời gian train", f"{dtw_m['train_time']:.4f}s (Lưu template)", f"{hmm_m['train_time']:.3f}s (Baum-Welch EM)", "HMM train chậm hơn do lặp EM"],
        ["Thời gian test/trial", f"{dtw_m['test_time_per_trial']*1000:.2f} ms", f"{hmm_m['test_time_per_trial']*1000:.2f} ms", "HMM nhanh hơn khi số lượng template lớn"],
        ["Khả năng giải thích", "Warping path (khoảng cách Euclid)", "Viterbi state path (tiến trình âm học)", "HMM ánh xạ trực tiếp sang các pha âm thanh"]
    ]
    df_summary = pd.DataFrame(summary_data, columns=["Chỉ số", "DTW (Lab 2)", "HMM (Lab 3)", "Nhận xét"])
    print(df_summary.to_string(index=False))

    print("\n[HOÀN THÀNH] Toàn bộ quy trình Lab 3 đã thực thi xuất sắc 100%!")

if __name__ == '__main__':
    main()
