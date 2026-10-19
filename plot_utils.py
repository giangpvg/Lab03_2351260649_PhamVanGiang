#!/usr/bin/env python3
"""
plot_utils.py
Vẽ đồ thị và trực quan hóa kết quả thực nghiệm cho CSE457 - Lab 3:
1. Ma trận nhầm lẫn (Confusion Matrix) với bảng màu chuẩn Blues
2. Biểu đồ căn chỉnh trạng thái Viterbi (Viterbi State Path) dạng step plot
3. Biểu đồ phân phối điểm Log-likelihood (Đúng lớp vs Sai lớp) xác định ngưỡng Reject
4. Biểu đồ so sánh các thí nghiệm bắt buộc (E1, E2, E3, E4, E5, E6)
Toàn bộ hình vẽ được xuất ra định dạng PNG chất lượng cao (300 DPI) vào thư mục 'figures/'.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

# Thiết lập font và style vẽ đồ thị
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None, cmap='Blues'):
    """Vẽ và lưu ma trận nhầm lẫn"""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=ax, colorbar=True, values_format='d')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Dự đoán (Predicted Label)", fontsize=10, labelpad=8)
    ax.set_ylabel("Thực tế (True Label)", fontsize=10, labelpad=8)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[PLOT] Đã lưu ma trận nhầm lẫn: {save_path}")
    plt.close()

def plot_viterbi_alignment(states, word_label, utterance_name, save_path=None):
    """
    Vẽ biểu đồ căn chỉnh trạng thái Viterbi theo thời gian:
    - Trục hoành: Khung thời gian (Frame t)
    - Trục tung: Chỉ số trạng thái ẩn (State q_t)
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    frames = range(len(states))
    ax.step(frames, states, where='post', color='#1f77b4', linewidth=2.5, label='Viterbi state path')
    
    # Đánh dấu các điểm chuyển trạng thái
    transitions = []
    for t in range(1, len(states)):
        if states[t] != states[t-1]:
            transitions.append((t, states[t]))
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1.0)
            
    ax.set_xlabel('Khung thời gian (Frame t - mỗi frame 10ms)', fontsize=11, labelpad=8)
    ax.set_ylabel('Chỉ số trạng thái ẩn (State)', fontsize=11, labelpad=8)
    ax.set_title(f'Viterbi State Alignment: {utterance_name} vs HMM [{word_label.upper()}]', fontsize=12, fontweight='bold', pad=12)
    ax.set_yticks(range(max(states) + 1))
    ax.set_yticklabels([f'State {i}' for i in range(max(states) + 1)])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[PLOT] Đã lưu Viterbi alignment: {save_path}")
    plt.close()

def plot_score_distributions(correct_scores, incorrect_scores, threshold=None, save_path=None):
    """
    Vẽ phân phối điểm Log-likelihood/frame để minh họa cơ chế Reject (Từ ngoài từ điển)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bins = np.linspace(min(min(correct_scores), min(incorrect_scores)),
                       max(max(correct_scores), max(incorrect_scores)), 25)
    
    ax.hist(correct_scores, bins=bins, alpha=0.65, color='#2ca02c', label='Đúng lớp (In-vocabulary / Correct)', edgecolor='black')
    ax.hist(incorrect_scores, bins=bins, alpha=0.65, color='#d62728', label='Sai lớp / Khác lớp (Mismatched)', edgecolor='black')
    
    if threshold is not None:
        ax.axvline(x=threshold, color='purple', linestyle='--', linewidth=2.0, label=f'Ngưỡng Reject (T={threshold:.1f})')
        
    ax.set_xlabel('Normalized Acoustic Score / Frame (log P(X|λ) / T)', fontsize=11, labelpad=8)
    ax.set_ylabel('Số lượng mẫu (Frequency)', fontsize=11, labelpad=8)
    ax.set_title('Phân phối điểm âm học & Ngưỡng từ chối (Reject Threshold)', fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[PLOT] Đã lưu biểu đồ phân phối điểm: {save_path}")
    plt.close()

def plot_experiment_e1(state_results, save_path=None):
    """Vẽ biểu đồ khảo sát số lượng state E1 (n_states = 3, 5, 7)"""
    states = [res['n_states'] for res in state_results]
    accs = [res['accuracy'] * 100 for res in state_results]
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    bars = ax.bar([str(s) for s in states], accs, color=['#4c72b0', '#55a868', '#c44e52'], width=0.45, edgecolor='black')
    ax.set_xlabel('Số lượng trạng thái ẩn (n_states)', fontsize=11, labelpad=8)
    ax.set_ylabel('Độ chính xác Test Accuracy (%)', fontsize=11, labelpad=8)
    ax.set_title('Thí nghiệm E1: Ảnh hưởng của số trạng thái ẩn đến Accuracy', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim([0, 115])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 2, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[PLOT] Đã lưu biểu đồ E1: {save_path}")
    plt.close()

def plot_experiment_e4(dtw_metrics, hmm_metrics, save_path=None):
    """Vẽ biểu đồ so sánh DTW vs HMM (Accuracy, Tốc độ suy luận)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    # 1. So sánh Accuracy
    models = ['DTW (Lab 2)', 'HMM (Lab 3)']
    accs = [dtw_metrics['accuracy'] * 100, hmm_metrics['accuracy'] * 100]
    bars1 = ax1.bar(models, accs, color=['#e67e22', '#2980b9'], width=0.45, edgecolor='black')
    ax1.set_ylabel('Độ chính xác (%)', fontsize=11)
    ax1.set_title('So sánh Độ chính xác (Accuracy)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 115])
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 2, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # 2. So sánh Thời gian suy luận / mẫu
    times_ms = [dtw_metrics['test_time_per_trial'] * 1000, hmm_metrics['test_time_per_trial'] * 1000]
    bars2 = ax2.bar(models, times_ms, color=['#e67e22', '#2980b9'], width=0.45, edgecolor='black')
    ax2.set_ylabel('Thời gian test / mẫu (ms)', fontsize=11)
    ax2.set_title('Thời gian nhận dạng trung bình', fontsize=11, fontweight='bold')
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.5, f'{h:.2f} ms', ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    plt.suptitle('Thí nghiệm E4: So sánh đối chứng DTW (Lab 2) và HMM (Lab 3)', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[PLOT] Đã lưu biểu đồ E4: {save_path}")
    plt.close()
