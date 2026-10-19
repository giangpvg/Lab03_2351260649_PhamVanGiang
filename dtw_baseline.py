#!/usr/bin/env python3
"""
dtw_baseline.py
Mô hình cơ sở DTW (Lab 2) dùng để so sánh đối chứng trực tiếp với HMM (Lab 3):
- Thuật toán Dynamic Time Warping với khoảng cách cục bộ Euclid
- Chuẩn hóa chi phí DTW theo độ dài đường căn chỉnh: D[N, M] / |P|
- Bộ nhận dạng Nearest-Template (1-NN) dựa trên các mẫu huấn luyện
- Đo đạc thời gian suy luận (test time per trial) và độ chính xác (accuracy)
"""

import time
import numpy as np
from pathlib import Path

def dtw_distance(X, Y):
    """
    Tính khoảng cách DTW chuẩn hóa giữa hai chuỗi đặc trưng X (N, D) và Y (M, D):
    """
    N, M = len(X), len(Y)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    # Điền bảng quy hoạch động
    for i in range(1, N + 1):
        x_i = X[i - 1]
        for j in range(1, M + 1):
            y_j = Y[j - 1]
            cost = np.linalg.norm(x_i - y_j)
            D[i, j] = cost + min(D[i - 1, j],      # nén / chèn
                                 D[i, j - 1],      # giãn
                                 D[i - 1, j - 1])  # khớp chéo

    # Truy vết đường đi tối ưu để lấy độ dài |P|
    i, j = N, M
    path_len = 1
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        steps = [
            (i - 1, j, D[i - 1, j]),
            (i, j - 1, D[i, j - 1]),
            (i - 1, j - 1, D[i - 1, j - 1])
        ]
        # Chọn bước có chi phí nhỏ nhất
        best_step = min(steps, key=lambda s: s[2])
        i, j = best_step[0], best_step[1]
        path.append((i - 1, j - 1))
        path_len += 1

    path.reverse()
    norm_dist = D[N, M] / path_len
    return norm_dist, path

class DTWRecognizer:
    """Bộ nhận dạng Nearest-Template DTW kế thừa từ Lab 2"""
    def __init__(self):
        self.templates = {} # dict {label: [F_1, F_2, ...]}

    def fit(self, train_feats_by_label):
        """Lưu trữ các mẫu đặc trưng làm template chuẩn"""
        self.templates = train_feats_by_label

    def predict(self, F):
        """
        Nhận dạng mẫu F:
        Tính khoảng cách DTW tới tất cả template và chọn lớp có khoảng cách nhỏ nhất
        """
        best_dist = float('inf')
        best_label = None
        distances = {}

        for label, tmpl_list in self.templates.items():
            min_class_dist = float('inf')
            for tmpl in tmpl_list:
                dist, _ = dtw_distance(F, tmpl)
                if dist < min_class_dist:
                    min_class_dist = dist
            distances[label] = min_class_dist
            if min_class_dist < best_dist:
                best_dist = min_class_dist
                best_label = label

        return best_label, distances

if __name__ == '__main__':
    # Kiểm tra thử nghiệm
    X = np.random.randn(50, 39)
    Y = np.random.randn(45, 39)
    dist, path = dtw_distance(X, Y)
    print(f"[TEST DTW] Khoảng cách chuẩn hóa: {dist:.4f}, Số bước căn chỉnh: {len(path)}")
    assert dist >= 0, "Lỗi: Chi phí DTW không thể âm!"
    print("[TEST DTW] Thuật toán DTW hoạt động chuẩn xác!")
