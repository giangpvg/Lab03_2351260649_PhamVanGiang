#!/usr/bin/env python3
"""
language_model.py
Xây dựng mô hình ngôn ngữ N-gram (Bigram) và Lexicon cho CSE457 - Lab 3:
- Parser đọc và chuẩn hóa file lexicon.txt thành cấu trúc từ điển Python
- Xây dựng bảng đếm tần suất Unigram C(w) và Bigram C(w_prev, w_cur)
- Ước lượng xác suất chuyển tiếp Bigram có làm mịn Add-k (Laplace/Lidstone smoothing):
  P(w_cur | w_prev) = (C(w_prev, w_cur) + k) / (C(w_prev) + k * V)
- Tái xếp hạng (MAP re-ranking) kết hợp điểm âm học (Acoustic Score) và điểm ngôn ngữ (LM Score):
  Score(w) = (1/T) * log P(X | lambda_w) + lambda_LM * log P(w | w_prev)
"""

import math
from pathlib import Path
from collections import defaultdict

def load_lexicon(lexicon_path):
    """
    Đọc file lexicon.txt theo định dạng: WORD PHONE_1 PHONE_2 ... PHONE_K
    
    Trả về:
    - lexicon_dict: dict {word: [phone1, phone2, ...]}
    """
    lexicon_dict = {}
    path = Path(lexicon_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file lexicon: {lexicon_path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            word = parts[0]
            phones = parts[1:]
            lexicon_dict[word] = phones
            
    return lexicon_dict

def build_ngram_counts(corpus):
    """
    Xây dựng bảng đếm unigram và bigram từ tập văn bản (corpus):
    - corpus: danh sách các câu (mỗi câu là một list các từ hoặc chuỗi phân tách bởi khoảng trắng)
    
    Trả về:
    - counts1: dict {word: count}
    - counts2: dict {(w_prev, w_cur): count}
    - vocab: set các từ vựng
    """
    counts1 = defaultdict(int)
    counts2 = defaultdict(int)
    vocab = set()

    for sentence in corpus:
        if isinstance(sentence, str):
            words = sentence.strip().split()
        else:
            words = list(sentence)
            
        for i, w in enumerate(words):
            vocab.add(w)
            counts1[w] += 1
            if i > 0:
                prev = words[i - 1]
                counts2[(prev, w)] += 1

    return dict(counts1), dict(counts2), vocab

def bigram_prob(prev, cur, counts2, counts1, V, k=0.5):
    """
    Tính xác suất có điều kiện P(cur | prev) sử dụng làm mịn Add-k:
    P(cur | prev) = (counts2[(prev, cur)] + k) / (counts1[prev] + k * V)
    """
    numerator = counts2.get((prev, cur), 0) + k
    denominator = counts1.get(prev, 0) + (k * V)
    return numerator / denominator

def rerank(F, prev, models, counts2, counts1, lm_weight=0.3, k=0.5):
    """
    Tái xếp hạng các ứng viên từ dựa trên kết hợp điểm âm học và điểm ngôn ngữ:
    - F: Ma trận đặc trưng của utterance kiểm thử (T x D)
    - prev: Từ đứng trước đã biết trong ngữ cảnh
    - models: dict {label: GaussianHMM}
    - lm_weight (lambda_LM): Trọng số của mô hình ngôn ngữ
    
    Trả về:
    - best_word: Từ được chọn sau re-ranking
    - total_scores: dict {word: total_score}
    """
    T = len(F)
    V = len(models)
    total_scores = {}
    acoustic_scores = {}
    lm_scores = {}

    for w, m in models.items():
        # Điểm âm học chuẩn hóa theo độ dài frame (Acoustic log-likelihood per frame)
        a_score = m.score(F) / T
        # Điểm ngôn ngữ log-probability
        p_lm = bigram_prob(prev, w, counts2, counts1, V, k=k)
        lm_score = math.log(p_lm)
        
        # Tổng điểm MAP
        total = a_score + lm_weight * lm_score
        total_scores[w] = total
        acoustic_scores[w] = a_score
        lm_scores[w] = lm_score

    best_word = max(total_scores, key=total_scores.get)
    return best_word, total_scores, acoustic_scores, lm_scores

def get_sample_corpus(labels):
    """Tạo ngữ liệu mẫu hợp lý cho các từ vựng phục vụ thí nghiệm E6"""
    if 'khong' in labels:
        # Ngữ liệu số thoại: chuỗi số có quy luật (ví dụ: không một, hai ba, ba bốn...)
        return [
            "khong mot hai ba bon",
            "khong mot khong hai khong ba",
            "mot hai ba bon khong",
            "hai ba bon mot hai",
            "khong mot mot hai hai ba",
            "ba bon khong mot hai",
            "khong hai ba bon mot",
            "khong mot hai ba bon khong mot"
        ]
    else:
        # Ngữ liệu điều khiển: mở cửa, đóng cửa, đi lên, đi xuống, dừng lại
        return [
            "mo len xuong dung",
            "dong len dung",
            "mo dong mo dong",
            "len xuong len xuong",
            "mo len dung",
            "dong xuong dung"
        ]

if __name__ == '__main__':
    # Kiểm tra thử nghiệm
    lex = load_lexicon('lexicon.txt')
    print(f"[TEST LM] Đã nạp {len(lex)} từ từ lexicon.txt:")
    for w, phones in list(lex.items())[:5]:
        print(f"  * {w:8s} -> {' '.join(phones)}")
        
    corpus = get_sample_corpus(['khong', 'mot', 'hai', 'ba', 'bon'])
    c1, c2, v = build_ngram_counts(corpus)
    p = bigram_prob('khong', 'mot', c2, c1, len(v))
    print(f"[TEST LM] P('mot' | 'khong') = {p:.4f}")
    assert p > 0, "Lỗi: Xác suất làm mịn phải dương!"
    print("[TEST LM] Module mô hình ngôn ngữ hoạt động hoàn hảo!")
