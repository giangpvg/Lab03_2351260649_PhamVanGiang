#!/usr/bin/env python3
"""
prepare_data.py
Chuẩn bị và phân chia tập dữ liệu cho CSE457 - Lab 3:
1. Phân chia dataset Speaker 1 thành:
   - dataset/train/<label>/ (các file _01, _02, _03)
   - dataset/test/<label>/  (các file _04, _05)
2. Phân chia dataset Speaker 2 thành:
   - dataset_speaker2/train/<label>/ (các file _01, _02, _03)
   - dataset_speaker2/test/<label>/  (các file _04, _05)
3. Tạo dataset kết hợp (Multi-speaker / 2 speakers):
   - dataset_multi/train/<label>/ (6 files/lớp: 3 spk1 + 3 spk2)
   - dataset_multi/test/<label>/  (4 files/lớp: 2 spk1 + 2 spk2)
"""

import os
import shutil
from pathlib import Path

LABELS_DIGITS = ['khong', 'mot', 'hai', 'ba', 'bon']
LABELS_COMMANDS = ['mo', 'dong', 'len', 'xuong', 'dung']

def organize_single_speaker(src_dir, dest_dir, labels, train_indices=(1, 2, 3), test_indices=(4, 5)):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    
    for label in labels:
        label_src = src_path / label
        if not label_src.exists():
            continue
            
        train_dst = dest_path / 'train' / label
        test_dst = dest_path / 'test' / label
        train_dst.mkdir(parents=True, exist_ok=True)
        test_dst.mkdir(parents=True, exist_ok=True)
        
        wavs = sorted(list(label_src.glob('*.wav')))
        for wav in wavs:
            # Lấy số thứ tự từ tên file (ví dụ ba_01.wav -> 1)
            try:
                idx = int(wav.stem.split('_')[-1])
            except ValueError:
                idx = 1
                
            if idx in train_indices:
                shutil.copy2(wav, train_dst / wav.name)
            elif idx in test_indices:
                shutil.copy2(wav, test_dst / wav.name)

def create_multi_speaker_dataset(spk1_dir, spk2_dir, multi_dir, labels):
    multi_path = Path(multi_dir)
    
    for label in labels:
        train_dst = multi_path / 'train' / label
        test_dst = multi_path / 'test' / label
        train_dst.mkdir(parents=True, exist_ok=True)
        test_dst.mkdir(parents=True, exist_ok=True)
        
        # Speaker 1: 01..03 train, 04..05 test
        s1_src = Path(spk1_dir) / label
        if s1_src.exists():
            for wav in sorted(s1_src.glob('*.wav')):
                try:
                    idx = int(wav.stem.split('_')[-1])
                except ValueError:
                    idx = 1
                if idx in (1, 2, 3):
                    shutil.copy2(wav, train_dst / f"spk1_{wav.name}")
                elif idx in (4, 5):
                    shutil.copy2(wav, test_dst / f"spk1_{wav.name}")
                    
        # Speaker 2: 01..03 train, 04..05 test
        s2_src = Path(spk2_dir) / label
        if s2_src.exists():
            for wav in sorted(s2_src.glob('*.wav')):
                try:
                    idx = int(wav.stem.split('_')[-1])
                except ValueError:
                    idx = 1
                if idx in (1, 2, 3):
                    shutil.copy2(wav, train_dst / f"spk2_{wav.name}")
                elif idx in (4, 5):
                    shutil.copy2(wav, test_dst / f"spk2_{wav.name}")

def main():
    base_dir = Path(__file__).resolve().parent
    print("[PREPARE_DATA] Bắt đầu chuẩn bị các tập dữ liệu...")
    
    # 1. Chuẩn bị cho Speaker 1 (thư mục dataset/)
    organize_single_speaker(base_dir / 'dataset', base_dir / 'dataset', LABELS_DIGITS)
    print(" - Đã chia dataset/ thành dataset/train và dataset/test.")
    
    # 2. Chuẩn bị cho Speaker 2 (thư mục dataset_speaker2/)
    organize_single_speaker(base_dir / 'dataset_speaker2', base_dir / 'dataset_speaker2', LABELS_DIGITS)
    print(" - Đã chia dataset_speaker2/ thành train và test.")
    
    # 3. Tạo tập multi-speaker (dataset_multi/)
    create_multi_speaker_dataset(base_dir / 'dataset', base_dir / 'dataset_speaker2', base_dir / 'dataset_multi', LABELS_DIGITS)
    print(" - Đã tạo tập kết hợp dataset_multi/ (6 train + 4 test mỗi lớp).")

    # In thống kê
    for dset, name in [(base_dir / 'dataset', 'Speaker 1'), 
                       (base_dir / 'dataset_speaker2', 'Speaker 2'),
                       (base_dir / 'dataset_multi', 'Multi-Speaker')]:
        train_count = len(list((dset / 'train').glob('*/*.wav')))
        test_count = len(list((dset / 'test').glob('*/*.wav')))
        print(f"   * {name}: {train_count} train files, {test_count} test files.")

if __name__ == '__main__':
    main()
