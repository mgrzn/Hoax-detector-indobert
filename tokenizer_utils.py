import re
import os
import pandas as pd
from transformers import AutoTokenizer

# Kata kunci yang menandakan berita bukan hoax (opsional dipakai di preprocessing dataset)
NON_HOAX_TAGS = ['berita', 'fakta', 'klarifikasi', 'benar', 'cek fakta']

def contains_non_hoax_tag(text):
    """Deteksi apakah teks punya tag non-hoax di awal, misalnya [BERITA]"""
    if pd.isna(text):
        return False
    match = re.match(r'[\[\(]\s*([^\]\)]+?)\s*[\]\)]', text, flags=re.IGNORECASE)
    if match:
        tag = match.group(1).strip().lower()
        return tag in NON_HOAX_TAGS
    return False

def clean_text(text: str) -> str:
    """Membersihkan teks dari tag, link, dan karakter non-alfabet"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'^[\[\(]\s*[^)\]]+\s*[\)\]]', '', text)  # hapus tag di awal
    text = re.sub(r'http\S+|www\S+', '', text)              # hapus link
    text = re.sub(r'[^a-z\s]', ' ', text)                   # hanya huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip()                # hapus spasi berlebih
    return text

def load_tokenizer(tokenizer_path: str):
    """Load tokenizer dari folder hasil save_pretrained"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer folder '{tokenizer_path}' tidak ditemukan.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer
