#!/usr/bin/env python3
import os
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ====================================================
# CONFIG
# ====================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "fingerprint.db"
TEMPLATE_DIR = BASE_DIR / "templates"
KEY_PATH = BASE_DIR / "aes_key.bin"

TEMPLATE_DIR.mkdir(exist_ok=True)

EMBEDDING_DIM = 256        # размер эмбеддинга
RANDOM_SEED = 42           # для генерации матрицы проекции
np.random.seed(RANDOM_SEED)

# случайная матрица для Random Projection
R = np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM)

def preprocess_fingerprint(img_path: str) -> Image.Image:
    """
    Улучшенная предобработка отпечатка:
    - CLAHE
    - нормализация
    - Gabor filter
    - upsample
    """

    # Загружаем как grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1. Выравнивание гистограммы (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 2. Увеличение размера
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    # 3. Gabor Filter (усиление линий риджей)
    def apply_gabor(img):
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        accum = np.zeros_like(img, dtype=np.float32)

        for theta in thetas:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=3.0,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0
            )
            fimg = cv2.filter2D(img, cv2.CV_32F, kernel)
            accum = np.maximum(accum, fimg)

        accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
        return accum.astype(np.uint8)

    img = apply_gabor(img)

    # 4. Сглаживание шума
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Конвертация обратно в PIL → RGB
    pil = Image.fromarray(img).convert("RGB")
    return pil


# ====================================================
# AES-ключ и функции шифрования
# ===================================================

def load_or_create_key() -> bytes:
    """
    Загружаем AES-ключ из файла или создаём новый.
    Используем 256-битный ключ для AES-GCM.
    """
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    key = AESGCM.generate_key(bit_length=256)
    KEY_PATH.write_bytes(key)
    return key


AES_KEY = load_or_create_key()


def encrypt_bytes(data: bytes) -> bytes:
    """
    Шифрование данных: nonce (12 байт) + ciphertext.
    """
    aesgcm = AESGCM(AES_KEY)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, data, None)
    return nonce + ct


def decrypt_bytes(enc: bytes) -> bytes:
    """
    Расшифровка данных: отделяем nonce (12 байт) и ciphertext.
    """
    aesgcm = AESGCM(AES_KEY)
    nonce = enc[:12]
    ct = enc[12:]
    return aesgcm.decrypt(nonce, ct, None)


# ====================================================
# СЕТЬ: предобученная CNN → embedding extractor
# ====================================================

class FingerprintEmbedder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # заменим последний классификатор на Identity
        base.classifier = nn.Identity()
        self.base = base

        # слой проекции в эмбеддинг
        self.fc = nn.Sequential(
            nn.Linear(1280, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


device = torch.device("cpu")
model = FingerprintEmbedder(EMBEDDING_DIM).to(device)
model.eval()

# трансформации для изображений отпечатков
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])



# ====================================================
# EMBEDDINGS + RANDOM PROJECTION
# ====================================================

def image_to_embedding(img_path: str) -> np.ndarray:
    """Преобразование изображения → embedding (numpy array) с улучшенной предобработкой."""
    
    # Улучшенная обработка (CLAHE + Gabor + resize)
    img = preprocess_fingerprint(img_path)

    # Преобразуем в torch tensor
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy()[0]

    return emb


def cancelable_template(embedding: np.ndarray) -> np.ndarray:
    """
    Анонимизация: z = R * embedding
    """
    return R.dot(embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ====================================================
# DATABASE
# ====================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            template_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            result TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_employee(employee_id: str, template_path: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO employees (employee_id, template_path, created_at)
        VALUES (?, ?, ?)
    """, (employee_id, template_path, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


def list_employees():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT employee_id, created_at FROM employees ORDER BY employee_id")
    rows = cur.fetchall()
    conn.close()
    return rows


def load_employee_template(employee_id: str) -> np.ndarray | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT template_path FROM employees WHERE employee_id=?", (employee_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    enc_data = Path(row[0]).read_bytes()
    raw_data = decrypt_bytes(enc_data)

    import io
    buf = io.BytesIO(raw_data)
    z = np.load(buf, allow_pickle=False)
    return z


def log_access(employee_id: str, result: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO access_logs (employee_id, timestamp, result) VALUES (?, ?, ?)",
        (employee_id, datetime.utcnow().isoformat(), result),
    )
    conn.commit()
    conn.close()


def list_logs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT employee_id, timestamp, result FROM access_logs "
        "ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


# ====================================================
# MAIN LOGIC: REGISTRATION + IDENTIFICATION
# ====================================================

def register_employee(employee_id: str, image_path: str):
    """
    Регистрация сотрудника:
      - embedding
      - random projection
      - шифрование AES
      - сохранение пути в БД
    """
    init_db()

    emb = image_to_embedding(image_path)
    z = cancelable_template(emb)

    # сериализуем numpy-массив в байты
    import io
    buf = io.BytesIO()
    np.save(buf, z, allow_pickle=False)
    raw_data = buf.getvalue()

    enc_data = encrypt_bytes(raw_data)

    template_path = TEMPLATE_DIR / f"{employee_id}.bin"
    template_path.write_bytes(enc_data)

    save_employee(employee_id, str(template_path))
    print(f"[OK] Employee '{employee_id}' registered. Template saved at {template_path}")


def verify_employee(employee_id: str, image_path: str, threshold=0.60) -> bool:
    """
    Проверка отпечатка:
      - загрузка и расшифровка шаблона сотрудника
      - embedding нового изображения
      - random projection
      - косинусное сходство
    """
    init_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT template_path FROM employees WHERE employee_id=?", (employee_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        print("[ERR] Employee not found")
        return False

    template_path = Path(row[0])
    enc_data = template_path.read_bytes()
    raw_data = decrypt_bytes(enc_data)

    import io
    buf = io.BytesIO(raw_data)
    z_db = np.load(buf, allow_pickle=False)

    emb_new = image_to_embedding(image_path)
    z_new = cancelable_template(emb_new)

    sim = cosine_similarity(z_db, z_new)
    print(f"[INFO] Similarity = {sim:.4f}")

    if sim >= threshold:
        print("[ACCESS] GRANTED")
        log_access(employee_id, f"ALLOW sim={sim:.4f}")
        return True
    else:
        print("[ACCESS] DENIED")
        log_access(employee_id, f"DENY sim={sim:.4f}")
        return False


# ====================================================
# CLI
# ====================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Register: python fingerprint_system.py register <EMP_ID> <image>")
        print("  Verify:   python fingerprint_system.py verify <EMP_ID> <image>")
        return

    cmd = sys.argv[1]

    if cmd == "register" and len(sys.argv) == 4:
        _, _, emp_id, img = sys.argv
        register_employee(emp_id, img)

    elif cmd == "verify" and len(sys.argv) == 4:
        _, _, emp_id, img = sys.argv
        verify_employee(emp_id, img)

    else:
        print("Invalid command")


if __name__ == "__main__":
    main()
