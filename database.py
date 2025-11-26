import json
import numpy as np
from pathlib import Path

DB_FILE = Path("embeddings.json")


def load_db():
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text())
    return {}


def save_db(data):
    DB_FILE.write_text(json.dumps(data, indent=4))


def save_embedding(user_id, emb, side):
    """
    Сохраняет embedding в БД для нужной стороны ("left" или "right").
    Формат:
    {
        "user123": {
            "left": [...],
            "right": [...]
        }
    }
    """
    db = load_db()

    if user_id not in db:
        db[user_id] = {"left": None, "right": None}

    db[user_id][side] = emb.tolist()

    save_db(db)


def find_best_match(emb, side, threshold=0.75):
    """
    Ищет совпадение embeddings только по указанной стороне.
    """
    db = load_db()

    best_user = None
    best_score = 0

    for user_id, record in db.items():

        # Если у пользователя нет embedding для этого side — пропускаем
        side_emb = record.get(side)
        if side_emb is None:
            continue

        vec = np.array(side_emb)
        sim = float(np.dot(vec, emb))  # cosine similarity for L2-normalized vectors

        if sim > best_score:
            best_score = sim
            best_user = user_id

    if best_score > threshold:
        return {"user": best_user, "score": best_score}

    return {"user": None, "score": best_score}
