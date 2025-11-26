from fastapi import FastAPI, UploadFile, File, Query
from detector import extract_palm_roi
from embedder import get_embedding
from database import save_embedding, find_best_match

app = FastAPI()


@app.get("/")
def root():
    return {"status": "Palm Recognition API running", "version": "2-hand-v1"}


# -------------------------------
# РЕГИСТРАЦИЯ
# -------------------------------
@app.post("/register/{user_id}")
async def register(
    user_id: str,
    side: str = Query(..., regex="^(left|right)$"),
    file: UploadFile = File(...)
):
    """
    Пример вызова:
    /register/user123?side=left
    /register/user123?side=right
    """
    img_bytes = await file.read()

    roi, detected_side = extract_palm_roi(img_bytes)

    if roi is None:
        return {"status": "error", "message": "Palm not detected"}

    # Сторону выбирает человек (внешний параметр)
    # detected_side можно использовать для контроля
    emb = get_embedding(roi)

    save_embedding(user_id, emb, side)

    return {
        "status": "ok",
        "user_id": user_id,
        "saved_side": side,
        "detected_side": detected_side
    }


# -------------------------------
# ВЕРИФИКАЦИЯ
# -------------------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    img_bytes = await file.read()

    roi, side = extract_palm_roi(img_bytes)

    if roi is None:
        return {"status": "error", "message": "Palm not detected"}

    emb = get_embedding(roi)

    match = find_best_match(emb, side)

    return {
        "detected_side": side,
        "result": match
    }
