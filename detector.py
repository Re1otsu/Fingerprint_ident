import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands

def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

def extract_palm_roi(image_bytes):
    """
    Возвращает:
      roi  — нормализованная ладонь 200x200
      side — строка "left" или "right"
    """

    # Декодируем изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w, _ = img.shape

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None, None

        # -------------- определение стороны руки ---------------------
        handedness = results.multi_handedness[0].classification[0].label.lower()

        # нормализуем имена
        if handedness == "left":
            side = "left"
        else:
            side = "right"

        lm = results.multi_hand_landmarks[0].landmark

        # -------- 1. Вычисление координат --------
        xs = np.array([p.x * w for p in lm])
        ys = np.array([p.y * h for p in lm])

        # -------- 2. Угол ладони --------
        wrist = lm[0]
        mid = lm[9]

        x1, y1 = wrist.x * w, wrist.y * h
        x2, y2 = mid.x * w, mid.y * h

        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = -(angle_rad * 180.0 / math.pi - 90)

        # -------- 3. Поворот изображения --------
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        rotated = rotate_image(img, angle_deg, center)

        # -------- 4. Bounding box после поворота --------
        xs_rot = []
        ys_rot = []

        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        for p in lm:
            xp = int(p.x * w)
            yp = int(p.y * h)
            vec = np.array([xp, yp, 1])
            new = np.dot(rot_mat, vec)
            xs_rot.append(new[0])
            ys_rot.append(new[1])

        x_min, x_max = int(min(xs_rot)), int(max(xs_rot))
        y_min, y_max = int(min(ys_rot)), int(max(ys_rot))

        # -------- 5. Padding --------
        pad_y = int(0.3 * (y_max - y_min))
        pad_x = int(0.2 * (x_max - x_min))

        y_min = max(0, y_min - pad_y)
        y_max = min(rotated.shape[0], y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(rotated.shape[1], x_max + pad_x)

        # -------- 6. Вырезаем ROI --------
        roi = rotated[y_min:y_max, x_min:x_max]

        # -------- 7. Resize 200x200 --------
        roi = cv2.resize(roi, (200, 200))

        return roi, side
