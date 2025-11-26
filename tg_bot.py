import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

BOT_TOKEN = "8372291937:AAEH3qtfCaBHYUwuMCzcXIh-8-2A6VLF3Rc"
API_URL = "http://127.0.0.1:8000"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --------------------------
# КЛАВИАТУРЫ
# --------------------------
main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📝 Регистрация")],
        [KeyboardButton(text="🔍 Проверка")]
    ],
    resize_keyboard=True
)

side_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="✋ Левая")],
        [KeyboardButton(text="🤚 Правая")]
    ],
    resize_keyboard=True
)


# --------------------------
# Состояния
# --------------------------
dp["mode"] = None     # "register", "verify"
dp["user_id"] = None  # ID пользователя
dp["side"] = None     # left / right


# --------------------------
# Команда /start
# --------------------------
@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    dp["mode"] = None
    dp["side"] = None
    dp["user_id"] = None

    await message.answer(
        "Привет! Я бот для распознавания ладони.\nВыберите действие:",
        reply_markup=main_kb
    )


# --------------------------
# Нажата кнопка "Регистрация"
# --------------------------
@dp.message(lambda m: m.text == "📝 Регистрация")
async def register_start(message: types.Message):
    dp["mode"] = "register"
    await message.answer("Введите ID пользователя:", reply_markup=types.ReplyKeyboardRemove())


# --------------------------
# Нажата кнопка "Проверка"
# --------------------------
@dp.message(lambda m: m.text == "🔍 Проверка")
async def verify_start(message: types.Message):
    dp["mode"] = "verify"
    await message.answer("Отправьте фото ладони для проверки.")


# --------------------------
# Ввод user_id
# --------------------------
@dp.message(lambda m: dp.get("mode") == "register" and dp.get("user_id") is None)
async def get_user_id(message: types.Message):
    dp["user_id"] = message.text.strip()
    await message.answer(
        f"ID принят: {dp['user_id']}\nВыберите руку:",
        reply_markup=side_kb
    )


# --------------------------
# Выбор левой / правой руки
# --------------------------
@dp.message(lambda m: m.text in ["✋ Левая", "🤚 Правая"])
async def choose_side(message: types.Message):
    if message.text == "✋ Левая":
        dp["side"] = "left"
    else:
        dp["side"] = "right"

    await message.answer(
        f"Выбрана {dp['side']} рука.\nТеперь отправьте фото ладони.",
        reply_markup=types.ReplyKeyboardRemove()
    )


# --------------------------
# Пришло фото
# --------------------------
@dp.message(lambda m: m.photo)
async def process_photo(message: types.Message):
    mode = dp.get("mode")

    # скачиваем фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_bytes = await bot.download_file(file.file_path)

    # регистрация ------------------------------
    if mode == "register":
        user_id = dp.get("user_id")
        side = dp.get("side")

        if not user_id or not side:
            await message.answer("Ошибка: не выбран user_id или сторона руки.")
            return

        url = f"{API_URL}/register/{user_id}?side={side}"

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", img_bytes.read(), filename="hand.jpg", content_type="image/jpeg")

            async with session.post(url, data=form) as resp:
                result = await resp.json()

        await message.answer(
            f"📌 Регистрация завершена:\n<code>{result}</code>",
            parse_mode="HTML",
            reply_markup=main_kb
        )

        # сброс состояний
        dp["mode"] = None
        dp["user_id"] = None
        dp["side"] = None
        return

    # проверка ------------------------------
    elif mode == "verify":
        url = f"{API_URL}/verify"

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", img_bytes.read(), filename="hand.jpg", content_type="image/jpeg")

            async with session.post(url, data=form) as resp:
                result = await resp.json()

        await message.answer(
            f"🔍 Результат:\n<code>{result}</code>",
            parse_mode="HTML",
            reply_markup=main_kb
        )

        dp["mode"] = None
        return

    else:
        await message.answer("Выберите действие: Регистрация или Проверка.")
        return


# --------------------------
# ЗАПУСК БОТА
# --------------------------
async def main():
    print("Bot started!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
