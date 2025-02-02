import sqlite3
import random
from datetime import datetime, timedelta
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from sklearn.linear_model import LinearRegression
import numpy as np

TOKEN = "YOUR_BOT_TOKEN"  # Твой токен
ADMIN_CHAT_IDS = ["ADMIN_CHAT_ID1", "ADMIN_CHAT_ID2"]  # Список ID администраторов

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Инициализация базы данных SQLite
conn = sqlite3.connect('orders.db')
cursor = conn.cursor()

# Создаем таблицы, если они не существуют
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    date TEXT,
    dessert TEXT,
    quantity INTEGER
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS desserts (
    name TEXT UNIQUE
)
""")
conn.commit()

# Клавиатура с выбором десертов и кнопкой возврата
menu_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🧁 Капкейки"), KeyboardButton(text="🍰 Торты")],
        [KeyboardButton(text="🍪 Печенье"), KeyboardButton(text="🥐 Круассаны")],
        [KeyboardButton(text="📊 Статистика по продажам"), KeyboardButton(text="🔙 Назад")]
    ], resize_keyboard=True
)

# Кнопки для возврата назад в статистику
back_button = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")]
    ]
)

# Кнопки для статистики по продажам
sales_stats_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="📅 Статистика по датам", callback_data="sales_by_date")],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")]
    ]
)

# Функция для добавления десерта в базу данных
def add_dessert(name):
    cursor.execute("INSERT OR IGNORE INTO desserts (name) VALUES (?)", (name,))
    conn.commit()

# Функция для добавления заказа в базу данных
def add_order(date, dessert, quantity):
    cursor.execute("INSERT INTO orders (date, dessert, quantity) VALUES (?, ?, ?)", (date, dessert, quantity))
    conn.commit()

# Функция для получения всех заказов на определенную дату
def get_orders_for_date(date):
    cursor.execute("SELECT dessert, quantity FROM orders WHERE date=?", (date,))
    return cursor.fetchall()

# Функция для статистики по продажам по датам
def get_sales_by_date():
    cursor.execute("SELECT date, dessert, SUM(quantity) FROM orders GROUP BY date, dessert ORDER BY date DESC")
    return cursor.fetchall()

# Функция для отправки статистики по продажам
async def send_sales_stats(message: types.Message):
    sales = get_sales_by_date()
    if not sales:
        await message.answer("Нет данных о продажах. 📉")
    else:
        response = "📊 Статистика по продажам:\n"
        for sale in sales:
            response += f"{sale[0]} - {sale[1]}: {sale[2]} порций\n"
        await message.answer(response, reply_markup=back_button)

# Функция для прогноза
def predict_order_with_explanation():
    cursor.execute("SELECT date, dessert, quantity FROM orders")
    orders = cursor.fetchall()
    
    cursor.execute("SELECT name FROM desserts")
    desserts = [row[0] for row in cursor.fetchall()]
    
    data = {dessert: [] for dessert in desserts}
    
    for order in orders:
        date_str, dessert, quantity = order
        date = datetime.strptime(date_str, '%Y-%m-%d')
        data[dessert].append((date, quantity))
    
    predictions = {}
    explanations = {}
    for dessert, orders in data.items():
        if len(orders) < 2:
            predictions[dessert] = random.randint(5, 15)
            explanations[dessert] = "Недостаточно данных для анализа, прогноз основан на случайных значениях."
            continue
        
        X = np.array([[(order[0] - min([o[0] for o in orders])).days] for order in orders]).reshape(-1, 1)
        y = np.array([order[1] for order in orders])
        
        model = LinearRegression()
        model.fit(X, y)
        
        next_day = (max([o[0] for o in orders]) + timedelta(days=1))
        X_new = np.array([[next_day.days - min([o[0] for o in orders]).days]])
        predicted_quantity = model.predict(X_new)
        predictions[dessert] = max(0, int(predicted_quantity[0]) + random.randint(0, 2))
        
        explanation = f"Прогноз на основе данных за последние дни. Коэффициент регрессии: {model.coef_[0]:.2f}, " \
                      f"перехват: {model.intercept_:.2f}. Прогнозируемое количество: {predictions[dessert]} порций."
        explanations[dessert] = explanation
    
    return predictions, explanations

# Обработчик команды /start с приветствием
@dp.message(commands=["start"])
async def start(message: types.Message):
    await message.answer(
        "Привет! 👋 Добро пожаловать в наш кондитерский бот! 🍰 "
        "Мы готовы помочь вам заказать вкуснейшие десерты! 😋\n\n"
        "Выберите, что вы хотите заказать сегодня:",
        reply_markup=menu_keyboard
    )

# Обработчик выбора десерта
@dp.message()
async def choose_dessert(message: types.Message):
    dessert = message.text
    if dessert not in ["🧁 Капкейки", "🍰 Торты", "🍪 Печенье", "🥐 Круассаны"]:
        await message.answer("Пожалуйста, выберите один из предложенных десертов:", reply_markup=menu_keyboard)
        return

    quantity_keyboard = InlineKeyboardMarkup(row_width=5)
    for i in range(1, 11):
        quantity_keyboard.add(InlineKeyboardButton(text=str(i), callback_data=f"quantity_{i}_{dessert}"))
    
    await message.answer(f"Сколько порций {dessert} вам нужно? Выберите количество:", reply_markup=quantity_keyboard)

# Обработчик команды статистики
@dp.message(commands=["sales_stats"])
async def sales_stats_command(message: types.Message):
    if str(message.chat.id) in ADMIN_CHAT_IDS:
        await send_sales_stats(message)
    else:
        await message.answer("❌ У вас нет прав доступа для просмотра статистики.")

# Обработчик выбора количества порций
@dp.callback_query()
async def choose_quantity(call: types.CallbackQuery):
    data = call.data.split("_")
    quantity = int(data[1])
    dessert = data[2]
    
    add_dessert(dessert)
    add_order(datetime.now().strftime('%Y-%m-%d'), dessert, quantity)
    
    await call.answer(f"✅ Вы заказали {quantity} порций {dessert}. Спасибо! Мы подготовим ваш заказ!")
    await send_sales_stats(call.message)

# Обработчик кнопки "Назад" (для всех разделов)
@dp.callback_query()
async def go_back(call: types.CallbackQuery):
    if call.data == "back_to_main":
        await call.message.answer(
            "Вы вернулись в главное меню! 🍰 Выберите, что хотите сделать:",
            reply_markup=menu_keyboard
        )

# Запуск бота с использованием asyncio
async def main():
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())