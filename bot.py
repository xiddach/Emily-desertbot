import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler

# Список назв десертів
DESSERTS = [
    "Макарон фісташка малина", "Макарон снікерс", "Макарон чізкейк нутела", "Макарон ягідний чізкейк",
    "Макарон груша горгонзола", "Макарон кокос мигдаль", "Макарон лотус", "Макарон гарбузове лате",
    "Еклер фісташковий", "Еклер кокос малина", "Еклер снікерс", "Тарт фісташковий", "Тарт цитрусовий"
]

# Файл для зберігання історичних даних
DATA_FILE = 'dessert_data.csv'

# Стани для ConversationHandler
DATE, DESSERTS_INPUT, PREDICT, VIEW_DATA, EDIT_DATA, ADD_DESSERT, REMOVE_DESSERT, VIEW_DATE_SELECTION = range(8)

# Завантаження історичних даних
def load_data():
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        data = pd.DataFrame(columns=['date'] + DESSERTS)
    return data

# Збереження даних у файл
def save_data(data):
    data.to_csv(DATA_FILE, index=False)

# Прогнозування замовлень
def predict_orders(dessert_name, data):
    dessert_data = data[['date', dessert_name]].dropna()
    if len(dessert_data) < 2:
        raise ValueError(f"Недостатньо даних для прогнозування {dessert_name}.")
    
    last_days = min(14, len(dessert_data))  # Беремо до 14 днів, якщо є менше – беремо всі наявні
    last_n_days = dessert_data.tail(last_days)

    X = np.array(range(len(last_n_days))).reshape(-1, 1)
    y = np.array(last_n_days[dessert_name])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    next_day = X[-1][0] + 1
    predicted_order = model.predict([[next_day]])[0]

    return round(predicted_order)

# Команда /start
def start(update: Update, context: CallbackContext) -> int:
    keyboard = [
        ["Ввести дані минулих днів"],
        ["Переглянути дані минулих днів"],
        ["Редагувати список десертів"],
        ["Почати прогнозування"],
        ["Назад до головного меню"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text("Привіт! Що ви хочете зробити?", reply_markup=reply_markup)
    return VIEW_DATA

# Обробка введення дати
def get_date(update: Update, context: CallbackContext) -> int:
    try:
        input_date = update.message.text.strip()
        date = datetime.strptime(input_date, '%d.%m.%Y').date()
        context.user_data['date'] = date.strftime('%d.%m.%Y')
        context.user_data['desserts'] = {}
        update.message.reply_text(f'Дякую! Тепер введіть залишки для кожного десерту.')
        return ask_next_dessert(update, context)
    except ValueError:
        update.message.reply_text('Неправильний формат дати. Введіть у форматі: дд.мм.рррр')
        return DATE

# Введення залишків для кожного десерту
def ask_next_dessert(update: Update, context: CallbackContext) -> int:
    desserts = context.user_data.get('desserts', {})
    remaining_desserts = [dessert for dessert in DESSERTS if dessert not in desserts]

    if remaining_desserts:
        dessert = remaining_desserts[0]
        context.user_data['current_dessert'] = dessert
        update.message.reply_text(f'Скільки залишилось {dessert}? (введіть число):')
        return DESSERTS_INPUT
    else:
        return finalize_data(update, context)

# Обробка введення залишків
def handle_dessert_input(update: Update, context: CallbackContext) -> int:
    try:
        amount = int(update.message.text)
        current_dessert = context.user_data['current_dessert']
        context.user_data['desserts'][current_dessert] = amount
        return ask_next_dessert(update, context)
    except ValueError:
        update.message.reply_text('Будь ласка, введіть число:')
        return DESSERTS_INPUT

# Збереження даних і прогноз
def finalize_data(update: Update, context: CallbackContext) -> int:
    date = context.user_data['date']
    desserts = context.user_data['desserts']

    data = load_data()
    new_row = {'date': date}
    new_row.update(desserts)
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    save_data(data)

    predictions = {}
    for dessert in DESSERTS:
        try:
            prediction = predict_orders(dessert, data)
            predictions[dessert] = prediction
        except ValueError:
            predictions[dessert] = 'Недостатньо даних'

    next_date = datetime.strptime(date, '%d.%m.%Y').date() + timedelta(days=1)
    response = f'Прогнозовані замовлення на {next_date.strftime("%d.%m.%Y")}:\n'
    for dessert, prediction in predictions.items():
        response += f'{dessert}: {prediction}\n'

    update.message.reply_text(response)
    return ConversationHandler.END

# Обробник команди /cancel
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Операція скасована.')
    return ConversationHandler.END

# Головна функція
def main():
    TOKEN = os.environ.get('TOKEN')
    if not TOKEN:
        print("ERROR: Environment variable 'TOKEN' is missing or empty!")
        exit(1)

    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            VIEW_DATA: [MessageHandler(Filters.text & ~Filters.command, get_date)],
            DATE: [MessageHandler(Filters.text & ~Filters.command, get_date)],
            DESSERTS_INPUT: [MessageHandler(Filters.text & ~Filters.command, handle_dessert_input)],
            PREDICT: [MessageHandler(Filters.text & ~Filters.command, finalize_data)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
