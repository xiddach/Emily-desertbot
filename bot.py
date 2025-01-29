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
    "Еклер фісташковий", "Еклер кокос малина", "Еклер снікерс", "Тарт фісташковий", "Тарт цитрусовий",
    "Подарунок", "Ялинка", "Гречаний медовик", "Купідон", "Три шоколади", "Злиток", "Кавове зерно",
    "Парі брест", "Пʼяна вишня", "Чізкейк", "Зайчик", "Зайчик 18+", "Вупі пай", "Горішок класичний",
    "Горішок кокосовий", "Горішок горгонзола", "Горішок шоколадний", "Печиво Емілі", "Мадлен персиковий",
    "Мадлен ягідний", "Мадлен шоколадний", "Зефір", "Куля какао", "Кіш з куркою і грибами", "Кіш 5 сирів",
    "Кіш з лососем"
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
        # Якщо файл не існує, створюємо новий DataFrame
        data = pd.DataFrame(columns=['date'] + DESSERTS)
    return data

# Збереження даних у файл
def save_data(data):
    data.to_csv(DATA_FILE, index=False)

# Прогнозування замовлень за допомогою Random Forest
def predict_orders(dessert_name, data):
    """
    Прогнозування замовлень десертів за допомогою Random Forest.
    :param dessert_name: назва десерту
    :param data: DataFrame з історичними даними
    :return: прогнозоване замовлення на наступний день
    """
    # Вибираємо дані лише для конкретного десерту
    dessert_data = data[['date', dessert_name]].dropna()
    if len(dessert_data) < 5:
        raise ValueError(f"Недостатньо даних для прогнозування {dessert_name}.")
    # Беремо останні 5 днів
    last_5_days = dessert_data.tail(5)
    X = np.array(range(len(last_5_days))).reshape(-1, 1)
    y = np.array(last_5_days[dessert_name])
    # Навчаємо модель Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Прогнозуємо замовлення на наступний день
    next_day = X[-1][0] + 1
    predicted_order = model.predict([[next_day]])[0]
    return round(predicted_order)

# Обробник команди /start
def start(update: Update, context: CallbackContext) -> int:
    keyboard = [
        ["Ввести дані минулих днів"],
        ["Переглянути дані минулих днів"],
        ["Редагувати список десертів"],
        ["Почати прогнозування"],
        ["Назад до головного меню"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text(
        "Привіт! Що ви хочете зробити?",
        reply_markup=reply_markup
    )
    return VIEW_DATA

# Обробник вибору дії
def handle_action(update: Update, context: CallbackContext) -> int:
    action = update.message.text
    if action == "Ввести дані минулих днів":
        update.message.reply_text('Будь ласка, введіть дату дня, для якого ви хочете ввести залишки (формат: ДД ММ РРРР):')
        return DATE
    elif action == "Переглянути дані минулих днів":
        return view_data(update, context)
    elif action == "Редагувати список десертів":
        return edit_desserts_menu(update, context)
    elif action == "Почати прогнозування":
        return start_prediction(update, context)
    elif action == "Назад до головного меню":
        return start(update, context)
    else:
        update.message.reply_text("Невідома команда. Будь ласка, виберіть дію з меню.")
        return VIEW_DATA

# Перегляд даних минулих днів
def view_data(update: Update, context: CallbackContext) -> int:
    data = load_data()
    if data.empty:
        update.message.reply_text("Історичні дані відсутні.")
        return VIEW_DATA
    
    dates = sorted(data['date'].unique(), key=lambda x: datetime.strptime(x, '%d %B %Y'))
    keyboard = [[date] for date in dates]
    keyboard.append(["Назад до головного меню"])
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text("Оберіть дату для перегляду:", reply_markup=reply_markup)
    return VIEW_DATE_SELECTION

# Дата вибору для перегляду даних
def view_date_selection(update: Update, context: CallbackContext) -> int:
    selected_date = update.message.text
    data = load_data()
    row = data[data['date'] == selected_date]
    if not row.empty:
        response = f"Дані за {selected_date}:\n"
        for dessert in DESSERTS:
            if dessert in row and not pd.isna(row[dessert].values[0]):
                response += f"  {dessert}: {row[dessert].values[0]}\n"
        update.message.reply_text(response)
    else:
        update.message.reply_text(f"Дані за {selected_date} не знайдені.")
    return view_data(update, context)

# Меню редагування списку десертів
def edit_desserts_menu(update: Update, context: CallbackContext) -> int:
    keyboard = [
        ["Додати десерт"],
        ["Видалити десерт"],
        ["Назад до головного меню"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text(
        "Оберіть дію для редагування списку десертів:",
        reply_markup=reply_markup
    )
    return EDIT_DATA

# Додавання нового десерту
def add_dessert(update: Update, context: CallbackContext) -> int:
    update.message.reply_text(
        "Введіть назву нового десерту:",
        reply_markup=ReplyKeyboardRemove()
    )
    return ADD_DESSERT

# Обробник додавання нового десерту
def handle_add_dessert(update: Update, context: CallbackContext) -> int:
    new_dessert = update.message.text.strip()
    if new_dessert in DESSERTS:
        update.message.reply_text(f"Десерт '{new_dessert}' вже існує.")
    else:
        DESSERTS.append(new_dessert)
        update.message.reply_text(f"Десерт '{new_dessert}' успішно додано.")
    return edit_desserts_menu(update, context)

# Видалення десерту
def remove_dessert(update: Update, context: CallbackContext) -> int:
    keyboard = [[dessert] for dessert in DESSERTS]
    keyboard.append(["Назад до головного меню"])
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text(
        "Оберіть десерт для видалення:",
        reply_markup=reply_markup
    )
    return REMOVE_DESSERT

# Обробник видалення десерту
def handle_remove_dessert(update: Update, context: CallbackContext) -> int:
    dessert_to_remove = update.message.text.strip()
    if dessert_to_remove in DESSERTS:
        DESSERTS.remove(dessert_to_remove)
        update.message.reply_text(f"Десерт '{dessert_to_remove}' успішно видалено.")
    else:
        update.message.reply_text(f"Десерт '{dessert_to_remove}' не знайдено.")
    return edit_desserts_menu(update, context)

# Початок прогнозування
def start_prediction(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Починаємо прогнозування...")
    # Тут можна додати логіку прогнозування
    return VIEW_DATA

# Обробник дати
def get_date(update: Update, context: CallbackContext) -> int:
    try:
        input_date = update.message.text
        date = datetime.strptime(input_date, '%d %B %Y').date()
        context.user_data['date'] = date.strftime('%d %B %Y')  # Save in the required format
        context.user_data['desserts'] = {}
        update.message.reply_text(f'Дякую! Тепер давайте введемо залишки для кожного десерту.')
        return ask_next_dessert(update, context)
    except ValueError:
        update.message.reply_text('Неправильний формат дати. Будь ласка, введіть дату у форматі ДД ММ РРРР:')
        return DATE

# Запит залишків для кожного десерту
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

# Обробник введення залишків
def handle_dessert_input(update: Update, context: CallbackContext) -> int:
    try:
        amount = int(update.message.text)
        current_dessert = context.user_data['current_dessert']
        context.user_data['desserts'][current_dessert] = amount
        return ask_next_dessert(update, context)
    except ValueError:
        update.message.reply_text('Будь ласка, введіть число:')
        return DESSERTS_INPUT

# Збереження даних і прогнозування
def finalize_data(update: Update, context: CallbackContext) -> int:
    date = context.user_data['date']
    desserts = context.user_data['desserts']
    # Завантажуємо існуючі дані
    data = load_data()
    # Додаємо нові дані
    new_row = {'date': date}
    new_row.update(desserts)
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    save_data(data)
    # Прогнозуємо замовлення
    predictions = {}
    for dessert in DESSERTS:
        try:
            prediction = predict_orders(dessert, data)
            predictions[dessert] = prediction
        except ValueError:
            predictions[dessert] = 'Недостатньо даних'
    # Розраховуємо дату прогнозу
    next_date = datetime.strptime(date, '%d %B %Y').date() + timedelta(days=1)
    # Формуємо повідомлення з прогнозами
    response = f'Прогнозовані замовлення на {next_date.strftime("%d %B %Y")}:\n'
    for dessert, prediction in predictions.items():
        response += f'{dessert}: {prediction}\n'
    update.message.reply_text(response)
    return ConversationHandler.END

# Обробник команди /cancel
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Операція скасована.')
    return ConversationHandler.END

# Handle actions in the edit desserts menu
def handle_edit_data(update: Update, context: CallbackContext) -> int:
    action = update.message.text.strip()
    if action == "Додати десерт":
        return add_dessert(update, context)
    elif action == "Видалити десерт":
        return remove_dessert(update, context)
    elif action == "Назад до головного меню":
        return start(update, context)
    else:
        update.message.reply_text("Невідома команда. Будь ласка, виберіть дію з меню.")
        return EDIT_DATA

# Головна функція
def main():
    TOKEN = os.environ.get('TOKEN')
    if not TOKEN:
        print("ERROR: Environment variable 'TOKEN' is missing or empty!")
        exit(1)
    print("TOKEN loaded successfully.")
    
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            VIEW_DATA: [MessageHandler(Filters.text & ~Filters.command, handle_action)],
            DATE: [MessageHandler(Filters.text & ~Filters.command, get_date)],
            DESSERTS_INPUT: [MessageHandler(Filters.text & ~Filters.command, handle_dessert_input)],
            PREDICT: [MessageHandler(Filters.text & ~Filters.command, finalize_data)],
            EDIT_DATA: [MessageHandler(Filters.text & ~Filters.command, handle_edit_data)],
            ADD_DESSERT: [MessageHandler(Filters.text & ~Filters.command, handle_add_dessert)],
            REMOVE_DESSERT: [MessageHandler(Filters.text & ~Filters.command, handle_remove_dessert)],
            VIEW_DATE_SELECTION: [MessageHandler(Filters.text & ~Filters.command, view_date_selection)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    dispatcher.add_handler(conv_handler)
    updater.start_polling()
    print("Bot started polling.")
    updater.idle()

if __name__ == '__main__':
    main()
