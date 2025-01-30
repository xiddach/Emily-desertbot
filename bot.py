import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler, CallbackQueryHandler

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
    if len(dessert_data) < 14:
        available_days = len(dessert_data)
        if available_days < 5:
            raise ValueError(f"Недостатньо даних для прогнозування {dessert_name}.")
        else:
            last_days = dessert_data.tail(available_days)
    else:
        last_days = dessert_data.tail(14)
    last_days['date'] = pd.to_datetime(last_days['date'], format='%d.%m.%Y')
    last_days = last_days.sort_values(by='date')
    X = np.array(range(len(last_days))).reshape(-1, 1)
    y = np.array(last_days[dessert_name])
    # Навчаємо модель Random Forest з меншою кількістю дерев
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Паралельне обчислення
    model.fit(X, y)
    # Прогнозуємо замовлення на наступний день
    next_day = X[-1][0] + 1
    predicted_order = model.predict([[next_day]])[0]
    return round(predicted_order)
# Обробник команди /start
def start(update: Update, context: CallbackContext) -> int:
    keyboard = [
        [InlineKeyboardButton("Ввести дані минулих днів", callback_data="enter_data")],
        [InlineKeyboardButton("Переглянути дані минулих днів", callback_data="view_data")],
        [InlineKeyboardButton("Редагувати список десертів", callback_data="edit_desserts")],
        [InlineKeyboardButton("Почати прогнозування", callback_data="start_prediction")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text(
        "Привіт! Що ви хочете зробити?",
        reply_markup=reply_markup
    )
    return VIEW_DATA
# Обробник вибору дії
def handle_action(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()
    action = query.data
    if action == "enter_data":
        query.edit_message_text(text='Будь ласка, введіть дату дня, для якого ви хочете ввести залишки (формат: ДД.ММ.РРРР):')
        return DATE
    elif action == "view_data":
        return view_data(update, context)
    elif action == "edit_desserts":
        return edit_desserts_menu(update, context)
    elif action == "start_prediction":
        return start_prediction(update, context)
    else:
        query.edit_message_text(text="Невідома команда. Будь ласка, виберіть дію з меню.")
        return VIEW_DATA
# Перегляд даних минулих днів
def view_data(update: Update, context: CallbackContext) -> int:
    data = load_data()
    if data.empty:
        update.callback_query.edit_message_text(text="Історичні дані відсутні.")
        return VIEW_DATA
    dates = sorted(data['date'].unique(), key=lambda x: datetime.strptime(x, '%d.%m.%Y'))
    keyboard = []
    for i in range(0, len(dates), 2):
        if i + 1 < len(dates):
            keyboard.append([
                InlineKeyboardButton(dates[i], callback_data=f"view_{dates[i]}"),
                InlineKeyboardButton(dates[i + 1], callback_data=f"view_{dates[i + 1]}")
            ])
        else:
            keyboard.append([
                InlineKeyboardButton(dates[i], callback_data=f"view_{dates[i]}")
            ])
    keyboard.append([InlineKeyboardButton("Назад до головного меню", callback_data="back_to_main")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(text="Оберіть дату для перегляду:", reply_markup=reply_markup)
    return VIEW_DATE_SELECTION
# Дата вибору для перегляду даних
def view_date_selection(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()
    selected_date = query.data.split('_')[1]
    data = load_data()
    row = data[data['date'] == selected_date]
    if not row.empty:
        response = f"Дані за {selected_date}:\n"
        for dessert in DESSERTS:
            if dessert in row and not pd.isna(row[dessert].values[0]):
                response += f"  {dessert}: {row[dessert].values[0]}\n"
        query.edit_message_text(text=response)
    else:
        query.edit_message_text(text=f"Дані за {selected_date} не знайдені.")
    return VIEW_DATE_SELECTION  # Залишаємося у тому ж стані
# Меню редагування списку десертів
def edit_desserts_menu(update: Update, context: CallbackContext) -> int:
    keyboard = [
        [InlineKeyboardButton("Додати десерт", callback_data="add_dessert")],
        [InlineKeyboardButton("Видалити десерт", callback_data="remove_dessert")],
        [InlineKeyboardButton("Назад до головного меню", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(text="Оберіть дію для редагування списку десертів:", reply_markup=reply_markup)
    return EDIT_DATA
# Додавання нового десерту
def add_dessert(update: Update, context: CallbackContext) -> int:
    update.callback_query.edit_message_text(text="Введіть назву нового десерту:")
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
    keyboard = []
    for i in range(0, len(DESSERTS), 2):
        if i + 1 < len(DESSERTS):
            keyboard.append([
                InlineKeyboardButton(DESSERTS[i], callback_data=f"remove_{DESSERTS[i]}"),
                InlineKeyboardButton(DESSERTS[i + 1], callback_data=f"remove_{DESSERTS[i + 1]}")
            ])
        else:
            keyboard.append([
                InlineKeyboardButton(DESSERTS[i], callback_data=f"remove_{DESSERTS[i]}")
            ])
    keyboard.append([InlineKeyboardButton("Назад до головного меню", callback_data="back_to_main")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(text="Оберіть десерт для видалення:", reply_markup=reply_markup)
    return REMOVE_DESSERT
# Обробник видалення десерту
def handle_remove_dessert(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()
    dessert_to_remove = query.data.split('_')[1]
    if dessert_to_remove in DESSERTS:
        DESSERTS.remove(dessert_to_remove)
        query.edit_message_text(text=f"Десерт '{dessert_to_remove}' успішно видалено.")
    else:
        query.edit_message_text(text=f"Десерт '{dessert_to_remove}' не знайдено.")
    return edit_desserts_menu(update, context)
# Початок прогнозування
def start_prediction(update: Update, context: CallbackContext) -> int:
    update.callback_query.edit_message_text(text="Починаємо прогнозування...")
    # Тут можна додати логіку прогнозування
    return VIEW_DATA
# Обробник дати
def get_date(update: Update, context: CallbackContext) -> int:
    try:
        input_date = update.message.text
        date = datetime.strptime(input_date, '%d.%m.%Y').date()
        context.user_data['date'] = date.strftime('%d.%m.%Y')  # Save in the required format
        update.message.reply_text(f'Дякую! Тепер введіть залишки для всіх десертів у форматі:\nДесерт 1 Кількість\nДесерт 2 Кількість\n...')
        return DESSERTS_INPUT
    except ValueError:
        update.message.reply_text('Неправильний формат дати. Будь ласка, введіть дату у форматі ДД.ММ.РРРР:')
        return DATE
# Нова функція для парсингу інформації про залишки з одного повідомлення
def parse_stock_info(message: str) -> dict:
    """
    Парсинг інформації про залишки з одного повідомлення.
    :param message: Повідомлення з інформацією про залишки
    :return: Словник з назвами десертів та їх кількостями
    """
    lines = message.strip().split('\n')
    stock_info = {}
    for line in lines:
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            dessert_name = ' '.join(parts[:-1]).strip()
            try:
                quantity = int(parts[-1].strip())
                if dessert_name not in DESSERTS:
                    DESSERTS.append(dessert_name)
                    print(f"Десерт '{dessert_name}' додано до списку.")
                stock_info[dessert_name] = quantity
            except ValueError:
                print(f"Попередження: Некоректна кількість для десерту '{dessert_name}'.")
        else:
            print(f"Попередження: Некоректний формат рядка: '{line}'")
    return stock_info
# Обробник введення залишків
def handle_dessert_input(update: Update, context: CallbackContext) -> int:
    stock_info = parse_stock_info(update.message.text)
    context.user_data['desserts'] = stock_info
    return finalize_data(update, context)
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
        except ValueError as e:
            predictions[dessert] = str(e)
    # Розраховуємо дату прогнозу
    next_date = datetime.strptime(date, '%d.%m.%Y').date() + timedelta(days=1)
    # Формуємо повідомлення з прогнозами
    response = f'Прогнозовані замовлення на {next_date.strftime("%d.%m.%Y")}:\n'
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
    query = update.callback_query
    query.answer()
    action = query.data
    if action == "add_dessert":
        return add_dessert(update, context)
    elif action == "remove_dessert":
        return remove_dessert(update, context)
    elif action == "back_to_main":
        return start(update, context)
    else:
        query.edit_message_text(text="Невідома команда. Будь ласка, виберіть дію з меню.")
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
            VIEW_DATA: [CallbackQueryHandler(handle_action)],
            DATE: [MessageHandler(Filters.text & ~Filters.command, get_date)],
            DESSERTS_INPUT: [MessageHandler(Filters.text & ~Filters.command, handle_dessert_input)],
            PREDICT: [MessageHandler(Filters.text & ~Filters.command, finalize_data)],
            EDIT_DATA: [CallbackQueryHandler(handle_edit_data)],
            ADD_DESSERT: [MessageHandler(Filters.text & ~Filters.command, handle_add_dessert)],
            REMOVE_DESSERT: [CallbackQueryHandler(handle_remove_dessert)],
            VIEW_DATE_SELECTION: [CallbackQueryHandler(view_date_selection)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    dispatcher.add_handler(conv_handler)
    updater.start_polling()
    print("Bot started polling.")
    updater.idle()

if __name__ == '__main__':
    main()
