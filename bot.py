import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
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
    dessert_data = data[['date', dessert_name]].dropna()
    if len(dessert_data) < 2:
        raise ValueError(f"Недостатньо даних для прогнозування {dessert_name}.")
    last_days = min(14, len(dessert_data))
    last_n_days = dessert_data.tail(last_days)
    X = np.array(range(len(last_n_days))).reshape(-1, 1)
    y = np.array(last_n_days[dessert_name])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    next_day = X[-1][0] + 1
    predicted_order = model.predict([[next_day]])[0]
    return round(predicted_order)

# Генерація inline-клавіатури
def create_inline_keyboard(options):
    keyboard = [[InlineKeyboardButton(option, callback_data=option)] for option in options]
    return InlineKeyboardMarkup(keyboard)

# Обробник команди /start
def start(update: Update, context: CallbackContext) -> int:
    keyboard = [
        ["Ввести дані минулих днів", "Переглянути дані минулих днів"],
        ["Редагувати список десертів", "Почати прогнозування"],
        ["Назад до головного меню"]
    ]
    reply_markup = create_inline_keyboard([item for sublist in keyboard for item in sublist])
    update.message.reply_text(
        "*Привіт!* Що ви хочете зробити?\n\n"
        "- Ввести дані минулих днів\n"
        "- Переглянути дані минулих днів\n"
        "- Редагувати список десертів\n"
        "- Почати прогнозування",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )
    return VIEW_DATA

# Обробник callback-запитів
def handle_callback_query(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    action = query.data
    if action == "Ввести дані минулих днів":
        query.edit_message_text('Будь ласка, введіть дату дня, для якого ви хочете ввести залишки (формат: ДД.ММ.РРРР):')
        return DATE
    elif action == "Переглянути дані минулих днів":
        view_data(update, context, query=query)
    elif action == "Редагувати список десертів":
        edit_desserts_menu(update, context, query=query)
    elif action == "Почати прогнозування":
        start_prediction(update, context, query=query)
    elif action == "Назад до головного меню":
        start(update, context)

# Перегляд даних минулих днів
def view_data(update: Update, context: CallbackContext, query=None) -> int:
    data = load_data()
    if data.empty:
        if query:
            query.edit_message_text("Історичні дані відсутні.")
        else:
            update.message.reply_text("Історичні дані відсутні.")
        return VIEW_DATA
    dates = sorted(data['date'].unique(), key=lambda x: datetime.strptime(x, '%d.%m.%Y'))
    keyboard = [[date] for date in dates]
    keyboard.append(["Назад до головного меню"])
    reply_markup = create_inline_keyboard([item for sublist in keyboard for item in sublist])
    if query:
        query.edit_message_text("Оберіть дату для перегляду:", reply_markup=reply_markup)
    else:
        update.message.reply_text("Оберіть дату для перегляду:", reply_markup=reply_markup)
    return VIEW_DATE_SELECTION

# Дата вибору для перегляду даних
def view_date_selection(update: Update, context: CallbackContext) -> int:
    selected_date = update.message.text
    data = load_data()
    row = data[data['date'] == selected_date]
    if not row.empty:
        response = f"Дані за {selected_date} ({datetime.strptime(selected_date, '%d.%m.%Y').strftime('%A')}):\n"
        for dessert in DESSERTS:
            if dessert in row and not pd.isna(row[dessert].values[0]):
                response += f"  {dessert}: {row[dessert].values[0]}\n"
        update.message.reply_text(response)
    else:
        update.message.reply_text(f"Дані за {selected_date} не знайдені.")
    return view_data(update, context)

# Меню редагування списку десертів
def edit_desserts_menu(update: Update, context: CallbackContext, query=None) -> int:
    current_desserts = "\n".join(DESSERTS)
    keyboard = [
        ["Додати десерт"],
        ["Видалити десерт"],
        ["Назад до головного меню"]
    ]
    reply_markup = create_inline_keyboard([item for sublist in keyboard for item in sublist])
    if query:
        query.edit_message_text(
            f"Поточний список десертів:\n{current_desserts}\n\nОберіть дію для редагування списку десертів:",
            reply_markup=reply_markup
        )
    else:
        update.message.reply_text(
            f"Поточний список десертів:\n{current_desserts}\n\nОберіть дію для редагування списку десертів:",
            reply_markup=reply_markup
        )
    return EDIT_DATA

# Обробник додавання нового десерту
def handle_add_dessert(update: Update, context: CallbackContext) -> int:
    new_dessert = update.message.text.strip()
    if new_dessert in DESSERTS:
        update.message.reply_text(f"Десерт '{new_dessert}' вже існує.")
    else:
        DESSERTS.append(new_dessert)
        update.message.reply_text(f"Десерт '{new_dessert}' успішно додано. Підтверджуйте додавання?")
        keyboard = [
            [InlineKeyboardButton("Так", callback_data=f"confirm_add_{new_dessert}")],
            [InlineKeyboardButton("Ні", callback_data="cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("Підтвердіть додавання:", reply_markup=reply_markup)
    return EDIT_DATA

# Обробник підтвердження додавання/видалення десерту
def handle_confirmation(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()
    action = query.data
    if action.startswith("confirm_add_"):
        new_dessert = action.replace("confirm_add_", "")
        DESSERTS.append(new_dessert)
        query.edit_message_text(f"Десерт '{new_dessert}' успішно додано.")
    elif action.startswith("confirm_remove_"):
        dessert_to_remove = action.replace("confirm_remove_", "")
        DESSERTS.remove(dessert_to_remove)
        query.edit_message_text(f"Десерт '{dessert_to_remove}' успішно видалено.")
    elif action == "cancel":
        query.edit_message_text("Операція скасована.")
    return edit_desserts_menu(update, context)

# Обробник видалення десерту
def handle_remove_dessert(update: Update, context: CallbackContext) -> int:
    keyboard = [[InlineKeyboardButton(dessert, callback_data=f"confirm_remove_{dessert}")] for dessert in DESSERTS]
    keyboard.append([InlineKeyboardButton("Назад до головного меню", callback_data="cancel")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text(
        "Оберіть десерт для видалення:",
        reply_markup=reply_markup
    )
    return REMOVE_DESSERT

# Обробник додавання нового десерту
def add_dessert(update: Update, context: CallbackContext) -> int:
    update.message.reply_text(
        "Введіть назву нового десерту:"
    )
    return ADD_DESSERT

# Обробник видалення десерту
def remove_dessert(update: Update, context: CallbackContext) -> int:
    handle_remove_dessert(update, context)
    return REMOVE_DESSERT

# Початок прогнозування
def start_prediction(update: Update, context: CallbackContext, query=None) -> int:
    if query:
        query.edit_message_text("Починаємо прогнозування...")
    else:
        update.message.reply_text("Починаємо прогнозування...")
    return VIEW_DATA

# Обробник дати
def get_date(update: Update, context: CallbackContext) -> int:
    try:
        input_date = update.message.text
        date = datetime.strptime(input_date, '%d.%m.%Y').date()
        context.user_data['date'] = date.strftime('%d.%m.%Y')  # Save in the required format
        context.user_data['desserts'] = {}
        update.message.reply_text(f'Дякую! Тепер давайте введемо залишки для кожного десерту.')
        return ask_next_dessert(update, context)
    except ValueError:
        update.message.reply_text('Неправильний формат дати. Будь ласка, введіть дату у форматі ДД.ММ.РРРР:')
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
    date = datetime.strptime(context.user_data['date'], '%d.%m.%Y').date()
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
    next_date = date + timedelta(days=1)
    day_of_week = next_date.strftime('%A')
    response = f'Прогнозовані замовлення на {day_of_week}, {next_date.strftime("%d.%m.%Y")}:\n'
    for dessert, prediction in predictions.items():
        response += f'{dessert}: {prediction}\n'
    update.message.reply_text(response)
    return ConversationHandler.END

# Обробник команди /cancel
def cancel(update: Update, context: CallbackContext) -> int:
    if update.message:
        update.message.reply_text('Операція скасована.')
    elif update.callback_query:
        update.callback_query.edit_message_text('Операція скасована.')
    return ConversationHandler.END

# Handle actions in the edit desserts menu
def handle_edit_data(update: Update, context: CallbackContext) -> int:
    action = update.message.text.strip() if update.message else update.callback_query.data
    if action == "Додати десерт":
        return add_dessert(update, context)
    elif action == "Видалити десерт":
        return remove_dessert(update, context)
    elif action == "Назад до головного меню":
        return start(update, context)
    else:
        if update.message:
            update.message.reply_text("Невідома команда. Будь ласка, виберіть дію з меню.")
        elif update.callback_query:
            update.callback_query.edit_message_text("Невідома команда. Будь ласка, виберіть дію з меню.")
        return EDIT_DATA

# Обробник вибору дії
def handle_action(update: Update, context: CallbackContext) -> int:
    action = update.message.text
    if action == "Ввести дані минулих днів":
        update.message.reply_text('Будь ласка, введіть дату дня, для якого ви хочете ввести залишки (формат: ДД.ММ.РРРР):')
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
            VIEW_DATA: [CallbackQueryHandler(handle_callback_query), MessageHandler(Filters.text & ~Filters.command, handle_action)],
            DATE: [MessageHandler(Filters.text & ~Filters.command, get_date)],
            DESSERTS_INPUT: [MessageHandler(Filters.text & ~Filters.command, handle_dessert_input)],
            PREDICT: [MessageHandler(Filters.text & ~Filters.command, finalize_data)],
            EDIT_DATA: [MessageHandler(Filters.text & ~Filters.command, handle_edit_data), CallbackQueryHandler(handle_confirmation)],
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
