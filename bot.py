import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler, CallbackQueryHandler

# List of dessert names
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

# File for storing historical data
DATA_FILE = 'dessert_data.csv'

# States for ConversationHandler
DATE, DESSERTS_INPUT, PREDICT, VIEW_DATA, EDIT_DATA, ADD_DESSERT, REMOVE_DESSERT, VIEW_DATE_SELECTION = range(8)

# Load historical data
def load_data():
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        data = pd.DataFrame(columns=['date'] + DESSERTS)
    return data

# Save data to file
def save_data(data):
    data.to_csv(DATA_FILE, index=False)

# Predict orders using Random Forest
def predict_orders(dessert_name, data):
    """
    Predict dessert orders using Random Forest.
    :param dessert_name: name of the dessert
    :param data: DataFrame with historical data
    :return: predicted order for the next day or None if not enough data
    """
    # Select data only for the specific dessert
    dessert_data = data[['date', dessert_name]].dropna()
    if len(dessert_data) < 21:
        available_days = len(dessert_data)
        if available_days < 5:
            return None  # Not enough data for prediction
        else:
            last_days = dessert_data.tail(available_days)
    else:
        last_days = dessert_data.tail(21)
    
    last_days['date'] = pd.to_datetime(last_days['date'], format='%d.%m.%Y')
    last_days = last_days.sort_values(by='date')
    X = np.array(range(len(last_days))).reshape(-1, 1)
    y = np.array(last_days[dessert_name])
    # Train Random Forest model with fewer trees
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Parallel computation
    model.fit(X, y)
    # Predict order for the next day
    next_day = X[-1][0] + 1
    predicted_order = model.predict([[next_day]])[0]
    return round(predicted_order)

# Handler for the /start command
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

# Handler for action selection
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

# View past data
def view_data(update: Update, context: CallbackContext) -> int:
    data = load_data()
    if data.empty:
        update.callback_query.edit_message_text(text="Історичні дані відсутні.")
        return VIEW_DATA
    
    dates = sorted(data['date'].unique(), key=lambda x: datetime.strptime(x, '%d.%m.%Y'))
    keyboard = []
    for i in range(0, len(dates), 2):
        if i + 1 < len(dates):
            day_1 = datetime.strptime(dates[i], '%d.%m.%Y').strftime('%A')
            day_2 = datetime.strptime(dates[i+1], '%d.%m.%Y').strftime('%A')
            keyboard.append([
                InlineKeyboardButton(f"{dates[i]} ({day_1})", callback_data=f"view_{dates[i]}"),
                InlineKeyboardButton(f"{dates[i + 1]} ({day_2})", callback_data=f"view_{dates[i + 1]}")
            ])
        else:
            day = datetime.strptime(dates[i], '%d.%m.%Y').strftime('%A')
            keyboard.append([
                InlineKeyboardButton(f"{dates[i]} ({day})", callback_data=f"view_{dates[i]}")
            ])
    keyboard.append([InlineKeyboardButton("Назад до головного меню", callback_data="back_to_main")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(text="Оберіть дату для перегляду:", reply_markup=reply_markup)
    return VIEW_DATE_SELECTION

# Date selection for viewing data
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
    return VIEW_DATE_SELECTION  # Stay in the same state

# Desserts editing menu
def edit_desserts_menu(update: Update, context: CallbackContext) -> int:
    keyboard = [
        [InlineKeyboardButton("Додати десерт", callback_data="add_dessert")],
        [InlineKeyboardButton("Видалити десерт", callback_data="remove_dessert")],
        [InlineKeyboardButton("Назад до головного меню", callback_data="back_to_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(text="Оберіть дію для редагування списку десертів:", reply_markup=reply_markup)
    return EDIT_DATA

# Adding a new dessert
def add_dessert(update: Update, context: CallbackContext) -> int:
    update.callback_query.edit_message_text(text="Введіть назву нового десерту:")
    return ADD_DESSERT

# Handler for adding a new dessert
def handle_add_dessert(update: Update, context: CallbackContext) -> int:
    new_dessert = update.message.text.strip()
    if new_dessert in DESSERTS:
        update.message.reply_text(f"Десерт '{new_dessert}' вже існує.")
    else:
        DESSERTS.append(new_dessert)
        update.message.reply_text(f"Десерт '{new_dessert}' успішно додано.")
    return edit_desserts_menu(update, context)

# Removing a dessert
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

# Handler for removing a dessert
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

# Start prediction
def start_prediction(update: Update, context: CallbackContext) -> int:
    update.callback_query.edit_message_text(text="Починаємо прогнозування...")
    # Here you can add prediction logic
    return VIEW_DATA

# Date handler
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

# New function for parsing stock information from one message
def parse_stock_info(message: str) -> dict:
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

# Handler for entering leftovers
def handle_dessert_input(update: Update, context: CallbackContext) -> int:
    stock_info = parse_stock_info(update.message.text)
    context.user_data['desserts'] = stock_info
    return finalize_data(update, context)

# Saving data and predicting
def finalize_data(update: Update, context: CallbackContext) -> int:
    date = context.user_data['date']
    desserts = context.user_data['desserts']
    # Load existing data
    data = load_data()
    # Add new data
    new_row = {'date': date}
    new_row.update(desserts)
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    save_data(data)
    # Predict orders
    predictions = {}
    for dessert in DESSERTS:
        try:
            prediction = predict_orders(dessert, data)
            if prediction is not None:
                predictions[dessert] = prediction
            else:
                predictions[dessert] = "Not enough data for prediction"
        except ValueError as e:
            predictions[dessert] = str(e)
    # Calculate the date of the prediction
    next_date = datetime.strptime(date, '%d.%m.%Y').date() + timedelta(days=1)
    # Formulate a message with predictions
    response = f'Прогнозовані замовлення на {next_date.strftime("%d.%m.%Y")}:\n'
    for dessert, prediction in predictions.items():
        if isinstance(prediction, (int, float)):
            if dessert in desserts:
                order_quantity = max(0, prediction - desserts.get(dessert, 0))
            else:
                order_quantity = prediction
            response += f'{dessert}: Прогноз {prediction}, Поточний залишок {desserts.get(dessert, 0)}, Потрібно замовити {order_quantity}\n'
        else:
            response += f'{dessert}: {prediction}\n'
    update.message.reply_text(response)
    return ConversationHandler.END

# Handler for the /cancel command
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

# Main function
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