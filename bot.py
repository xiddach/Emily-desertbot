import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup
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
DATE, DESSERTS_INPUT, PREDICT = range(3)

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
    update.message.reply_text('Привіт! Я бот для прогнозування замовлень десертів.\n'
                              'Будь ласка, введіть дату дня, для якого ви хочете ввести залишки (формат: YYYY-MM-DD):')
    return DATE

# Обробник дати
def get_date(update: Update, context: CallbackContext) -> int:
    try:
        input_date = update.message.text
        date = datetime.strptime(input_date, '%Y-%m-%d').date()
        context.user_data['date'] = date
        context.user_data['desserts'] = {}
        
        update.message.reply_text(f'Дякую! Тепер давайте введемо залишки для кожного десерту.')
        return ask_next_dessert(update, context)
    except ValueError:
        update.message.reply_text('Неправильний формат дати. Будь ласка, введіть дату у форматі YYYY-MM-DD:')
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
    data = data.append(new_row, ignore_index=True)
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
    next_date = date + timedelta(days=1)
    
    # Формуємо повідомлення з прогнозами
    response = f'Прогнозовані замовлення на {next_date}:\n'
    for dessert, prediction in predictions.items():
        response += f'{dessert}: {prediction}\n'
    
    update.message.reply_text(response)
    
    return ConversationHandler.END

# Обробник команди /cancel
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Операція скасована.')
    return ConversationHandler.END

def main() -> None:
    # Токен вашого Telegram-бота
    TOKEN = 7631205077:AAFt7ryCShzyBA43ou-7IldEWGsAO0TyB9E
    
    # Створюємо Updater і передаємо йому токен бота
    updater = Updater(TOKEN)

    # Отримуємо диспетчер для реєстрації обробників
    dispatcher = updater.dispatcher

    # Налаштовуємо ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            DATE: [MessageHandler(Filters.text & ~Filters.command, get_date)],
            DESSERTS_INPUT: [MessageHandler(Filters.text & ~Filters.command, handle_dessert_input)],
            PREDICT: [MessageHandler(Filters.text & ~Filters.command, finalize_data)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Запускаємо бота
    updater.start_polling()

    # Робимо бота активним до моменту зупинки
    updater.idle()

if __name__ == '__main__':
    main()
