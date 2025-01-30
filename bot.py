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
    explanations = {}
    for dessert in DESSERTС:
        try:
            prediction = predict_orders(dessert, data)
            predictions[dessert] = prediction
            # Додатковий аналіз для пояснення прогнозу
            dessert_data = data[['date', dessert]].dropna()
            dessert_data['date'] = pd.to_datetime(dessert_data['date'], format='%d.%m.%Y')
            dessert_data = dessert_data.sort_values(by='date')
            if len(dessert_data) < 14:
                available_days = len(dessert_data)
                last_days = dessert_data.tail(available_days)
            else:
                last_days = dessert_data.tail(14)
            X = np.array(range(len(last_days))).reshape(-1, 1)
            y = np.array(last_days[dessert])
            # Навчаємо модель Random Forest з меншою кількістю дерев
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Паралельне обчислення
            model.fit(X, y)
            # Прогнозуємо замовлення на наступний день
            next_day = X[-1][0] + 1
            predicted_order = model.predict([[next_day]])[0]
            rounded_prediction = round(predicted_order)
            
            # Пояснення прогнозу
            explanation = f"Прогнозовано {rounded_prediction} одиниць {dessert} на {date} на основі аналізу останніх {len(last_days)} днів.\n"
            explanation += f"Останні доступні дані показують наступні значення:\n"
            for idx, row in last_days.iterrows():
                explanation += f"  {row['date'].strftime('%d.%m.%Y')}: {row[dessert]}\n"
            explanation += f"Модель Random Forest визначила, що на наступний день ({next_day}) очікується приблизно {rounded_prediction} одиниць."
            explanations[dessert] = explanation
        except ValueError as e:
            predictions[dessert] = str(e)
            explanations[dessert] = f"Недостатньо даних для прогнозування {dessert}. Необхідно більше інформації."
    # Розраховуємо дату прогнозу
    next_date = datetime.strptime(date, '%d.%m.%Y').date() + timedelta(days=1)
    # Формуємо повідомлення з прогнозами
    response = f'Прогнозовані замовлення на {next_date.strftime("%d.%m.%Y")}:\n'
    for dessert, prediction in predictions.items():
        response += f'{dessert}: {prediction}\n'
        response += f"{explanations[dessert]}\n"
    update.message.reply_text(response)
    return ConversationHandler.END
