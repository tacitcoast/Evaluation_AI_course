predictions = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]

true_labels = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1]

email_subjects = [
    "Поздравляем! Вы выиграли миллион!",
    "Встреча завтра в 15:00",
    "Срочно! Переведите деньги",
    "Отчет по проекту",
    "Счет за интернет",
    "Новости компании",
    "Кликните здесь и получите приз!",
    "Приглашение на конференцию",
    "Специальное предложение только сегодня!",
    "График отпусков",
    "Скидка 90% на все товары!",
    "Акция! Звоните прямо сейчас!",
    "Протокол совещания",
    "Уважаемый клиент, срочное сообщение",
    "Семинар по машинному обучению"
]


def calculate_accuracy(predictions, true_labels):
    if len(predictions) != len(true_labels):
        raise ValueError("Number of predictions must match number of labels")

    correct_count = 0
    for pred, true in zip(predictions, true_labels):
        if pred == true:
            correct_count += 1

    return correct_count / len(predictions)


total_emails = len(email_subjects)
print(f"Total emails: {total_emails}")

predicted_spam_count = 0
for prediction in predictions:
    if prediction == 1:
        predicted_spam_count += 1

print(f"Predicted as spam: {predicted_spam_count}")

actual_spam_count = 0
for true_label in true_labels:
    if true_label == 1:
        actual_spam_count += 1

print(f"Actually spam: {actual_spam_count}")

accuracy = calculate_accuracy(predictions, true_labels)
print(f"Model accuracy: {accuracy:.3f}")
print(f"Model accuracy in percent: {accuracy * 100:.1f}%")

print("")
print("Model errors:")

errors_count = 0
for i in range(len(predictions)):
    if predictions[i] != true_labels[i]:
        errors_count += 1

        if predictions[i] == 1:
            predicted_text = "spam"
        else:
            predicted_text = "not spam"

        if true_labels[i] == 1:
            actual_text = "spam"
        else:
            actual_text = "not spam"

        print(f"Email #{i + 1}: {email_subjects[i]}")
        print(f"   Predicted: {predicted_text}")
        print(f"   Actual: {actual_text}")

print("")
print(f"Total errors: {errors_count} out of {total_emails}")
