import csv


experiment_data = [
    ["experiment_id", "model_name", "accuracy", "training_time", "dataset_size"],
    ["exp_001", "BERT-small", 0.847, 45.2, 1000],
    ["exp_002", "GPT-2", 0.923, 120.8, 1000],
    ["exp_003", "BERT-large", 0.891, 180.5, 1000],
    ["exp_004", "Random-Forest", 0.756, 12.1, 1000],
    ["exp_005", "BERT-small", 0.852, 44.8, 2000],
    ["exp_006", "GPT-2", 0.945, 125.3, 2000],
    ["exp_007", "BERT-large", 0.912, 182.1, 2000],
    ["exp_008", "Random-Forest", 0.778, 23.4, 2000]
]

with open("experiments_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for row in experiment_data:
        writer.writerow(row)

print("Data saved to experiments_data.csv")


def read_experiments(filename):
    experiments = []

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            experiment = {
                "experiment_id": row[0],
                "model_name": row[1],
                "accuracy": float(row[2]),
                "training_time": float(row[3]),
                "dataset_size": int(row[4])
            }
            experiments.append(experiment)

    return experiments


experiments = read_experiments("experiments_data.csv")

print(f"Loaded experiments: {len(experiments)}")
for experiment in experiments:
    print(f"{experiment['experiment_id']}: {experiment['model_name']} - accuracy {experiment['accuracy']:.3f}")

high_accuracy_experiments = []
for experiment in experiments:
    if experiment["accuracy"] > 0.9:
        high_accuracy_experiments.append(experiment)

print("")
print(f"Experiments with accuracy above 0.9: {len(high_accuracy_experiments)}")
for experiment in high_accuracy_experiments:
    print(f"{experiment['experiment_id']}: {experiment['model_name']} - accuracy {experiment['accuracy']:.3f}")

bert_experiments = []
for experiment in experiments:
    if "BERT" in experiment["model_name"]:
        bert_experiments.append(experiment)

print("")
print(f"BERT experiments: {len(bert_experiments)}")
for experiment in bert_experiments:
    print(f"{experiment['experiment_id']}: {experiment['model_name']} - accuracy {experiment['accuracy']:.3f}")


def save_report(experiments, high_accuracy_experiments, bert_experiments, filename):
    total_accuracy = 0
    for experiment in experiments:
        total_accuracy += experiment["accuracy"]

    average_accuracy = total_accuracy / len(experiments)

    best_experiment = experiments[0]
    for experiment in experiments:
        if experiment["accuracy"] > best_experiment["accuracy"]:
            best_experiment = experiment

    bert_accuracy = 0
    for experiment in bert_experiments:
        bert_accuracy += experiment["accuracy"]

    average_bert_accuracy = bert_accuracy / len(bert_experiments)

    with open(filename, "w") as file:
        file.write("EXPERIMENT REPORT\n")
        file.write("\n")
        file.write(f"Total experiments: {len(experiments)}\n")
        file.write(f"Average accuracy: {average_accuracy:.3f}\n")
        file.write(f"Best experiment: {best_experiment['experiment_id']} ({best_experiment['model_name']}) - accuracy {best_experiment['accuracy']:.3f}\n")
        file.write("\n")
        file.write(f"Experiments with accuracy above 0.9: {len(high_accuracy_experiments)}\n")
        for experiment in high_accuracy_experiments:
            file.write(f"{experiment['experiment_id']}: {experiment['model_name']} - accuracy {experiment['accuracy']:.3f}\n")
        file.write("\n")
        file.write(f"BERT experiments: {len(bert_experiments)}\n")
        for experiment in bert_experiments:
            file.write(f"{experiment['experiment_id']}: {experiment['model_name']} - accuracy {experiment['accuracy']:.3f}\n")
        file.write(f"Average BERT accuracy: {average_bert_accuracy:.3f}\n")

    print(f"Report saved to {filename}")


save_report(experiments, high_accuracy_experiments, bert_experiments, "report.txt")

print("")
with open("report.txt", "r") as file:
    print(file.read())
