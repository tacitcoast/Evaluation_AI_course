import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from classification import (
    device,
    load_mnist_data,
    VerySimpleCNN,
    SimpleCNN,
    train_model,
    evaluate_model_with_metrics,
    calculate_confidence_scores,
    calculate_top_k_accuracy,
    calculate_entropy,
    calculate_calibration_error,
    plot_training_history,
    plot_confusion_matrix,
)

NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

plot_name = 'plot'


def save_instead_of_show(*args, **kwargs):
    filename = f"{plot_name}.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {filename}")


plt.show = save_instead_of_show


def run_one_model(name, model_class, train_loader, val_loader, test_loader, classes):
    global plot_name

    print("")
    print("=" * 80)
    print(f"MODEL: {name}")
    print("=" * 80)

    model = model_class(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("")
    print("Training...")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS
    )

    print("")
    print("Final evaluation on the test set...")
    predictions, true_labels, softmax_probs, logits = evaluate_model_with_metrics(
        model, test_loader, device, classes
    )

    test_accuracy = accuracy_score(true_labels, predictions) * 100
    f1_macro = f1_score(true_labels, predictions, average='macro')
    confidence_scores = calculate_confidence_scores(softmax_probs)
    entropy_values = calculate_entropy(softmax_probs)
    ece, bin_data = calculate_calibration_error(softmax_probs, predictions, true_labels)

    print("")
    print(f"RESULTS FOR {name}")
    print("-" * 60)
    print(f"Best validation accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"Test accuracy:            {test_accuracy:.2f}%")
    print(f"F1 macro:                 {f1_macro:.4f}")
    print(f"Epochs actually run:      {len(history['train_losses'])}")
    for k in [1, 3, 5]:
        print(f"Top-{k} accuracy:           {calculate_top_k_accuracy(softmax_probs, true_labels, k=k):.2f}%")
    print(f"Average confidence:       {np.mean(confidence_scores):.3f}")
    print(f"Average entropy:          {np.mean(entropy_values):.3f}")
    print(f"Calibration error (ECE):  {ece:.3f}")

    print("")
    print("Per-class report:")
    print(classification_report(true_labels, predictions, target_names=classes, digits=3))

    cm = confusion_matrix(true_labels, predictions)
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], classes[i], classes[j]))
    confusions.sort(reverse=True)

    print("Most frequent confusions:")
    for count, actual, predicted in confusions[:5]:
        print(f"  digit {actual} predicted as {predicted}: {count} times")

    plot_name = f"{name}_training"
    plot_training_history(history)

    plot_name = f"{name}_confusion"
    plot_confusion_matrix(true_labels, predictions, classes)

    return {
        'Model': name,
        'Parameters': total_params,
        'Test accuracy': test_accuracy,
        'F1 macro': f1_macro,
        'Best val accuracy': history['best_val_accuracy'],
        'Epochs': len(history['train_losses']),
        'Avg confidence': np.mean(confidence_scores),
        'Avg entropy': np.mean(entropy_values),
        'ECE': ece
    }


def main():
    print("=" * 80)
    print("HOMEWORK 2: MNIST CLASSIFICATION WITH TWO CNN ARCHITECTURES")
    print("=" * 80)

    print("")
    print("Loading MNIST...")
    train_loader, val_loader, test_loader, classes = load_mnist_data(
        batch_size=BATCH_SIZE, val_split=0.1)
    print(f"Classes: {classes}")

    results = []
    results.append(run_one_model('VerySimpleCNN', VerySimpleCNN,
                                 train_loader, val_loader, test_loader, classes))
    results.append(run_one_model('SimpleCNN', SimpleCNN,
                                 train_loader, val_loader, test_loader, classes))

    print("")
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    header = f"{'Model':<16}{'Params':>12}{'Test acc':>11}{'F1 macro':>11}{'Val acc':>10}{'Epochs':>8}{'Conf':>8}{'Entropy':>9}{'ECE':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Model']:<16}{r['Parameters']:>12,}{r['Test accuracy']:>10.2f}%"
              f"{r['F1 macro']:>11.4f}{r['Best val accuracy']:>9.2f}%{r['Epochs']:>8}"
              f"{r['Avg confidence']:>8.3f}{r['Avg entropy']:>9.3f}{r['ECE']:>8.3f}")


if __name__ == '__main__':
    main()
