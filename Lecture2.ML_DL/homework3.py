import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from regression import DataLoader, ModelTrainer, ModelEvaluator, DataVisualizer

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.01
RUNS = 5

plot_name = 'plot'


def save_instead_of_show(*args, **kwargs):
    filename = f"{plot_name}.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {filename}")


plt.show = save_instead_of_show


def build_model(input_dim, units, learning_rate=LEARNING_RATE):
    model = keras.Sequential([
        layers.Dense(units[0], activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(units[1], activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(units[2], activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


def run_experiment(name, units, loader, X_train, X_val, X_test, y_train, y_val, y_test,
                   learning_rate=LEARNING_RATE, make_plot=True):
    global plot_name

    print("")
    print("=" * 70)
    print(f"{name}: Dense layers {units[0]} - {units[1]} - {units[2]} - 1, lr {learning_rate}")
    print("=" * 70)

    model = build_model(X_train.shape[1], units, learning_rate)
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    model.summary()

    trainer = ModelTrainer()
    history = trainer.train_model(model, name, X_train, y_train, X_val, y_val,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE)

    evaluator = ModelEvaluator(loader.scaler_y, target_scale=loader.target_scale)
    metrics = evaluator.evaluate_model(model, name, X_test, y_test,
                                       trainer.training_times[name])
    evaluator.print_metrics(name)

    if make_plot:
        plot_name = f"diamonds_{name.replace(' ', '_')}"
        DataVisualizer(loader).plot_results(trainer, evaluator)

    val_losses = history.history['val_loss']
    best_epoch = int(np.argmin(val_losses)) + 1
    print(f"Best epoch: {best_epoch} of {len(val_losses)}")

    return {
        'Model': name,
        'Dense layers': f"{units[0]}-{units[1]}-{units[2]}-1",
        'LR': learning_rate,
        'Best epoch': best_epoch,
        'Parameters': total_params,
        'RMSE': metrics['rmse'],
        'MAE': metrics['mae'],
        'R2': metrics['r2'],
        'Epochs': len(history.history['loss']),
        'Training time': trainer.training_times[name],
        'y_test': metrics['y_test_original'],
        'y_pred': metrics['y_pred_original']
    }


def errors_by_price(result):
    y_test = result['y_test']
    y_pred = result['y_pred']
    errors = np.abs(y_test - y_pred)

    edges = [0, 1000, 2500, 5000, 10000, 20000]
    labels = ['under $1k', '$1k-2.5k', '$2.5k-5k', '$5k-10k', 'over $10k']

    rows = []
    for i, label in enumerate(labels):
        mask = (y_test >= edges[i]) & (y_test < edges[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            'Price range': label,
            'Diamonds': int(mask.sum()),
            'Mean price': y_test[mask].mean(),
            'MAE': errors[mask].mean(),
            'MAE % of price': errors[mask].mean() / y_test[mask].mean() * 100
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("HOMEWORK 3: DIAMONDS PRICE REGRESSION")
    print("=" * 70)

    loader = DataLoader()
    df = loader.load_diamonds()
    print("")
    print(f"Rows: {len(df)}, features: {df.shape[1] - 1}")
    print(f"Price from ${df['target'].min():,} to ${df['target'].max():,}, "
          f"mean ${df['target'].mean():,.0f}")

    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data()

    configs = [
        ('Baseline', (64, 32, 16), 0.01),
        ('Tuned', (256, 128, 64), 0.01),
        ('Tuned lr 0.001', (256, 128, 64), 0.001),
    ]

    all_runs = []
    keep = {}
    for name, units, lr in configs:
        for run in range(1, RUNS + 1):
            r = run_experiment(f"{name}", units, loader,
                               X_train, X_val, X_test, y_train, y_val, y_test,
                               learning_rate=lr, make_plot=(run == 1))
            r['Run'] = run
            all_runs.append(r)
            if run == 1:
                keep[name] = r

    df_runs = pd.DataFrame([{k: r[k] for k in
                             ['Model', 'Run', 'RMSE', 'MAE', 'R2', 'Epochs', 'Best epoch']}
                            for r in all_runs])

    print("")
    print("=" * 70)
    print(f"ALL RUNS ({RUNS} per configuration)")
    print("=" * 70)
    print(df_runs.to_string(index=False, float_format='%.4f'))

    print("")
    print("=" * 70)
    print("AVERAGE OVER RUNS")
    print("=" * 70)
    summary = df_runs.groupby('Model', sort=False).agg(
        RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
        MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
        R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
        R2_min=('R2', 'min'), R2_max=('R2', 'max')).reset_index()
    print(summary.to_string(index=False, float_format='%.4f'))

    for name in keep:
        print("")
        print(f"Errors by price range, {name} (run 1):")
        print(errors_by_price(keep[name]).to_string(index=False, float_format='%.1f'))


if __name__ == '__main__':
    main()
