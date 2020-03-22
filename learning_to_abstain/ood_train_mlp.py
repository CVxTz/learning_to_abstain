import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models import get_model_mlp
import matplotlib.pyplot as plt


def train():
    x_train = [[0, 1]] * 5000 + [[3, 2]] * 5000
    y_train = [0] * 5000 + [1] * 5000

    y_train = np.array(y_train)
    y_train = np.eye(2)[y_train, :]
    x_train = np.array(x_train)
    x_train = x_train + np.random.normal(0, 0.5, size=x_train.shape)

    x_test = [[0, 1]] * 200 + [[3, 2]] * 200 + [[-1, 4]] * 100 + [[4, -1]] * 100
    y_test = [0] * 200 + [1] * 200 + [1] * 200

    y_test = np.array(y_test)
    y_test = np.eye(2)[y_test, :]

    x_test = np.array(x_test).astype(float)

    x_test[:400, :2] = x_test[:400, :2] + np.random.normal(0, 0.5, size=(400, 2))

    x_test[400:, :2] = x_test[400:, :2] + np.random.normal(0, 0.2, size=(200, 2))

    reduce = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=50, min_lr=1e-7)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=300)
    checkpoint = ModelCheckpoint(
        "mlp.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min",
    )

    xg, yg = np.mgrid[-3:6:50j, -3:6:50j]
    g_sample = np.vstack([xg.ravel(), yg.ravel()]).T

    model = get_model_mlp(n_classes=2)

    for k in range(20):

        ood_x_train = np.random.uniform(-10, 10, size=x_train.shape)
        ood_y_train = np.ones(y_train.shape) / 2

        X = np.concatenate([x_train, ood_x_train], axis=0)
        Y = np.concatenate([y_train, ood_y_train], axis=0)

        model.fit(
            X,
            Y,
            validation_data=(x_test, y_test),
            epochs=5,
            callbacks=[checkpoint, reduce, early],
            verbose=2,
        )

        y_grid_sample = model.predict(g_sample)
        y_grid_max = np.max(y_grid_sample, axis=-1)
        y_grid_max = y_grid_max.reshape(xg.shape)

        pred = model.predict(x_test)

        abstain = np.max(pred, axis=-1).tolist()

        class_pred = np.argmax(pred, axis=-1).tolist()

        # th = np.quantile(abstain, q=0.1)
        # print("th", th)

        th = 0.9

        i = np.random.randint(0, x_test.shape[0])
        print(pred[i, -1])

        for a, b in zip(pred[i, ...].tolist(), y_test[i, ...].tolist()):
            print(round(a, 2), b)

        plt.figure(figsize=(12, 12))

        cntr2 = plt.contourf(xg, yg, y_grid_max, levels=14, cmap="RdBu_r")

        s1 = plt.scatter(
            x_test[:200, 0],
            x_test[:200, 1],
            c=[
                "r" if b < th else ("b" if a == 0 else "g")
                for a, b in zip(class_pred[:200], abstain[:200])
            ],
            s=40,
            marker="o",
        )
        s2 = plt.scatter(
            x_test[200:400, 0],
            x_test[200:400, 1],
            c=[
                "r" if b < th else ("b" if a == 0 else "g")
                for a, b in zip(class_pred[200:400], abstain[200:400])
            ],
            s=40,
            marker="^",
        )
        s3 = plt.scatter(
            x_test[-200:, 0],
            x_test[-200:, 1],
            c=[
                "r" if b < th else ("b" if a == 0 else "g")
                for a, b in zip(class_pred[-200:], abstain[-200:])
            ],
            s=40,
            marker="x",
        )

        plt.legend((s1, s2, s3), ("Class 0", "Class 1", "Out of distribution"))
        plt.colorbar(cntr2)
        plt.savefig("../output/ood_plot_%s.jpg" % k)
        plt.close()


if __name__ == "__main__":
    train()
