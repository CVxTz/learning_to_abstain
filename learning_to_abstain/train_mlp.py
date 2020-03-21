import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models import get_model_mlp
import matplotlib.pyplot as plt


def train():
    x_train = [[1, 1]] * 5000 + [[2, 2]] * 5000
    y_train = [0] * 5000 + [1] * 5000

    y_train = np.array(y_train)
    y_train = np.eye(2)[y_train, :]
    x_train = np.array(x_train)
    x_train = x_train + np.random.normal(0, 0.5, size=x_train.shape)

    x_test = [[1, 1]] * 200 + [[2, 2]] * 200 + [[-1, 4]] * 100 + [[4, -1]] * 100
    y_test = [0] * 200 + [1] * 200 + [1] * 200

    y_test = np.array(y_test)
    y_test = np.eye(2)[y_test, :]
    x_test = np.array(x_test)
    x_test = x_test + np.random.normal(0, 0.5, size=x_test.shape)

    reduce = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=50, min_lr=1e-7)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=300)
    checkpoint = ModelCheckpoint(
        "mlp.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min",
    )

    model = get_model_mlp(n_classes=2)

    for k in range(100):
        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=1,
            callbacks=[checkpoint, reduce, early],
            verbose=2,
        )

        pred = model.predict(x_test)
        class_pred = np.argmax(pred[:, :-1], axis=-1).tolist()
        abstain = pred[:, -1].tolist()

        th = np.quantile(abstain, q=0.66)

        i = np.random.randint(0, x_test.shape[0])
        print(pred[i, -1])

        for a, b in zip(pred[i, ...].tolist(), y_test[i, ...].tolist()):
            print(round(a, 2), b)

        plt.figure(figsize=(12, 12))
        s1 = plt.scatter(
            x_test[:200, 0],
            x_test[:200, 1],
            c=["r" if b > th else ("b" if a == 0 else "g") for a, b in zip(class_pred[:200], abstain[:200])],
            s=40,
            marker="o",
        )
        s2 = plt.scatter(
            x_test[200:400, 0],
            x_test[200:400, 1],
            c=["r" if b > th else ("b" if a == 0 else "g") for a, b in zip(class_pred[200:400], abstain[200:400])],
            s=40,
            marker="s",
        )
        s3 = plt.scatter(
            x_test[-200:, 0],
            x_test[-200:, 1],
            c=["r" if b > th else ("b" if a == 0 else "g") for a, b in zip(class_pred[-200:], abstain[-200:])],
            s=40,
            marker="x",
        )

        plt.legend((s1, s2, s3), ("Class 0", "Class 1", "Out of distribution"))

        plt.savefig("../output/plot_%s.jpg" % k)
        plt.close()


if __name__ == "__main__":
    train()
