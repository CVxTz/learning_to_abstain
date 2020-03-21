from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.datasets import mnist
from models import get_model_mnist
import numpy as np


def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[..., np.newaxis].astype(np.float32) / 255

    x_test = x_test[..., np.newaxis].astype(np.float32) / 255

    y_train = np.eye(10)[y_train, :]

    y_test = np.eye(10)[y_test, :]

    reduce = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=50, min_lr=1e-7)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=300)
    checkpoint = ModelCheckpoint(
        "mnist.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min",
    )

    model = get_model_mnist(n_classes=10)

    for _ in range(100):
        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=1,
            callbacks=[checkpoint, reduce, early],
            use_multiprocessing=True,
            workers=8,
            verbose=2,
        )

        pred = model.predict(x_test[:100, ...])

        i = np.random.randint(0, 100)
        print(pred[i, -1])

        for a, b in zip(pred[i, ...].tolist(), y_test[i, ...].tolist()):
            print(round(a, 2), b)


if __name__ == "__main__":

    train()
