# src/models/autoencoder.py

from __future__ import annotations

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def build_autoencoder(
    input_size: int,
    latent_size: int,
    widths: list[int],
    learning_rate: float = 5e-5,
    seed: int = 42,
):
    """
    Build a symmetric MLP autoencoder and its encoder.

    Returns
    -------
    ae : tf.keras.Model
        Autoencoder model x -> x_hat.
    encoder : tf.keras.Model
        Encoder model x -> z.
    """
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    x_in = layers.Input(shape=(input_size,), name="x")
    x = x_in

    for i, width in enumerate(widths):
        x = layers.Dense(
            width,
            activation="relu",
            kernel_initializer=initializer,
            name=f"enc_dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)

    z = layers.Dense(
        latent_size,
        activation="tanh",
        kernel_initializer=initializer,
        name="latent",
    )(x)

    x = z
    for i, width in enumerate(reversed(widths)):
        x = layers.Dense(
            width,
            activation="relu",
            kernel_initializer=initializer,
            name=f"dec_dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)

    x_hat = layers.Dense(input_size, activation="linear", name="x_hat")(x)

    ae = models.Model(x_in, x_hat, name=f"AE_in{input_size}_z{latent_size}")
    encoder = models.Model(x_in, z, name=f"ENC_in{input_size}_z{latent_size}")

    ae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return ae, encoder


def train_autoencoder(
    X_train,
    X_val,
    latent_size: int = 16,
    widths: list[int] | None = None,
    learning_rate: float = 5e-5,
    batch_size: int = 128,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
    verbose: int = 0,
):
    """
    Train an autoencoder with early stopping on validation reconstruction loss.

    Returns
    -------
    ae : tf.keras.Model
    encoder : tf.keras.Model
    history : tf.keras.callbacks.History
    """
    if widths is None:
        widths = [128, 64, 64]

    K.clear_session()
    set_global_seed(seed)

    input_size = X_train.shape[1]

    ae, encoder = build_autoencoder(
        input_size=input_size,
        latent_size=latent_size,
        widths=widths,
        learning_rate=learning_rate,
        seed=seed,
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = ae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early],
        verbose=verbose,
    )

    return ae, encoder, history


def encode_with_model(encoder, X, batch_size: int = 2048):
    """
    Compute latent representation z = encoder(x).
    """
    return encoder.predict(X, batch_size=batch_size, verbose=0)