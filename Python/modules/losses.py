import tensorflow as tf


def normal_divergence(mean1, mean2, sigma1, sigma2):
    """
    Compute KL-divergence for univariate normal distributions.
    """
    div = tf.math.log(sigma2 / sigma1) + (sigma1 ** 2 + (mean1 - mean2) ** 2) / (2 * sigma2 ** 2) - 1 / 2

    return div


class MeanSquaredErrorKLD(tf.keras.losses.Loss):
    """
    Custom loss using MSE and KL-Divergence.
    """
    def call(self, y_true, output):

        mask = output[:, 2:]
        y_pred = tf.reshape(output[:, 0], (-1, 1))

        _y_true = tf.cast(y_true * tf.cast(mask, y_true.dtype), tf.float64)
        _y_pred = tf.cast(y_pred * tf.cast(mask, y_pred.dtype), tf.float64)

        n_true = tf.reduce_sum(tf.cast(_y_true != 0, _y_true.dtype), axis=0)
        n_pred = tf.reduce_sum(tf.cast(_y_pred != 0, _y_pred.dtype), axis=0)

        mean_true = tf.reduce_sum(_y_true, axis=0) / n_true
        mean_pred = tf.reduce_sum(_y_pred, axis=0) / n_pred

        std_true = tf.reduce_sum((_y_true - mean_true * tf.cast(mask, _y_true.dtype)) ** 2, axis=0) / n_true
        std_pred = tf.reduce_sum((_y_pred - mean_pred * tf.cast(mask, _y_true.dtype)) ** 2, axis=0) / n_pred

        gamma = tf.constant(1, dtype=tf.float64)

        kld = gamma * tf.reduce_sum(normal_divergence(mean_true, mean_pred, std_true, std_pred))

        return tf.keras.losses.mean_squared_error(y_true, y_pred) + tf.cast(kld, tf.float32)
