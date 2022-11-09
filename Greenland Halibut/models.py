import tensorflow as tf


def get_full_model(img_model, cat_model, image_shape):

    cat_input = tf.keras.Input(shape=(1,), name="gender", dtype="string")
    categories = cat_model(cat_input)

    img_input = tf.keras.layers.Input(image_shape)
    predictions = img_model(img_input)

    inputs = [cat_input, img_input]

    outputs = tf.keras.layers.Dot(axes=1)([predictions, categories])

    _model = tf.keras.models.Model(inputs, outputs)

    return _model


def get_categorical_model(vocabulary):

    index_layer = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=vocabulary
    )

    encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        num_tokens=index_layer.vocabulary_size(), output_mode="binary"
    )

    inputs = tf.keras.Input(shape=(1,), name="gender", dtype="string")

    outputs = encoding_layer(index_layer(inputs))

    _model = tf.keras.models.Model(inputs, outputs)

    return _model


def get_deployable_model(image_shape, initial_bias, dropout_rate):

    base_model = tf.keras.applications.Xception(
        input_shape=image_shape, include_top=False, pooling="avg"
    )

    inputs = tf.keras.layers.Input(image_shape)

    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = tf.keras.layers.RandomTranslation(0, 0.1)(x)
    x = tf.keras.layers.RandomRotation(0.1, fill_mode="constant")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(
        4, "relu", bias_initializer=tf.keras.initializers.Constant(initial_bias)
    )(x)

    _model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return _model
