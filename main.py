import pyautogui
import tensorflow # main ai lib
import sklearn # used for data prepoccesing

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = keras.Sequential(
        [
            ## convolutional layer
            layers.Conv2D(16, (3, 3), activation="relu", input_shape = (30, 30, 3), name = "conv1"),
            layers.MaxPooling2D(pool_size = (2, 2), name = "maxpol1"),
            layers.Conv2D(64, (3, 3), activation="relu", name = "conv2"),
            layers.MaxPooling2D(pool_size = (2, 2), name = "maxpol2"),
            #layers.Conv2D(64, (3, 3), activation="relu", name = "conv3"),
            #layers.MaxPooling2D((2, 2), name = "maxpol3"),

            ## flatten dimensions
            layers.Flatten(),

            ## dense 1-D layers
            layers.Dense(1028, activation = "relu", name = "layer1"),
            layers.Dropout(0.3),
            layers.Dense(512, activation = "relu", name = "layer2"),
            layers.Dropout(0.2),
            layers.Dense(256, activation = "relu", name = "layer3"),

            ## output layer
            layers.Dense(NUM_CATEGORIES, activation = "softmax", name = "output")
            #layers.Dense(NUM_CATEGORIES, activation = "relu", name = "outputlayer")
        ]
    )
    model.compile(optimizer='adam',
        loss="categorical_crossentropy", #use this because get some error "logits and labels must have the same first
                                        # dimension, got logits shape [32,3] and labels shape [96]"
        #loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    # model.compile(
    #     optimizer=keras.optimizers.RMSprop(),  # Optimizer
    #     # Loss function to minimize
    #     loss=keras.losses.SparseCategoricalCrossentropy(),
    #     # List of metrics to monitor
    #     metrics=[keras.metrics.SparseCategoricalAccuracy()],
    # )
    return model


## split data

## compile model
model = get_model()

## fit model
model.fit(x, y, epochs=EPOCHS)

## evaluate model




