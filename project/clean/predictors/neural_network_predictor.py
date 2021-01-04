from project.clean.predictors.predictor import Predictor
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2 as cv


class NeuralNetworkPredictor(Predictor):

    def __init__(self):
        super().__init__()
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Dense(364, activation='relu', input_dim=self.hog_output_shape),
            BatchNormalization(),
            Dropout(0.5),
            Dense(52, activation='relu', input_dim=self.hog_output_shape),
            BatchNormalization(),
            Dense(9, activation='softmax'),

        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.load_weights('resources/weights.h5')
        return model

    def pre_process(self, image):
        return image

    def predict(self, image):
        # cv.imshow('test ', image)
        # cv.waitKey(0)
        histogram = self.hog_compute(self.pre_process(image))
        return self.model.predict(histogram)
