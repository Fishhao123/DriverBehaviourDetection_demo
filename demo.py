import matplotlib.pyplot as plt
from keras.layers import *
from keras.callbacks import *
from keras.applications import *
from keras.models import Model
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import load_img, img_to_array
import time
import argparse

class Predicting:
    def __init__(self):
        self.inception_res_v2_init() #Initialize the model structure
        self.class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left',
                        'talking_on_phone_left', 'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
        while True:
            img_path = input("\033[0;37;42m Please input the image PATH: \033[0m")
            try:
                img = load_img(img_path, target_size=(299, 299))
            except Exception as e:
                print("\033[0;37;41m Wrong PATH. Check your previous input. \033[0m")
            else:
                self.predict_image(img)

    def inception_res_v2_init(self):
        MODEL = InceptionResNetV2
        size = (299, 299)
        drop = 0.5
        weights_path = "./model_weights/InceptionResNetV2.hdf5"
        preprocess_function = inception_resnet_v2.preprocess_input
        self.init_model(size, MODEL, drop, weights_path, preprocess_function)

    def init_model(self, size, MODEL, drop, weights_path, preprocess_function):
        width = size[0]
        height = size[1]
        input_tensor = Input((height, width, 3))
        x = Lambda(preprocess_function)(input_tensor)

        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)  # 初始权重采用ImageNet
        new_output = GlobalAveragePooling2D()(base_model.output)
        new_output = Dropout(drop)(new_output)
        new_output = Dense(10, activation="softmax")(new_output)
        self.model = Model(base_model.input, new_output)
        self.model.load_weights(weights_path)

    # Predict from single picture
    def predict_image(self, img):
        tik = time.time()
        img_array = np.expand_dims(img_to_array(img), axis=0)
        result = self.model.predict(img_array).clip(min=0.005, max=0.995).tolist()[0]
        tok = time.time()
        category_index = result.index(max(result))
        category = self.class_labels[category_index]
        print("Category：{}, Probabilty: {}".format(category, result[category_index]))
        print("Time Cost: {} s".format((time.time() - tik)))
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    predicting = Predicting()
