from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.activations import relu, softmax, linear
from tensorflow.python.keras import backend as K


class RLModel:


    def __init__(self):
        self.frames = 4


        self._build_model()

    def _build_model(self):

        input_image = Input(shape=(84, 84, self.frames))
        x = Conv2D(32, (8, 8), (2, 2), 'same', activation=relu)(input_image)
        x = Conv2D(64, (4, 4), (2, 2), 'same', activation=relu)(x)
        x = Conv2D(128, (2, 2), (2, 2), 'same', activation=relu)(x)
        x = Conv2D(256, (1, 1), (2, 2), 'same', activation=relu)(x)
        x = Flatten()(x)

        x = Dense(512, activation=relu)(x)

        out_mouse = Dense(2, activation=softmax)(x)
        out_action = Dense(4, activation=linear)(x)



        model = Model(
            inputs=[input_image],
            outputs=[out_mouse, out_action]
        )

        model.summary()



    def proximal_policy_optimization_loss(actual_value, predicted_value, old_prediction):
        advantage = actual_value - predicted_value

        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred)
            old_prob = K.sum(y_true * old_prediction)
            r = prob/(old_prob + 1e-10)

            return -K.log(prob + 1e-10) * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))
        return loss







if __name__ == "__main__":

    RLModel()

