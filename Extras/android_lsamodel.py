import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty

import tensorflow as tf
import numpy as np


class SignLanguageApp(App):

    camera = ObjectProperty()
    labels = ['A', 'B', 'C', 'D']
    interpreter = None
    input_details = None
    output_details = None

    def build(self):
        self.camera = Camera()
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        self.interpreter = tf.lite.Interpreter(model_path="Tensorflow/workspace/models/tflite_models/converted_model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        return self.camera
    
    def update(self, dt):

        print("Update called")

        frame = self.camera.texture
        if frame is None:
            return

        w, h = frame.size

        # Convert the texture to a numpy array
        buf = frame.pixels
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

        # Preprocess the image
        image = tf.image.resize(image, [224, 224])
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # Run inference on the model
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Get the predicted label
        predicted_label = self.labels[np.argmax(output_data)]
        self.on_start(predicted_label)

    def on_start(self, predicted_label):
        print("on_start called with predicted_label:", predicted_label)
        if not self.camera:
            print("Camera not found")
            return
        

        if not self.camera.texture:
            print("Binding update_texture")
            self.camera.bind(on_texture=lambda *args: self.update_texture(self.camera, self.camera.texture, predicted_label))
            self.camera.play()

        else:
            print("Calling update_texture")
            self.update_texture(self.camera, self.camera.texture, predicted_label)

    def update_texture(self, camera, texture, predicted_label):
        print("update_texture called with predicted_label:", predicted_label)
        self.camera.texture = texture

        # Draw the predicted label on the texture
        texture = self.draw_label(texture, predicted_label)

        self.camera.texture = texture

    def draw_label(self, texture, label):
        # Create a new texture to draw on
        new_texture = Texture.create(size=(texture.size[0], texture.size[1]), colorfmt='rgba')

        # Create a canvas and set the texture as its background
        with new_texture:
            kivy.graphics.Color(1, 1, 1, 1)
            kivy.graphics.Rectangle(pos=(0, 0), size=(texture.size[0], texture.size[1]), texture=texture)

            # Draw the predicted label
            kivy.graphics.Color(1, 0, 0, 1)
            kivy.graphics.Rectangle(pos=(0, 0), size=(texture.size[0], texture.size[1] // 8))
            kivy.graphics.Color(0, 0, 0, 1)
            kivy.graphics.Label(text=label, font_size=texture.size[1] // 16, halign='center',
                                pos=(texture.size[0] // 2, texture.size[1] // 16))

        return new_texture
    
class MainApp(App):
    def build(self):
        return SignLanguageApp().build()


if __name__ == '__main__':
    MainApp().run()
