import cyberpi
import pyaudio
import tensorflow as tf
import numpy as np
import time

# Initialize console and print welcome message
cyberpi.console.clear()
cyberpi.console.print("hello, car charmer!")

# Define paths
model_path = 'model.tflite'
labels_path = 'labels.txt'

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(f'Input Shape: {input_shape}')

# Load labels
with open(labels_path, 'r') as f:
    class_names = f.read().splitlines()
print(f'Class Names: {class_names}')

# Print command instructions
print('-----\nHrame!\n-----\n')
print('dopredu -- V')
print('dozadu -- Z')
print('dolava -- L')
print('doprava -- P\n')

# Define audio stream constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 44032
DESIRED_DURATION = 0.5
FRAMES_PER_BUFFER = int(RATE * DESIRED_DURATION)

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

SPEED = 50
ANGLE = 90
TIME = 0.5


def predict_command(interpreter, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    confidence = output_data[0][np.argmax(output_data)]
    prediction = np.argmax(output_data) if confidence > 0.4 else 0
    return prediction, confidence


try:
    while True:
        data = stream.read(CHUNK)
        data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        input_data = np.expand_dims(data, axis=0)

        prediction, confidence = predict_command(interpreter, input_data)

        print(f'Confidence: {confidence}')
        print(f'Prediction: {class_names[prediction]}')
        print('-------')

        if prediction == 3:
            cyberpi.led.show('white white white white white')
            cyberpi.mbot2.forward(speed=SPEED, t=TIME)
            cyberpi.console.clear()
            cyberpi.console.print("dopredu")

        elif prediction == 4:
            cyberpi.led.show('red red red red red')
            cyberpi.mbot2.backward(speed=SPEED, t=TIME)
            cyberpi.console.clear()
            cyberpi.console.print("dozadu")


        elif prediction == 1:
            cyberpi.led.show('orange orange white white white')
            cyberpi.mbot2.turn(angle=-ANGLE, speed=SPEED)
            cyberpi.console.clear()
            cyberpi.console.print("dolava")


        elif prediction == 2:
            cyberpi.led.show('white white white orange orange')
            cyberpi.mbot2.turn(angle=ANGLE, speed=SPEED)
            cyberpi.console.clear()
            cyberpi.console.print("doprava")

        else:
            print('Command not recognized')

        #time.sleep(0.1)

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    audio.terminate()
