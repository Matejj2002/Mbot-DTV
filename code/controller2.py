# mbot libraries
import cyberpi
import pyaudio

# machine learning libraries
import tensorflow as tf
import numpy as np

cyberpi.console.clear()
cyberpi.console.print("hello, car charmer!")

folder_path = '2023-2024-STHDF-Projects\Project_004_CarCharm_STHDF_2023-2024\code\\'
model_path = folder_path + 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print(input_shape)

# load labels.txt into class_names
class_names = []
with open(folder_path + 'labels.txt', 'r') as f:
    class_names = f.read().splitlines()
print(class_names)

print('-----\nstart playing!\n-----\n')
print('forward -- b5')
print('backward -- d6')
print('left -- e5')
print('right -- g5\n')


# define stream constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 44032

desired_duration = 0.5
frames_per_buffer = int(RATE * desired_duration)

# create an audio object
audio = pyaudio.PyAudio()

# start recording
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=frames_per_buffer)

# define mbot parameters
SPEED = 50
ANGLE = 90
TIME = 0.5

try:
    while True:
        data = stream.read(CHUNK)
        data = np.frombuffer(data, dtype=np.int16)
        data = data.astype(np.float32)
        input_data = np.expand_dims(data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], [data])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)

        # get the index of the highest confidence (if the model is confident enough)
        prediction = np.argmax(output_data) if output_data[0][np.argmax(output_data)] > 0.4 else 0
        print(output_data[0][np.argmax(output_data)])
        print(class_names[prediction])
        print('-------')

        if prediction == 1:
            cyberpi.mbot2.forward(speed = SPEED, t = TIME)
        elif prediction == 2:
            cyberpi.mbot2.backward(speed = SPEED, t = TIME)
        elif prediction == 3:
            cyberpi.mbot2.turn(angle = -ANGLE, speed = SPEED)
        elif prediction == 4:
            cyberpi.mbot2.turn(angle = ANGLE, speed = SPEED)
        else:
            print('not recognized')

except KeyboardInterrupt:

    stream.stop_stream()
    stream.close()

    # terminate the audio object
    audio.terminate()
