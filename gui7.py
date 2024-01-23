import tkinter as tk
from tkinter import filedialog
from tkinter import *
from pydub import AudioSegment
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import cv2
import os

class MultiModalEmotionDetectorApp:
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Emotion Detector")
        self.root.geometry("800x600")
        self.root.configure(background='#CDCDCD')

        self.heading = Label(root, text='Multimodal Emotion Detector', pady=20, font=('arial', 25, 'bold'))
        self.heading.configure(background='#CDCDCD', foreground="#364156")
        self.heading.pack()

        self.label_image = Label(root, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.label_image.pack(side='top', pady=20, expand='True')

        self.button_upload_image = Button(root, text="Upload Image", command=self.upload_image, padx=10, pady=5)
        self.button_upload_image.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
        self.button_upload_image.pack(side='top', pady=10)

        self.button_upload_audio = Button(root, text="Upload Audio", command=self.upload_audio, padx=10, pady=5)
        self.button_upload_audio.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
        self.button_upload_audio.pack(side='bottom', pady=10)

        self.button_detect_emotion = Button(root, text="Detect Emotion", command=self.detect_emotion, state=tk.DISABLED)
        self.button_detect_emotion.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
        self.button_detect_emotion.pack(side='right', padx=20)

        self.button_reset = Button(root, text="Reset", command=self.reset, padx=10, pady=5)
        self.button_reset.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
        self.button_reset.pack(side='left', padx=20)

        self.result_label = Label(root, text="", font=('arial', 20), wraplength=600)
        self.result_label.pack(pady=20, expand='True')

        self.loaded_model_image = None
        self.loaded_model_audio = None
        self.image_path = None
        self.audio_path = None

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((self.root.winfo_width() / 2.25), (self.root.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)

            self.label_image.configure(image=im)
            self.label_image.image = im
            self.result_label.configure(text='')
            self.button_detect_emotion.configure(state=tk.NORMAL)
            self.image_path = file_path
            self.photo_image = im  # Store a reference to the PhotoImage object
        except Exception as e:
            print(e)

    def upload_audio(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
            self.audio_path = file_path
            self.result_label["text"] = f"Audio file: {file_path}"
            self.button_detect_emotion.configure(state=tk.NORMAL)
        except Exception as e:
            print(e)

    def load_model_image(self):
        # Load the model architecture from JSON file
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model_image = model_from_json(loaded_model_json)

        # Load the model weights from H5 file
        self.loaded_model_image.load_weights('model_weights.h5')

    def load_model_audio(self):
        # Load the model architecture from JSON file
        with open('model_a.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model_audio = model_from_json(loaded_model_json)

        # Load the model weights from H5 file
        self.loaded_model_audio.load_weights('Model_67abcde.h5')

    def detect_emotion(self):
        try:
            if self.loaded_model_image is not None and self.loaded_model_audio is not None:
                # Detect emotion for the image
                self.detect_emotion_image()

                # Detect emotion for the audio
                self.detect_emotion_audio()
        except Exception as e:
            print(e)

    def detect_emotion_image(self):
        try:
            if self.image_path:
                image = cv2.imread(self.image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray_image, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_crop = gray_image[y:y + h, x:x + w]
                    roi = cv2.resize(face_crop, (48, 48))
                    pred = self.EMOTIONS_LIST[np.argmax(self.loaded_model_image.predict(roi[np.newaxis, :, :, np.newaxis]))]

                print("Predicted Emotion from Image is " + pred)
                self.result_label["text"] += f"\nImage Emotion: {pred}"
            else:
                print("No image uploaded.")
        except Exception as e:
            print(e)

    def detect_emotion_audio(self):
        try:
            if self.audio_path:
                audio = AudioSegment.from_file(self.audio_path)
                audio_array = np.array(audio.get_array_of_samples())

                target_length = 7740
                if len(audio_array) < target_length:
                    audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
                elif len(audio_array) > target_length:
                    audio_array = audio_array[:target_length]

                audio_array = audio_array.reshape(1, 1, target_length)
                audio_array = audio_array / np.max(np.abs(audio_array))

                prediction = self.loaded_model_audio.predict(audio_array)
                emotion_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']
                emotion_index = np.argmax(prediction)
                detected_emotion = emotion_labels[emotion_index]

                print("Predicted Emotion from Audio is " + detected_emotion)
                self.result_label["text"] += f"\nAudio Emotion: {detected_emotion}"
            else:
                print("No audio uploaded.")
        except Exception as e:
            print(e)

    def reset(self):
        try:
            if self.image_path:
                os.remove(self.image_path)  # Remove the image file
            if hasattr(self, 'photo_image') and self.photo_image:
                self.photo_image.__del__()  # Explicitly delete the PhotoImage object
        except Exception as e:
            print(e)

        self.image_path = None
        self.audio_path = None
        self.label_image.configure(image=None)
        self.result_label.configure(text='')
        self.button_detect_emotion.configure(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiModalEmotionDetectorApp(root)
    app.load_model_image()
    app.load_model_audio()
    root.mainloop()
