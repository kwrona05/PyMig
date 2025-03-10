import cv2
import numpy as np
import mediapipe as mp
import csv
import os
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from gtts import gTTS
import pygame
import time

# Inicjalizacja modeli MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

# Plik CSV do zapisywania danych
data_file = "gestures_psl.csv"
if not os.path.exists(data_file):
    with open(data_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"x{i}" for i in range(63)] + ["label"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

print("Nagrywanie gestów... Naciśnij 'W' aby zakończyć zbieranie danych.")
gesture_dict = {}

while True:
    gesture_name = input("Podaj nazwę gestu, który chcesz nagrać (lub wpisz 'koniec' aby zakończyć nagrywanie): ").strip()
    if gesture_name == "koniec":
        break
    if gesture_name in gesture_dict:
        print(f"Ten gest juz mam zapisany '{gesture_name}'. Nagraj inny gest.")
        continue
    gesture_dict[gesture_name] = []
    
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    features = extract_hand_landmarks(hand_landmarks)
                    gesture_dict[gesture_name].append(features)
                    
                    with open(data_file, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([*features, gesture_name])
            
            cv2.imshow("Nagrywanie gestów", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("w"):
                break

cap.release()
cv2.destroyAllWindows()

decision = input("Czy chcesz przejść do testowania (t/n): ").strip().lower()
if decision != "t":
    print("Zakończono trenowanie. Uruchamianie programu")
    exit()

try:
    data = pd.read_csv(data_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    if len(set(y)) < 2:
        print("Musisz nagrać conajmniej 2 gesty")
        exit()
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    print("Dokładność modelu: ", model.score(X_test, y_test))
    
    joblib.dump(model, "psl_model.pkl")
    joblib.dump(encoder, "psl_encoder.pkl")
    
except Exception as e:
    print(f"Błąd podczas uruchamiania programu: {e}")
    exit()

test_cap = cv2.VideoCapture(0)
model = joblib.load("psl_model.pkl")
encoder = joblib.load("psl_encoder.pkl")

pygame.mixer.init()

print("Program uruchomiony. Naciśnij 'q' aby zakończyć działanie programu")

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while test_cap.isOpened():
        ret, frame = test_cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                features = extract_hand_landmarks(hand_landmarks)
                prediction = model.predict([features])
                label = encoder.inverse_transform(prediction)[0]
                
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                #Text to speech conversion
                tts = gTTS(text=label, lang='pl')
                tts.save("gesture.mp3")
                pygame.mixer.music.load("gesture.mp3")
                pygame.mixer.music.play()
        else:
            cv2.putText(frame, "Nie wykryto dłoni", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Tłumacz języka migowego", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

test_cap.release()
cv2.destroyAllWindows()