# LIBRERIAS FUNDAMENTALES 
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# LIBRERIA PARA CONVERTIR EL TEXTO A VOZ
import pyttsx3
engine = pyttsx3.init()


# DEFINICION DE CLASES (ACCIONES) CON EXCEPCION DE "J"
actions = np.array([
    'A','B','C','D','E','F','G','H','I', 
    'K','L','M','N','Ñ','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
])  # 26 letras


# MODELO (SOLO INFERENCIA)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ACTIVAR LA RED NEURONAL PARA CARGAR EL MODELO KERAS
model = Sequential([
    Input(shape=(30, 126)),  
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model.load_weights("pesos.keras")  # Cargar pesos entrenados

#FUNCIONES IMPORTANTES PARA ACTIVAR MEDIAPIPE HOLISTIC
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    #Dibujar solo la mano que este mas cerca de la camara 
    if results.left_hand_landmarks and results.right_hand_landmarks:
        avg_z_lh = np.mean([lm.z for lm in results.left_hand_landmarks.landmark])
        avg_z_rh = np.mean([lm.z for lm in results.right_hand_landmarks.landmark])

        if avg_z_lh < avg_z_rh:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        else:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
    elif results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    elif results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else None
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else None

    # Comparar el promedio del valor de Z (Proufundidad) de ambas manos
    if lh is not None and rh is not None:
        if np.mean(lh[:, 2]) < np.mean(rh[:, 2]):
            lh = lh.flatten()
            rh = np.zeros(63)
        else:
            rh = rh.flatten()
            lh = np.zeros(63)
    elif lh is not None:
        lh = lh.flatten()
        rh = np.zeros(63)
    elif rh is not None:
        rh = rh.flatten()
        lh = np.zeros(63)
    else:
        lh = np.zeros(63)
        rh = np.zeros(63)

    return np.concatenate([lh, rh])


# VISUALIZACION
def generate_colors(n):
    np.random.seed(42)
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)]

colors = generate_colors(len(actions))

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    bar_height = 12
    spacing = 3

    for num, prob in enumerate(res):
        bar_width = int(prob * 150)
        y_start = 50 + num * (bar_height + spacing)
        y_end = y_start + bar_height

        cv2.rectangle(output_frame, (10, y_start), (10 + bar_width, y_end), colors[num], -1)
        cv2.putText(
            output_frame, actions[num], (15, y_start + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA
        )
    return output_frame


# INFERENCIA EN TIEMPO REAL

sequence = []
sentence = []
predictions = []
threshold = 0.9 

last_action_time = time.time()  
last_action = None  
timer_value = 5

cap = cv2.VideoCapture(0)

# Establecer mediapipe 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Leer pantalla
        ret, frame = cap.read()

        # Hacer las detecciones
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Dibujar los puntos de las manos
        draw_styled_landmarks(image, results)
        
        # Logica de prediccion 
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
        


        #3. Logica de visualizacion 
            if np.unique(predictions[-10:])[0] == np.argmax(res): 
                if res[np.argmax(res)] > threshold: 

                    if len(sentence) > 0: 
                        detected_action = actions[np.argmax(res)]
                        current_time = time.time()

                        if detected_action != sentence[-1]:
                            if last_action == detected_action:
                                if current_time - last_action_time >= timer_value:  # temporizador
                                    sentence.append(detected_action)
                                    last_action_time = current_time  
                            else:
                                last_action = detected_action
                                last_action_time = current_time  # Reiniciar temporizador al ser detectado
                    else:
                        sentence.append(actions[np.argmax(res)])
                        last_action_time = time.time()  # Reiniciar temporizador

            if len(sentence) > 15: 
                sentence = sentence[-15:]

            # Barra de Probabilidades
            image = prob_viz(res, actions, image, colors)

        # Mostrar letras detectadas
        cv2.rectangle(image, (0,0), (640, 40), (159, 0, 0), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        

        # Mostrar la pantalla
        cv2.imshow('Deteccion-LSM', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):  # Salir presionando la letra 'q'
            break
        elif key == ord('d'):  
            sentence.clear()
            print("Sentence cleared!")                  # Reiniciar la oracion 
        elif key == ord('p') or key == ord('P'):        # Presionar 'P' para detectar la accion 
            actions_text = ' '.join(sentence)
            print("Detected actions:", actions_text)    # Imprimir oracion 
            x1 = actions_text.replace(' ', '')          # Quitar espacios 
            engine.say(x1)                              # Pasar oracion al TTS
            engine.runAndWait()                     

    cap.release()
    cv2.destroyAllWindows()
