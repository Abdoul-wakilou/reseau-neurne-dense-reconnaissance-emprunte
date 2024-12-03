# les biblio necessaires
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt


# Charger les données : 

def load_data_from_folder(folder_path):
    images = []
    labels = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".BMP"):  # Vérifie que c'est bien une image
            # Charger l'image en niveaux de gris
            img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # Redimensionner pour uniformiser (ex : 128x128)
            
            # Ajouter l'image à la liste
            images.append(img)
            
            # Extraire les informations pour l'étiquette
            parts = file_name.split("__")
            subject_id = int(parts[0])  # ID du sujet
            gender = 0 if parts[1][0] == "M" else 1  # 0 = Homme, 1 = Femme
            hand = 0 if "Left" in parts[1] else 1  # 0 = Gauche, 1 = Droite
            
            # Normaliser le nom du doigt en minuscules
            finger_name = parts[1].split("_")[2].lower()  # Convertir en minuscules
            try:
                finger_index = ["thumb", "index", "middle", "ring", "little"].index(finger_name)
            except ValueError:
                print(f"Nom de doigt inconnu dans le fichier : {file_name}. Ignoré.")
                continue  # Ignore ce fichier si le nom du doigt est invalide
            
            # Créer une étiquette combinée (par exemple, un tableau ou une liste)
            label = [subject_id, gender, hand, finger_index]
            labels.append(label)
    
    # Convertir en tableaux NumPy
    images = np.array(images, dtype=np.float32) / 255.0  # Normaliser entre 0 et 1
    labels = np.array(labels)
    
    return images, labels



# Charger les données réelles
x_real, y_real = load_data_from_folder("Real")

# Charger les données altérées
x_easy, y_easy = load_data_from_folder("Altered/Altered-Easy")
x_medium, y_medium = load_data_from_folder("Altered/Altered-Medium")
x_hard, y_hard = load_data_from_folder("Altered/Altered-Hard")

# Combiner les données altérées
x_altered = np.concatenate([x_easy, x_medium, x_hard], axis=0)
y_altered = np.concatenate([y_easy, y_medium, y_hard], axis=0)



# Prétraitement des données
# Combiner toutes les données (réelles + altérées)
x_all = np.concatenate([x_real, x_altered], axis=0)
y_all = np.concatenate([y_real, y_altered], axis=0)

# Diviser en données d'entraînement (80%) et de test (20%)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

print(f"Train set: {x_train.shape}, {y_train.shape}")
print(f"Test set: {x_test.shape}, {y_test.shape}")



"""Préparation pour le modèle dense"""

# Aplatir les images (de 128x128 à 16384 pour chaque image)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"Train set (flattened): {x_train_flat.shape}")
print(f"Test set (flattened): {x_test_flat.shape}")


"""construire et entraîner un modèle dense"""

# Définir le modèle dense
model = Sequential([
    Dense(256, activation='relu', input_shape=(x_train_flat.shape[1],)),  # Couche dense avec 256 neurones
    Dense(128, activation='relu'),  # Couche dense avec 128 neurones
    Dense(64, activation='relu'),   # Couche dense avec 64 neurones
    Dense(4, activation='softmax')  # Couche de sortie pour 4 classes (sujets, genre, main, doigt)
])

# Compiler le modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimiseur avec un taux d'apprentissage de 0.001
    loss='categorical_crossentropy',  # Fonction de perte pour classification multi-classe
    metrics=['accuracy']  # Mesurer la précision pendant l'entraînement
)

# Entraîner le modèle
history = model.fit(
    x_train_flat, y_train,
    epochs=20,  # Nombre d'époques
    batch_size=32,  # Taille des lots
    validation_split=0.2,  # 20 % des données d'entraînement utilisées pour la validation
    verbose=1  # Afficher les logs pendant l'entraînement
)

# Évaluer le modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy:.4f}")


"""Sauvegarder le modèle"""
# Sauvegarder le modèle après l'entraînement
model.save('model_fingerprint_recognition.h5')

# Charger le modèle déjà entraîné
model = load_model('model_fingerprint_recognition.h5')



"""tester et afficher les détails"""
# Charger une image du dossier Real
test_image_path = "Real/1__M_Left_index_finger.BMP"
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image_resized = cv2.resize(test_image, (128, 128))  # Redimensionner pour correspondre au modèle
test_image_normalized = test_image_resized.astype(np.float32) / 255.0  # Normaliser entre 0 et 1

# Préparer l'image pour la prédiction (aplatir l'image)
test_image_flat = test_image_normalized.flatten().reshape(1, -1)

# Faire la prédiction
prediction = model.predict(test_image_flat)
predicted_label = np.argmax(prediction, axis=1)
confidence = prediction[0][predicted_label[0]]

# Dictionnaire pour les informations
fingers = ["Pouce(Thumb)", "Index(Index)", "Majeur(Middle)", "Annulaire(Ring)", "Auriculaire(Little)"]
genders = ["Homme", "Femme"]
hands = ["Gauche", "Droite"]

# Extraire les informations du label
subject_id = y_train[predicted_label[0]][0]
gender = genders[y_train[predicted_label[0]][1]]
hand = hands[y_train[predicted_label[0]][2]]
finger = fingers[y_train[predicted_label[0]][3]]

# Reconnaissance : Oui ou Non en fonction de la confiance
recognition = "Oui" if confidence >= 0.8 else "Non"  # Si la confiance est >= 80%, on considère que la reconnaissance est correcte

# Affichage des détails
plt.imshow(test_image, cmap='gray')
plt.title(f"Reconnaissance : {recognition}\n"
          f"Subject ID : {subject_id}\n"
          f"Gender : {gender}\n"
          f"Hand : {hand}\n"
          f"Finger : {finger}\n"
          f"Confiance : {confidence:.2f}")
plt.axis('off')
plt.show()

