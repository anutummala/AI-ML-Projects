import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("students.csv")

df['Name_Length'] = df['Name'].apply(len)

# Convert City (categorical) using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
city_encoded = encoder.fit_transform(df[['City']])

# Combine numerical features
numerical_features = df[['Age', 'Name_Length']].values

X = np.hstack([numerical_features,city_encoded])

def categorize_age(age) :
    if (age < 18) :
        return 0 #Minor
    elif (age <= 25) :
        return 1 #Young
    else :
        return 2 #Adult
    
y = df['Age'].apply(categorize_age).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_encoder = OneHotEncoder(sparse_output=False)
y_encoded = y_encoder.fit_transform(y.reshape(-1,1))

#Setup the model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_scaled.shape[1],)))
model.add(Dense(16,activation='relu'))
model.add(Dense(y_encoded.shape[1],activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Train the model
model.fit(X_scaled, y_encoded, epochs=100, verbose=0)
loss, accuracy = model.evaluate(X_scaled, y_encoded, verbose=0)
print("Neural Network Accuracy:", accuracy)

predictions = model.predict(X_scaled)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes:", predicted_classes)