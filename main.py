
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


csv_file_path = "home_PP/Housing.csv"  
data = pd.read_csv(csv_file_path)

target_variable_name = 'price'
X = data.drop(target_variable_name, axis=1)
y = data[target_variable_name]

numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train_preprocessed.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_preprocessed, y_train, epochs=100, batch_size=32, verbose=0)
predictions = model.predict(X_test_preprocessed).flatten()
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
def predict_house_price(new_input):
  
    new_input_preprocessed = preprocessor.transform(new_input)
    
   
    predicted_price = model.predict(new_input_preprocessed).flatten()[0]
    
    return predicted_price

new_input = pd.DataFrame({
    'area': [7420],
    'bedrooms': [4],
    'bathrooms': [2],
    'stories': [3],
    'mainroad': ['yes'],
    'guestroom': ['no'],
    'basement': ['no'],
    'hotwaterheating': ['no'],
    'airconditioning': ['yes'],
    'parking': [2],
    'prefarea': ['yes'],
    'furnishingstatus': ['furnished']
})

predicted_price = predict_house_price(new_input)
print(f'Predicted House Price: {predicted_price}')
