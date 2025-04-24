from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import ValidationError
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\sjave\Music\CALORIE_ESTIMATION\FRONTEND\Food.h5')

# Define your class names
class_names = np.array([
    'Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog',
    'Sandwich', 'Taco', 'Taquito', 'apple_pie', 'burger',
    'butter_naan', 'chai', 'chapati', 'cheesecake', 'chicken_curry',
    'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice',
    'ice_cream', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer',
    'kulfi', 'masala_dosa', 'momos', 'omelette', 'paani_puri',
    'pakode', 'pav_bhaji', 'pizza', 'samosa', 'sushi'
])

# Load nutritional information CSV
nutrition_df = pd.read_csv(r'C:\Users\sjave\Music\CALORIE_ESTIMATION\FRONTEND\food_nutrition_100g.csv')

def validate_image(img):
    """Ensure the uploaded file is an image."""
    try:
        Image.open(img).verify()
    except Exception:
        raise ValidationError("Uploaded file is not a valid image.")

def preprocess_image(image):
    """Preprocess the image to match the model's input."""
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def home(request):
    return render(request, 'index.html')

def input(request):
    file_name = 'account.txt'
    name = request.POST.get('name')
    password = request.POST.get('password')
    with open(file_name, 'r') as file:
        account_list = [line.split() for line in file]
    for account in account_list:
        if account[0] == name and account[1] == password:
            return render(request, 'input.html')
    return HttpResponse('Wrong Password or Name', content_type='text/plain')

def output(request):
    img = request.FILES['file']

    try:
        validate_image(img)
    except ValidationError as e:
        return HttpResponse(str(e), content_type="text/plain")

    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    food_info = nutrition_df[nutrition_df['Food Item'].str.lower().str.strip() == predicted_class.lower().strip()]

    if not food_info.empty:
        nutrition_data = food_info.iloc[0]
        nutrition = {
            'Calories': nutrition_data.get('Calories (kcal)', 'N/A'),
            'Carbohydrates': nutrition_data.get('Carbohydrates (g)', 'N/A'),
            'Proteins': nutrition_data.get('Proteins (g)', 'N/A'),
            'Fats': nutrition_data.get('Fats (g)', 'N/A'),
            'Fiber': nutrition_data.get('Fiber (g)', 'N/A'),
            'Sugars': nutrition_data.get('Sugars (g)', 'N/A'),
        }
    else:
        nutrition = {
            'Calories': 'N/A',
            'Carbohydrates': 'N/A',
            'Proteins': 'N/A',
            'Fats': 'N/A',
            'Fiber': 'N/A',
            'Sugars': 'N/A',
        }

    context = {
        'out': predicted_class,
        'nutrition': nutrition
    }

    return render(request, 'output.html', context)
