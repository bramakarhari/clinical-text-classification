from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages



from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from django.shortcuts import render
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
###################################################


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})



def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'mtsamples.csv'
    df = pd.read_csv(path)
    # Convert DataFrame to HTML table
    df_html = df.to_html(classes='display', index=False)  # Add 'display' class for DataTables
    return render(request, 'users/viewdataset.html', {'data': df_html})
####################################################################
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import os

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Global variables
model = None
vectorizer = None
train_accuracy = None

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in custom_stopwords]
    return ' '.join(tokens)

# Training View
from sklearn.metrics import classification_report

from django.http import HttpResponse
import io
import base64

import base64
import io
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse

def training(request):
    global model, vectorizer, train_accuracy

    from django.conf import settings
    import os
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Load and preprocess data
    data_path = os.path.join(settings.MEDIA_ROOT, 'mtsamples.csv')  # Adjust this path as necessary
    if not os.path.exists(data_path):
        return HttpResponse(f"CSV file not found at {data_path}")

    print("CSV file found at", data_path)
    
    df = pd.read_csv(data_path)
    
    # Check if the CSV file has the necessary columns
    required_columns = ['description', 'transcription', 'sample_name', 'keywords', 'medical_specialty']
    if not all(col in df.columns for col in required_columns):
        return HttpResponse("CSV file does not contain required columns")

    print("CSV loaded successfully")
    
    # Remove duplicates and drop missing values
    df = df.drop_duplicates()
    df = df.dropna(subset=required_columns)
    
    # Preprocess the text fields
    for col in ['description', 'transcription', 'sample_name', 'keywords']:
        df[col] = df[col].apply(preprocess_text)
    
    print("Preprocessing completed")
    
    # Combine all text fields into a single feature
    df['combined_text'] = df['description'] + ' ' + df['transcription'] + ' ' + df['sample_name'] + ' ' + df['keywords']
    
    # Vectorize the combined text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['combined_text']).toarray()
    y = df['medical_specialty']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Undersampling to balance the training set
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print("Data resampling completed")

    # Initialize and train the CatBoost model
    model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, eval_metric='Accuracy', verbose=True)
    model.fit(X_resampled, y_resampled)

    print("Model training completed")

    # Predictions on the training set
    y_pred_train = model.predict(X_resampled)
    
    # Calculate training accuracy
    train_accuracy = accuracy_score(y_resampled, y_pred_train)
    
    # Generate classification report
    classification_rep = classification_report(y_resampled, y_pred_train, output_dict=True)
    
    # Convert classification report to a displayable format
    classification_rep_str = classification_report(y_resampled, y_pred_train)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    
    import joblib  # For saving and loading models

    # Save model and vectorizer after training
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model and vectorizer saved")

    # Predictions on the test set
    y_pred_test = model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Create a BytesIO object to save the plot
    buf = io.BytesIO()
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues,ax=plt.gca())
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # Save the plot to the BytesIO object

    # Adjust label rotation to prevent overlap
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0, ha='right')

    plt.tight_layout()  # Ensure everything fits without overlapping
    plt.savefig(buf, format='png', dpi=300)  # Save the plot to the BytesIO object
    plt.close()  # Close the plot

    # Encode the image to base64
    buf.seek(0)  # Seek to the beginning of the BytesIO object
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')  # Encode as base64 string
    img_data = f"data:image/png;base64,{img_str}"  # Create the data URI

    # Render the training results with the confusion matrix image
    return render(request, 'users/training.html', {
        'train_acc': f"{train_accuracy * 100:.2f}%",
        'classification_report': classification_rep_str,
        'confusion_matrix_img': img_data,  # Pass the image data to the template
    })





    







#####################################################################
# Prediction View
def prediction(request):
    global model, vectorizer
    
    # Load model and vectorizer if they are not loaded in memory
    import os
    import joblib

    def load_model_and_vectorizer():
        global model, vectorizer
        
        if model is None or vectorizer is None:
            if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
                model = joblib.load('model.pkl')
                vectorizer = joblib.load('vectorizer.pkl')
            else:
                return False
        return True
    
    if not load_model_and_vectorizer():
        return render(request, 'users/predictForm.html', {'error': 'Model has not been trained yet. Please train the model first.'})
    
    if request.method == 'POST':
        user_description = request.POST.get('description')
        user_transcription = request.POST.get('transcription')
        user_sample_name = request.POST.get('sample_name')
        user_keywords = request.POST.get('keywords')
        
        # Check if all inputs are provided
        if not (user_description and user_transcription and user_sample_name and user_keywords):
            return render(request, 'users/predictForm.html', {'error': 'Please enter all the fields.'})
        
        # Combine the inputs into one text
        combined_text = f"{user_description} {user_transcription} {user_sample_name} {user_keywords}"

        # Preprocess the combined text
        preprocessed_text = preprocess_text(combined_text)
        
        # Vectorize the combined text
        vectorized_text = vectorizer.transform([preprocessed_text]).toarray()
        
        # Predict the medical specialty
        predicted_specialty = model.predict(vectorized_text)[0]
        
        # Return the prediction result to the template
        return render(request, 'users/predictForm.html', {
            'user_description':user_description,
            'user_transcription':user_transcription,
            'user_sample_name':user_sample_name,
            'user_keywords':user_keywords,
            'output': predicted_specialty
        })
    
    return render(request, 'users/predictForm.html')




       







 