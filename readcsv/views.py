from .models import ClassificationResult
from django.http import HttpResponse
from django.shortcuts import render, render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix as sk_confusion_matrix

import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64

    # Mapping dictionaries
varietas_mapping = {
    "0.0": "Beras Hitam",
    "0.6": "Ciheran",
    "0.4": "IR 64",
    "0.8": "Mi Kongga",
    "0.2": "Beras Merah",
    "1.0": "Pandan Wangi"
}

warna_mapping = {
    "0.67": "Merah",
    "0.0": "Coklat",
    "0.33": "Hitam",
    "1.0": "putih",
}

rasa_mapping = {
    "0.0": "Pulen",
    "1.0": "Sangat Pulen"
}

musim_mapping = {
    "0.0": "Hujan",
    "1.0": "Kemarau"
}

penyakit_mapping = {
    "0.0": "Burung",
    "0.25": "Penggerek Batang",
    "0.5": "Tikus",
    "1.0": "Wereng Hijau",
    "0.75": "Wereng Coklat"
}

teknik_mapping = {
    "0.0": "Jajar Legowo",
    "1.0": "SRI"
}

# Prediction mapping
prediction_mapping = {
    0: "Kelas A",
    1: "Kelas B",
    2: "Kelas C",
    3: "Kelas D"
}
# Reverse mappings
reverse_varietas_mapping = {v: k for k, v in varietas_mapping.items()}
reverse_warna_mapping = {v: k for k, v in warna_mapping.items()}
reverse_rasa_mapping = {v: k for k, v in rasa_mapping.items()}
reverse_musim_mapping = {v: k for k, v in musim_mapping.items()}
reverse_penyakit_mapping = {v: k for k, v in penyakit_mapping.items()}
reverse_teknik_mapping = {v: k for k, v in teknik_mapping.items()}

class Preprocessing_read_csv:
    def read_data(self):
        file_path = os.path.join(settings.BASE_DIR, 'readcsv\data\padi-clean.csv')
        df = pd.read_csv(file_path)
        return df

def show_data(request):
    preprocessing = Preprocessing_read_csv()
    df = preprocessing.read_data()

    data_html = df.to_html(classes="table table-striped", index=False)
    total_rows = len(df)

    return render(request, 'data_table.html', {'data_table': data_html, 'total_rows': total_rows})


x_normalization_file = os.path.join(settings.BASE_DIR,r'readcsv\data\x_normalisasi.csv')
y_smote_file = os.path.join(settings.BASE_DIR,r'readcsv\data\y_smote.csv')
X_norm = pd.read_csv(x_normalization_file)
y_smote = pd.read_csv(y_smote_file)

data_x = os.path.join(settings.BASE_DIR,r'readcsv\data\x.csv')
data_y = os.path.join(settings.BASE_DIR,r'readcsv\data\y.csv')
X_real = pd.read_csv(data_x)
Y_real = pd.read_csv(data_y)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_smote, test_size=0.3, random_state=50)
data_X_train_r = os.path.join(settings.BASE_DIR,r'readcsv\data\X_train_r.csv')
data_X_test_r = os.path.join(settings.BASE_DIR,r'readcsv\data\X_test_r.csv')
data_Y_train_r = os.path.join(settings.BASE_DIR,r'readcsv\data\y_train_r.csv')
data_Y_test_r = os.path.join(settings.BASE_DIR,r'readcsv\data\y_test_r.csv')
X_train_r = pd.read_csv(data_X_train_r)
X_test_r = pd.read_csv(data_X_test_r)
y_train_r = pd.read_csv(data_Y_train_r)
y_test_r = pd.read_csv(data_Y_test_r)

# Parameter grid for DecisionTreeClassifier
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def training(request):
    if 'train' in request.POST:
        clf = DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        pickle.dump(best_model, open('model.pkl', 'wb'))
        return render(request, 'data_table.html')
    if 'train_r' in request.POST:
        clf_r = DecisionTreeClassifier()
        grid_search_r = GridSearchCV(estimator=clf_r, param_grid=param_grid, cv=10)
        grid_search_r.fit(X_train_r, y_train_r)
        best_model_r = grid_search_r.best_estimator_
        pickle.dump(best_model_r, open('model_r.pkl', 'wb'))
        return render(request, 'data_table.html')

def testing(request):
    if 'test' in request.POST:
        model = pickle.load(open('model.pkl', 'rb'))
        y_pred = model.predict(X_test)

        print(accuracy_score(y_test, y_pred))
        print(y_pred)
        y_pred = pd.DataFrame(y_pred,columns=['output'])
        y_pred.to_csv('testing.csv')
        
        filename = 'testing.csv'
        response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')               
        response['Content-Length'] = os.path.getsize(filename)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'testing.csv'
        return response
    
    if 'test_r' in request.POST:
        model = pickle.load(open('model_r.pkl', 'rb'))
        y_pred_r = model.predict(X_test_r)

        print(accuracy_score(y_test_r, y_pred_r))
        print(y_pred_r)
        y_pred_r = pd.DataFrame(y_pred_r,columns=['output'])
        y_pred_r.to_csv('testing_r.csv')
        
        filename = 'testing_r.csv'
        response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')
        response['Content-Length'] = os.path.getsize(filename)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'testing.csv'
        return response
    
def evaluation(request):
    if 'result' in request.POST:
        model = pickle.load(open('model.pkl', 'rb'))
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        acc = accuracy_score(y_test, y_pred) #test acc
        rec = recall_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Calculating metrics for the train set
        train_acc = accuracy_score(y_train, y_pred_train)

        overfitting_score = train_acc - acc

        cm = sk_confusion_matrix(y_test, y_pred)
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save the plot to a PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        buf.close()
        context = {
            'acc': acc,
            'rec': rec,
            'prec': prec,
            'f1': f1,
            'train_acc': train_acc,
            'overfitting_score': overfitting_score,
            'confusion_matrix': uri
        }
        return render(request, 'data_table.html', context)
    if 'result_r' in request.POST:
        model = pickle.load(open('model_r.pkl', 'rb'))
        y_pred_r = model.predict(X_test_r)
        y_pred_train_r = model.predict(X_train_r)

        acc = accuracy_score(y_test_r, y_pred_r) #test acc
        rec = recall_score(y_test_r, y_pred_r, average='macro')
        prec = precision_score(y_test_r, y_pred_r, average='macro')
        f1 = f1_score(y_test_r, y_pred_r, average='macro')

        # Calculating metrics for the train set
        train_acc = accuracy_score(y_train_r, y_pred_train_r)

        overfitting_score = train_acc - acc

        cm = sk_confusion_matrix(y_test_r, y_pred_r)
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save the plot to a PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        buf.close()

        context = {
            'acc': acc,
            'rec': rec,
            'prec': prec,
            'f1': f1,
            'train_acc': train_acc,
            'overfitting_score': overfitting_score,
            'confusion_matrix': uri
        }
        return render(request, 'data_table.html', context)

# @login_required
def classification(request):
    return render(request, 'input.html')

def prediction(request):
    if 'prediction' in request.POST:
        # Extract the values from the form
        varietas = request.POST['Varietas']
        warna = request.POST['Warna']
        rasa = request.POST['rasa']
        musim = request.POST['Musim']
        penyakit = request.POST['Penyakit']
        teknik = request.POST['teknik']
        ph = request.POST['PH']
        boron = request.POST['boron']
        fosfor = request.POST['fosfor']

        # Combine the values into a single array
        data = np.array([[float(varietas), float(warna), float(rasa), float(musim), float(penyakit), float(teknik), float(ph), float(boron), float(fosfor)]])

        model = pickle.load(open('model_r.pkl', 'rb'))

        prediction = model.predict(data)

        # Map the prediction to class name
        prediction_class = prediction_mapping.get(prediction[0], "Unknown")

        # Save the result to the database
        result = ClassificationResult(
            varietas=varietas_mapping[varietas],
            warna=warna_mapping[warna],
            rasa=rasa_mapping[rasa],
            musim=musim_mapping[musim],
            penyakit=penyakit_mapping[penyakit],
            teknik=teknik_mapping[teknik],
            ph=ph,
            boron=boron,
            fosfor=fosfor,
            prediction=prediction_class
        )
        result.save()


        # Prepare the context for rendering
        context = {
            'varietas': varietas_mapping[varietas],
            'warna': warna_mapping[warna],
            'rasa': rasa_mapping[rasa],
            'musim': musim_mapping[musim],
            'penyakit': penyakit_mapping[penyakit],
            'teknik': teknik_mapping[teknik],
            'ph': ph,
            'boron': boron,
            'fosfor': fosfor,
            'prediction': prediction_class  # Assuming prediction is a single value
        }

    # Fetch all classification results from the database
    results = ClassificationResult.objects.all()
    context['results'] = results
    return render(request, 'input.html', context)

def to_excel(request):
    return render(request, 'excel.html')

# Mappings
varietas_csv_mapping = {
    "Beras Hitam": 0.0,
    "Pandan Wangi": 1.0,
    "IR 64": 0.4,
    "Beras Merah": 0.6,
    "Mi Kongga": 0.8,
    "Panda Wangi": 0.2
}

warna_csv_mapping = {
    "Coklat": 0.0,
    "Hitam": 0.33,
    "Merah": 0.67,
    "putih": 1.0
}

rasa_csv_mapping = {
    "Pulen": 0.0,
    "Sangat Pulen": 1.0
}

musim_csv_mapping = {
    "Hujan": 0.0,
    "Kemarau": 1.0
}

penyakit_csv_mapping = {
    "Burung": 0.0,
    "Penggerek Batang": 0.25,
    "Tikus": 0.5,
    "Wereng Coklat": 0.75,
    "Wereng Hijau": 1.0
}

teknik_csv_mapping = {
    "Jajar Legowo": 0.0,
    "SRI": 1.0
}

def importCsv(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        data = pd.read_csv(file_path)

        # Mapping string values to numeric values
        data['Varietas'] = data['Varietas'].map(varietas_csv_mapping)
        data['Warna'] = data['Warna'].map(warna_csv_mapping)
        data['rasa'] = data['rasa'].map(rasa_csv_mapping)
        data['Musim'] = data['Musim'].map(musim_csv_mapping)
        data['Penyakit'] = data['Penyakit'].map(penyakit_csv_mapping)
        data['teknik'] = data['teknik'].map(teknik_csv_mapping)

        model_path = os.path.join(settings.BASE_DIR, 'model_r.pkl')
        model = pickle.load(open(model_path, 'rb'))

        # Drop the 'Varietas' column if it's used as the target variable, otherwise include it
        data_transformed = data.drop(columns=['Varietas'])

        # Predict using the model
        predictions = model.predict(data_transformed)

        # Map the predictions to classes
        prediction_classes = ["Kelas A" if pred == 1 else "Kelas B" for pred in predictions]

        # Combine original data and predictions for display
        results = list(zip(data.values, prediction_classes))

        context = {
            "predictions": results,
            "varietas_mapping": varietas_csv_mapping,
            "warna_mapping": warna_csv_mapping,
            "rasa_mapping": rasa_csv_mapping,
            "musim_mapping": musim_csv_mapping,
            "penyakit_mapping": penyakit_csv_mapping,
            "teknik_mapping": teknik_csv_mapping,
        }

        return render(request, "excel.html", context)

    return render(request, "excel.html")

def importExcel(request):
    if request.method == 'POST' and request.FILES['excel_file']:
        excel_file = request.FILES['excel_file']
        df = pd.read_excel(excel_file)

        # Ambil baris kedua untuk prediksi
        prediction_data = df.iloc[1]

        # Hapus kolom Varietas hingga PH
        # prediction_data.drop(["Varietas", "fosfor", "boron", "Warna", "rasa", "teknik", "Musim", "Penyakit", "PH"], inplace=True)

        # Proses prediksi klasifikasi
        prediction_result = predict_classification(prediction_data)

        context = {
            'prediction_data': prediction_data,
            'prediction_result': prediction_result,
        }
        return render(request, 'excel.html', context)

    return render(request, 'excel.html')

def predict_classification(prediction_data):
    # Di sini Anda dapat menambahkan logika untuk memproses dan memprediksi klasifikasi
    # Misalnya, Anda dapat menggunakan model klasifikasi yang telah dilatih sebelumnya
    # atau algoritma klasifikasi lainnya sesuai kebutuhan Anda.
    # Contoh sederhana:

    # Misalkan prediksi klasifikasi didasarkan pada nilai dari salah satu kolom
    # dalam data prediksi.
    value_to_predict = prediction_data['NamaKolom'].values[0]

    # Lakukan prediksi klasifikasi
    if value_to_predict == 'Nilai tertentu':
        prediction_result = 'Kelas A'
    else:
        prediction_result = 'Kelas B'

    return prediction_result

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful." )
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"You are now logged in as {username}.")
                return redirect('classification')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.") 
    return redirect('login')
