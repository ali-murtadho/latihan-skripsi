from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix as sk_confusion_matrix
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings

import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import urllib, base64

class Preprocessing_read_csv:
    def read_data(self):
        file_path = os.path.join(settings.BASE_DIR, 'readcsv\data\padi-clean.csv')
        df = pd.read_csv(file_path)
        return df

def show_data(request):
    preprocessing = Preprocessing_read_csv()
    df = preprocessing.read_data()

    data_html = df.to_html(classes="table table-striped", index=False)

    return render(request, 'data_table.html', {'data_table': data_html})


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

# NORMALISASI SMOTE BEST PARAM
# Accuracy: 0.9984496124031008
# F1 Score: 0.9985090066537489
# Precision: 0.9985585887663974
# Recall: 0.9984662576687117
# Overfitting Score: 0.0015503875968991832

# ASLI
# Accuracy: 0.9987546699875467
# F1 Score: 0.9987481100107445
# Precision: 0.9981617647058824
# Recall: 0.9993421052631579
# Overfitting Score: 0.001245330012453305

def training(request):
    if 'train' in request.POST:
        clf = DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        pickle.dump(best_model, open('model.pkl', 'wb'))
        return render(request, 'data_table.html')
    if 'train_r' in request.POST:
        clf_r = DecisionTreeClassifier().fit(X_train_r, y_train_r)
        pickle.dump(clf_r, open('model_r.pkl', 'wb'))
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

def classification(request):
    return render(request, 'input.html')

def prediction(request):
    # Mapping dictionaries
    varietas_mapping = {
        "0.0": "Beras Hitam",
        "0.6": "Beras Merah",
        "0.4": "IR 64",
        "0.8": "Mi Kongga",
        "0.2": "Panda Wangi",
        "1.0": "Pandan Wangi"
    }

    warna_mapping = {
        "0.67": "Merah",
        "0.0": "Coklat",
        "0.33": "Hitam",
        "1.0": "Putih"
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

    grade_mapping = {
        0: "Kelas A",
        1: "Kelas B", 
        2: "Kelas C",
        3: "Kelas D"
    }

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
    return render(request, 'input.html', context)

def to_excel(request):
    return render(request, 'excel.html')

# Prediction mapping
prediction_mapping = {
    0: "Kelas A",
    1: "Kelas B",
    2: "Kelas C",
    3: "Kelas D"
}

def importExcel(request):
    if request.method == 'POST' and request.FILES['excel_file']:
        excel_file = request.FILES['excel_file']
        data = pd.read_excel(excel_file)
        
        # Assuming the columns in the Excel file match the expected input order
        input_data = data.values
        
        # Load the trained model
        model = pickle.load(open('model.pkl', 'rb'))
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Map the predictions to class names
        prediction_classes = [prediction_mapping.get(int(pred), "Unknown") for pred in predictions]
        
        # Prepare the context for rendering
        context = {
            'predictions': zip(data.values, prediction_classes)
        }
        
        return render(request, 'results.html', context)
