from flask import Flask, render_template, request
import numpy as np
import pickle

diabetes_model = pickle.load(open('models/diabetes.pkl', 'rb'))
cancer_model = pickle.load(open('models/cancer.pkl', 'rb'))
heart_model = pickle.load(open('models/heart.pkl', 'rb'))
liver_model = pickle.load(open('models/liver.pkl', 'rb'))
kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET','POST'])
def cancer():
    return render_template('cancer.html')

@app.route("/heart", methods=['GET','POST'])
def heart():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET','POST'])
def kidney():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET','POST'])
def liver():
    return render_template('liver.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if(len([float(x) for x in request.form.values()])==8):
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])
            
            data = np.array([[preg,glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = diabetes_model.predict(data)
            
            return render_template('predict.html', prediction=my_prediction)
        elif(len([float(x) for x in request.form.values()])==10):
            Age = int(request.form['Age'])
            Total_Bilirubin = float(request.form['Total_Bilirubin'])
            Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
            Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
            Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
            Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
            Total_Protiens = float(request.form['Total_Protiens'])
            Albumin = float(request.form['Albumin'])
            Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
            Gender_Male = int(request.form['Gender_Male'])

            data = np.array([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_Male]])
            my_prediction = liver_model.predict(data)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==13):
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = heart_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==18):
            age = float(int(request.form['age']))
            bp = float(request.form['bp'])
            al = float(request.form['al'])
            su = float(request.form['su'])
            rbc = int(request.form['rbc'])
            pc = int(request.form['pc'])
            pcc = int(request.form['pcc'])
            ba = int(request.form['ba'])
            bgr = float(request.form['bgr'])
            bu = float(request.form['bu'])
            sc = float(request.form['sc'])
            pot = float(request.form['pot'])
            wc = int(request.form['wc'])
            htn = int(request.form['htn'])
            dm = int(request.form['dm'])
            cad = int(request.form['cad'])
            pe = int(request.form['pe'])
            ane = int(request.form['ane'])

            data = [age,bp,al,su,rbc,pc,pcc,ba,bgr,bu,sc,pot,wc,htn,dm,cad,pe,ane]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = kidney_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==26):
            radius_mean = float(request.form['radius_mean'])
            texture_mean = float(request.form['texture_mean'])
            perimeter_mean = float(request.form['perimeter_mean'])
            area_mean = float(request.form['area_mean'])
            smoothness_mean = float(request.form['smoothness_mean'])
            compactness_mean = float(request.form['compactness_mean'])
            concavity_mean = float(request.form['concavity_mean'])
            concave_points_mean = float(request.form['concave points_mean'])
            symmetry_mean = float(request.form['symmetry_mean'])
            radius_se = float(request.form['radius_se'])
            perimeter_se = float(request.form['perimeter_se'])
            area_se = float(request.form['area_se'])
            compactness_se = float(request.form['compactness_se'])
            concavity_se = float(request.form['concavity_se'])
            concave_points_se = float(request.form['concave points_se'])
            fractal_dimension_se = float(request.form['fractal_dimension_se'])
            radius_worst = float(request.form['radius_worst'])
            texture_worst = float(request.form['texture_worst'])
            perimeter_worst = float(request.form['perimeter_worst'])
            area_worst = float(request.form['area_worst'])
            smoothness_worst = float(request.form['smoothness_worst'])
            compactness_worst = float(request.form['compactness_worst'])
            concavity_worst = float(request.form['concavity_worst'])
            concave_points_worst = float(request.form['concave points_worst'])
            symmetry_worst = float(request.form['symmetry_worst'])
            fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

            data = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,radius_se,perimeter_se,area_se,compactness_se,concavity_se,concave_points_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
            data1 = np.array(data).reshape(1,-1)
            my_prediction  = cancer_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)












if __name__ == "__main__":
    app.run(debug=True)