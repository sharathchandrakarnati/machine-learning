from flask import Flask,render_template,json,jsonify,request
import pickle
import numpy as np 
import requests
from sklearn.preprocessing import StandardScaler 


## initilazing app 
app = Flask(__name__,template_folder='templates')
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        CRIM=float(request.form['CRIM'])
        print(CRIM)
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        CHAS =float(request.form['CHAS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        AGE = float(request.form['AGE'])
        DIS = float(request.form['DIS'])
        RAD = float(request.form['RAD'])
        TAX = float(request.form['TAX'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])
        ##loading pickle file 
        filename = 'ridge_new_model.pickle'
        loaded_model = pickle.load(open(filename,'rb'))
        data = np.array([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]).reshape(1,-1)
        sc = StandardScaler()
        x = sc.fit_transform(data)
        my_prediction = loaded_model.predict(x)
        print(my_prediction)
        return render_template('index.html',Prediction_text = my_prediction)
if __name__ =='__main__':
    app.run(debug=True)        



