from flask import Flask,render_template,json,jsonify,request
import pickle
import numpy as np 
import requests


## initilazing app 
app = Flask(__name__,template_folder='templates')
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        CRIM=float(request.form['CRIM'])
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
        ##loading pickle file 
        filename = 'ridge_new_model.pickle'
        loaded_model = pickle.load(open(filename,'rb'))
        data = np.array(([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B]))
        data.reshape(1,2)
        
        my_prediction = loaded_model.predict(data)
        return render_template('index.html',prediction = my_prediction)
if __name__ =='__main__':
    app.run(debug=True)        

