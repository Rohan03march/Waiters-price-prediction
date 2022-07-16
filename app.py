import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model_lr.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='Tips :{}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)