import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import gunicorn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    double_features = [float(x) for x in request.form.values()]
    print(double_features)
    final_features = [np.array(double_features)]
    prediction = model.predict(final_features)
    print(prediction)
    output = round(prediction[0], 6)

    return render_template('index.html', prediction_text='Predicted Medical Insurance Cost is : {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)