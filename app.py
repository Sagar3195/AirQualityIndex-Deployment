from flask import *
import pandas as pd
import pickle

##load the model from disk
model = pickle.load(open('random_forest_regression.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv("real_2016.csv")
    my_predict = model.predict(df.iloc[:,:-1].values)
    my_predict = my_predict.tolist()
    return render_template('result.html', prediction = my_predict)

if __name__ == '__main__':
    app.run(debug= True)

