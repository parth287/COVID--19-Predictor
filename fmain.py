from flask import Flask, render_template , request
import pickle
app = Flask(__name__)

# importing the model
file = open("covid.pkl", "rb")
model = pickle.load(file)
file.close()

@app.route("/", methods = ["GET","POST"])
def index():
    if request.method == "POST":
        info = request.form
        fever = int(info['fever'])
        age = int(info['age'])
        cough = int(info['cough'])
        tiredness = int(info['tiredness'])
        pain = int(info['pain'])
        breath = int(info['breath'])
        throat = int(info['throat'])
        inf = model.predict_proba([[fever, cough, tiredness, throat, pain, breath, age]])[0][1]
        inf = inf * 100
        return render_template("result.html", inf = inf)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)


