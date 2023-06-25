from flask import Flask, render_template, request, redirect,url_for,jsonify
from flask_bootstrap import Bootstrap
from model import *

app = Flask(__name__ , static_url_path='/static')
Bootstrap(app)

def create_app():
    app = Flask(__name__)
    Bootstrap(app)
    return app

@app.route("/")
def home():
    if request.method == "POST":
        string = mainModel()
        return redirect(url_for('tes'))
    else :
        return render_template('home.html')

@app.route("/mealplan",methods =["GET", "POST"])
def meal_plan():
    if request.method == "POST" :
        # string = mainModel()
        bodyLevel,calNeeds,protNeeds,carboNeeds,fatNeeds = calculateNeeds(request.form['age'],request.form['weight'],request.form['height'],request.form['gender'],request.form['activityLevel'])
        mealPlan=createMealPlan(calNeeds,protNeeds,carboNeeds,fatNeeds)
        return render_template('mealplan.html',bodyLevel=bodyLevel,calNeeds=calNeeds,protNeeds=protNeeds,carboNeeds=carboNeeds,fatNeeds=fatNeeds,mealPlan=mealPlan)
    if request.method == "GET":
        return redirect(url_for('home'))

@app.route("/get-body-status",methods =["POST"])
def measuring_body():
    bmi,bodyLevel = calculatingBody(request.form['weight'], request.form['height'])
    # bmi=23
    # bodyLevel="fff"
    dataBody = {
        'bmi' : bmi,
        'bodyLevel' : bodyLevel
    }
    return jsonify(dataBody)
if __name__ == '__main__':
    app.run(debug=True)   