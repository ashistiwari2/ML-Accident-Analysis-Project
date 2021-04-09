from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('templates/DataSets/mlp.pkl', 'rb'))


app = Flask(__name__)


def calculate(form):
    age_of_driver = form['age']
    vehicle_type = form['vehicle_type']
    age_of_vehicle = form['vehicle_age']
    engine_cc = form['v_capacity']
    day = form['day']
    weather = form['weather_condition']
    light = form['light_condition']
    road_condition = form['road_condition']
    gender = form['gender']
    speed_limit = form['speed_limit']
    arr = np.array([[age_of_driver, vehicle_type, engine_cc, day, weather, road_condition, age_of_vehicle, light, gender, speed_limit]])
    pred = model.predict(arr)
    return render_template("result.html", prediction=pred)


@app.route('/', methods=['GET'])
def homepage():
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    road_conditions = ['Dry', 'Wet or Damp', 'Snow', 'Frost or ice', 'Flood over 3cm. deep', 'Oil or diesel', 'Mud']
    weather_conditions = ['Fine no high winds', 'Raining no high winds', 'Snowing no high winds', 'Fine + high '
                                                                                                  'winds',
                          'Raining + high winds', 'Snowing + high winds', 'Fog or mist', 'Other']
    return render_template("index.html", road_conditions=road_conditions, days=days, weather_conditions=weather_conditions)


@app.route('/', methods=['POST'])
def name():
    return calculate(request.form)


if __name__ == '__main__':
    app.run(debug=True)
