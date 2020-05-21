import math
import os

import numpy as np
import pandas as pd
import scipy.signal as sgl
import scipy.stats as st
from tensorflow import keras
from zipfile import ZipFile

from flask import render_template, request, redirect, url_for, flash
from signature_verification import app, bcrypt, db
from signature_verification.models import User, Notification, History
from flask_login import login_user, current_user, logout_user, login_required


@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    return render_template('login.html')


@app.route('/signup')
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    return render_template('signup.html')


@app.route('/signedup', methods=["GET", "POST"])
def signedup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == "POST":
        try:
            name_cond = User.query.filter_by(name=request.form['name']).first()
            email_cond = User.query.filter_by(name=request.form['email']).first()
            if name_cond:
                flash('Name you entered is already taken!')
                return redirect(url_for('signup'))
            if email_cond:
                flash('Email you entered is already taken!')
                return redirect(url_for('signup'))

            hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

            print(request.form['company'])
            # checking whether the user is a representative of a company
            if int(request.form['company']) == 1:
                user = User(
                    name=request.form['name'],
                    email=request.form['email'],
                    password=hashed_password,
                    address=request.form['address'],
                    represents_company=request.form['company'],
                    company_name=request.form['company_name'],
                    no_of_employees=request.form['users'],
                )
                db.session.add(user)

                # extract zip file and place the files in the correct directory
                if request.files:
                    initial_path = os.getcwd() 
                    print('cwd:', initial_path)
                    print("Zip file available.")
                    zipFile = request.files["zipfile"]
                    
                    dir_path = './models/' + user.company_name + '/train'
                    print('trying to make directories at:', dir_path)
                    os.makedirs(dir_path)
                    print('made directories at:', dir_path)
                    
                    os.chdir(dir_path)
                    print('extracting zip at:', os.getcwd())
                    zipFile.save('uploaded.zip')

                    with ZipFile('uploaded.zip', 'r') as zip:
                        zip.extractall()
                        print('extracting all the files done!')

                    print('zip extracted and placed in its rightful place as the earl!')

                    os.remove('uploaded.zip')
                    os.chdir(initial_path)
                    print('cwd:', os.getcwd())
                    
                    admin_email = User.query.filter_by(name='admin').first().email
                    notif_to_admin = Notification(
                        from_user = user.email,
                        to_user = admin_email, 
                        message = 'A new user representing a company, '+ user.company_name + ' has signed in. Please update the models.'
                    )
                    db.session.add(notif_to_admin)

                    notif_to_user = Notification(
                        from_user = admin_email,
                        to_user = user.email, 
                        message = 'Please wait for the administrator to upload the Deep Learning models built from your signature data.'
                    )
                    db.session.add(notif_to_user)
                    
                    db.session.commit()
                else:
                    print('error while extracting and placing the zip!')
                
            else:
                user = User(
                    name=request.form['name'],
                    email=request.form['email'],
                    password=hashed_password,
                    address=request.form['address'],
                    represents_company=request.form['company'],
                    company_name=request.form['name'],
                )
                db.session.add(user)

                # extract zip file and place the files in the correct directory
                if request.files:
                    initial_path = os.getcwd() 
                    print('cwd:', initial_path)
                    print("Zip file available.")
                    zipFile = request.files["zipfile"]
                    
                    dir_path = './models/' + user.name + '/train'
                    print('trying to make directories at:', dir_path)
                    os.makedirs(dir_path)
                    print('made directories at:', dir_path)
                    
                    os.chdir(dir_path)

                    # zip_path = './models/' + user.company_name + '/train/uploaded.zip'
                    print('extracting zip at:', os.getcwd())
                    zipFile.save('uploaded.zip')

                    with ZipFile('uploaded.zip', 'r') as zip:
                        zip.extractall()
                        print('extracting all the files done!')

                    print('zip extracted and placed in its rightful place as the earl!')

                    os.remove('uploaded.zip')
                    os.chdir(initial_path)
                    print('cwd:', os.getcwd())
                    
                    if request.form['name'] != 'admin':
                        admin_email = User.query.filter_by(name='admin').first().email
                        notif_to_admin = Notification(
                            from_user = user.email,
                            to_user = admin_email, 
                            message = 'A new user, '+ user.name + ' has signed in. Please update the models.'
                        )
                        db.session.add(notif_to_admin)

                        notif_to_user = Notification(
                            from_user = admin_email,
                            to_user = user.email, 
                            message = 'Please wait for the administrator to upload the Deep Learning models built from your signature data.'
                        )
                        db.session.add(notif_to_user)
                        
                    db.session.commit()
                else:
                    print('error while extracting and placing the zip in single user section!')

            print('Sign up successful!')

            return redirect(url_for('login'))
        except Exception as e:
            print('Sign Up failed!')
            print(e)
            return redirect(url_for('signup'))


@app.route('/', methods=["GET", "POST"])
def home():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            user = User.query.filter_by(email=request.form['email']).first()
            if user and bcrypt.check_password_hash(user.password, request.form['password']):
                login_user(user, remember=True)
                print('Login successful!')
                return redirect(url_for('index'))
            else:
                print("Login failed, check your credentials!")
                return redirect(url_for('login'))
        except Exception as e:
            print('Login failed!')
            print(e)

    return redirect(url_for('login'))


@app.route('/index')
@login_required
def index():
    return render_template('index.html', email=current_user.email)


@app.route('/sign_out')
def sign_out():
    logout_user()
    return redirect(url_for('login'))


@app.route('/notifications')
@login_required
def notifications():
    my_notifications = current_user.notifications
    print('getting current user notifications:', current_user.name)

    return render_template('notifications.html', notifications = my_notifications, email=current_user.email)


@app.route('/solved', methods=["POST"])
def solved():
    if current_user.name == 'admin':
        user_email = Notification.query.filter_by(message=request.form['message']).first().from_user
        Notification.query.filter_by(message=request.form['message']).delete()
        db.session.commit()

        admin_email = User.query.filter_by(name='admin').first().email

        print('Solved notification:')
        print('from:', admin_email)
        print('to:', current_user.email)
        notif_to_user = Notification(
            from_user = admin_email, 
            to_user = user_email,
            message = 'Custom built machine learning models have been added by the administrator. You can now verify new signatures!'
        )
        db.session.add(notif_to_user)
        db.session.commit()
    else:
        Notification.query.filter_by(message=request.form['message']).delete()
        db.session.commit()

    return redirect(url_for('notifications'))


@app.route('/file_upload', methods=["GET", "POST"])
@login_required
def file_upload():
    return render_template('file_upload.html', email=current_user.email)


def extract_features():
    V = []
    SDX = []
    SDY = []
    A = []
    SDV = []
    SDA = []

    file = pd.read_csv("uploadedFile.txt", delimiter=' ', names=['X', 'Y', 'TS', 'BS', 'AZ', 'AL', 'P'],
                       header=None,
                       skiprows=1)
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    aX = sum(X) / file_size
    aY = sum(Y) / file_size
    for k in range(0, file_size - 1):
        if TS[k] == TS[k + 1]:
            X[k + 1] = (X[k] + X[k + 1]) / 2
            Y[k + 1] = (Y[k] + Y[k + 1]) / 2
            TS[k + 1] = (TS[k] + 1)
            BS[k + 1] = (BS[k] + BS[k + 1]) / 2
            AZ[k + 1] = (AZ[k] + AZ[k + 1]) / 2
            AL[k + 1] = (AL[k] + AL[k + 1]) / 2
            P[k + 1] = (P[k] + P[k + 1]) / 2
        if k < file_size:
            V.append(((math.sqrt((X[k + 1] - X[k]) ** 2 + (Y[k + 1] - Y[k]) ** 2)) * (TS[file_size - 1] - TS[0])) / (
                    TS[k + 1] - TS[k]))
        SDX.append((X[k] - aX) ** 2)
        SDY.append((Y[k] - aY) ** 2)
    SDX.append((X[file_size - 1] - aX) ** 2)
    SDY.append((Y[file_size - 1] - aY) ** 2)
    V.append(0)
    data = {'X': X, 'Y': Y, 'TS': TS, 'BS': BS, 'AZ': AZ,
            'AL': AL, 'P': P, 'V': V, 'SDX': SDX, 'SDY': SDY}
    df = pd.DataFrame(data)

    file = df
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    V = file['V']
    SDX = file['SDX']
    SDY = file['SDY']
    for k in range(0, file_size):
        if k < file_size - 1:
            A.append(((abs(V[k + 1] - V[k])) *
                      (TS[file_size - 1] - TS[0])) / (TS[k + 1] - TS[k]))
    A.append(0)
    data = {'X': X, 'Y': Y, 'TS': TS, 'BS': BS, 'AZ': AZ,
            'AL': AL, 'P': P, 'V': V, 'SDX': SDX, 'SDY': SDY, 'A': A}
    df = pd.DataFrame(data)

    file = df
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    V = file['V']
    SDX = file['SDX']
    SDY = file['SDY']
    A = file['A']
    aV = sum(V) / file_size
    aA = sum(A) / file_size
    for k in range(0, file_size):
        SDV.append((V[k] - aV) ** 2)
        SDA.append((A[k] - aA) ** 2)
    data = {'X': X, 'Y': Y, 'TS': TS, 'BS': BS, 'AZ': AZ, 'AL': AL, 'P': P, 'V': V, 'SDX': SDX, 'SDY': SDY, 'A': A,
            'SDV': SDV, 'SDA': SDA}
    df = pd.DataFrame(data)

    avgX = []
    avgY = []
    avgSDX = []
    avgSDY = []
    avgV = []
    avgA = []
    avgSDV = []
    avgSDA = []
    pen_down = []
    pen_up = []
    pen_ratio = []
    sign_width = []
    sign_height = []
    width_height_ratio = []
    total_sign_duration = []
    range_pressure = []

    max_pressure = []
    sample_points = []
    sample_points_to_width = []
    mean_pressure = []
    pressure_variance = []
    avg_x_velocity = []
    avg_y_velocity = []
    max_x_velocity = []
    max_y_velocity = []
    samples_positive_x_velocity = []
    samples_positive_y_velocity = []
    variance_x_velocity = []
    variance_y_velocity = []
    std_x_velocity = []
    std_y_velocity = []
    median_x_velocity = []
    median_y_velocity = []
    mode_x_velocity = []
    mode_y_velocity = []
    corr_x_y_velocity = []
    mean_x_acceleration = []
    mean_y_acceleration = []
    corr_x_y_acceleration = []
    variance_x_acceleration = []
    variance_y_acceleration = []
    std_x_acceleration = []
    std_y_acceleration = []
    x_local_minima = []
    y_local_minima = []

    file = df
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    V = file['V']
    SDX = file['SDX']
    SDY = file['SDY']
    A = file['A']
    SDV = file['SDV']
    SDA = file['SDA']

    avgX.append(sum(X) / file_size)
    avgY.append(sum(Y) / file_size)
    avgSDX.append(sum(SDX) / file_size)
    avgSDY.append(sum(SDY) / file_size)
    avgV.append(sum(V) / file_size)
    avgA.append(sum(A) / file_size)
    avgSDV.append(sum(SDV) / file_size)
    avgSDA.append(sum(SDA) / file_size)
    pen_down.append(sum(BS))
    pen_up.append(file_size - sum(BS))
    pen_ratio.append((sum(BS)) / (file_size - sum(BS)))
    sign_width.append(max(X) - min(X))
    sign_height.append(max(Y) - min(Y))
    width_height_ratio.append((max(X) - min(X)) / (max(Y) - min(Y)))
    total_sign_duration.append(TS[file_size - 1] - TS[0])
    range_pressure.append(max(P) - min(P))

    sample_points.append(file_size)
    sample_points_to_width.append(file_size / (max(X) - min(X)))
    max_pressure.append(max(P))
    mean_pressure.append(np.mean(P))
    pressure_variance.append(np.var(P))

    # calculating x, y velocities
    x_displacement = []
    y_displacement = []
    x_velocity = []
    y_velocity = []
    x_acceleration = []
    y_acceleration = []
    for k in range(0, file_size - 1):
        x_displacement = X[k + 1] - X[k]
        y_displacement = Y[k + 1] - Y[k]
        time = TS[k + 1] - TS[k]

        x_velocity.append(x_displacement / time)
        y_velocity.append(y_displacement / time)

        x_acceleration.append(x_displacement / (time ** 2))
        y_acceleration.append(y_displacement / (time ** 2))

    avg_x_velocity.append(np.mean(x_velocity))
    avg_y_velocity.append(np.mean(y_velocity))

    max_x_velocity.append(max(x_velocity))
    max_y_velocity.append(max(y_velocity))

    samples_positive_x_velocity.append(len([x for x in x_velocity if x > 0]))
    samples_positive_y_velocity.append(len([y for y in y_velocity if y > 0]))

    variance_x_velocity.append(np.var(x_velocity))
    variance_y_velocity.append(np.var(y_velocity))

    std_x_velocity.append(np.std(x_velocity))
    std_y_velocity.append(np.std(y_velocity))

    median_x_velocity.append(np.median(x_velocity))
    median_y_velocity.append(np.median(y_velocity))

    #         mode_x_velocity.append(max(set(x_velocity), key=x_velocity.count))
    #         mode_y_velocity.append(max(set(y_velocity), key=y_velocity.count))

    corr_velocity, _ = st.pearsonr(x_velocity, y_velocity)
    corr_x_y_velocity.append(corr_velocity)

    mean_x_acceleration.append(np.mean(x_acceleration))
    mean_y_acceleration.append(np.mean(y_acceleration))

    corr_acceleration, _ = st.pearsonr(x_acceleration, y_acceleration)
    corr_x_y_acceleration.append(corr_acceleration)

    variance_x_acceleration.append(np.var(x_acceleration))
    variance_y_acceleration.append(np.var(y_acceleration))

    std_x_acceleration.append(np.std(x_acceleration))
    std_y_acceleration.append(np.std(y_acceleration))

    x_local_minima.append(len(sgl.argrelextrema(np.array(X), np.less)[0]))
    y_local_minima.append(len(sgl.argrelextrema(np.array(Y), np.less)[0]))

    data = {'avgX': avgX,
            'avgY': avgY,
            'avgSDX': avgSDX,
            'avgSDY': avgSDY,
            'avgV': avgV,
            'avgA': avgA,
            'avgSDV': avgSDV,
            'avgSDA': avgSDA,
            'pen_down': pen_down,
            'pen_up': pen_up,
            'pen_ratio': pen_ratio,
            'sign_width': sign_width,
            'sign_height': sign_height,
            'width_height_ratio': width_height_ratio,
            'total_sign_duration': total_sign_duration,
            'range_pressure': range_pressure,

            'max_pressure': max_pressure,
            'sample_points': sample_points,
            'sample_points_to_width': sample_points_to_width,
            'mean_pressure': mean_pressure,
            'pressure_variance': pressure_variance,
            'avg_x_velocity': avg_x_velocity,
            'avg_y_velocity': avg_y_velocity,
            'max_x_velocity': max_x_velocity,
            'max_y_velocity': max_y_velocity,
            'samples_positive_x_velocity': samples_positive_x_velocity,
            'samples_positive_y_velocity': samples_positive_y_velocity,
            'variance_x_velocity': variance_x_velocity,
            'variance_y_velocity': variance_y_velocity,
            'std_x_velocity': std_x_velocity,
            'std_y_velocity': std_y_velocity,
            'median_x_velocity': median_x_velocity,
            'median_y_velocity': median_y_velocity,
            #     'mode_x_velocity': mode_x_velocity,
            #     'mode_y_velocity': mode_y_velocity,
            'corr_x_y_velocity': corr_x_y_velocity,
            'mean_x_acceleration': mean_x_acceleration,
            'mean_y_acceleration': mean_y_acceleration,
            'corr_x_y_acceleration': corr_x_y_acceleration,
            'variance_x_acceleration': variance_x_acceleration,
            'variance_y_acceleration': variance_y_acceleration,
            'std_x_acceleration': std_x_acceleration,
            'std_y_acceleration': std_y_acceleration,
            'x_local_minima': x_local_minima,
            'y_local_minima': y_local_minima}

    df = pd.DataFrame(data)

    company_name = current_user.company_name
    dataset = pd.read_csv('./models/' + company_name + '/Features.csv')
    dataset = dataset[
        ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
         'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure', 'max_pressure',
         'sample_points', 'sample_points_to_width', 'mean_pressure', 'pressure_variance', 'avg_x_velocity',
         'avg_y_velocity', 'max_x_velocity', 'max_y_velocity', 'samples_positive_x_velocity',
         'samples_positive_y_velocity', 'variance_x_velocity', 'variance_y_velocity', 'std_x_velocity',
         'std_y_velocity', 'median_x_velocity', 'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration',
         'mean_y_acceleration', 'corr_x_y_acceleration', 'variance_x_acceleration', 'variance_y_acceleration',
         'std_x_acceleration', 'std_y_acceleration', 'x_local_minima', 'y_local_minima']]

    df = (df - dataset.min()) / (dataset.max() - dataset.min())

    return list(df.iloc[0])


@app.route('/predict', methods=["POST"])
@login_required
def predict():
    if request.method == "POST":
        if request.files:
            print('File uploaded is available.')
            signFile = request.files["signature_data"]
            signFile.save("uploadedFile.txt")

            try:
                features = extract_features()
            except ZeroDivisionError as err:
                return render_template('404.html')

            print(features)

            company_name = current_user.company_name

            if current_user.represents_company:
                # model 1 detects the id of the user of the signature
                model_one_name = './models/' + company_name + '/model1.h5'

                model = keras.models.load_model(model_one_name)

                user = np.argmax(model.predict([features]))

                print('user prediction result of the uploaded file:', user)

                # model 2 predicts if the signature is genuine or forged
                model2_name = './models/' + company_name + '/user_models/model2_' + str(user) + '_op.h5'
                model2 = keras.models.load_model(model2_name)

                forgery_status = model2.predict([features])
            else:
                # model 2 predicts if the signature is genuine or forged
                model2_name = './models/' + company_name + '/user_models/model2_' + '0' + '_op.h5'
                model2 = keras.models.load_model(model2_name)

                forgery_status = model2.predict([features])


            print('forgery status:', int(round(forgery_status[0][0])))

            if int(round(forgery_status[0][0])) == 0:
                return render_template('genuine.html')
            else:
                return render_template('forged.html')

    else:
        return render_template('404.html')
