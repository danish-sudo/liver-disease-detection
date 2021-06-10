from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow import keras
import numpy as np
import pydicom
from PIL import Image
from numpy import genfromtxt
from skimage.transform import resize
from Model_Development.training import resUnet, datagen
import os

UPLOAD_FOLDER = '/media/danish/Mint/FYP/Web/static/Upload'

app = Flask(__name__)
app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
image_datatype = float


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect')
def predict():
    return render_template('predict.html')


@app.route('/report', methods=['POST', 'GET'])
def report():
    upload_path = '/Upload'
    name = session['name']
    email = session['uemail']
    dob = session['dob']
    filename = session['filename']

    input_image = upload_path + '/Abdominal/' + filename + '.png'
    liver_mask = os.path.join(upload_path, 'Detected_Liver', filename + '_liver.png')
    tumor_mask = os.path.join(upload_path, 'Detected_Tumors', filename + '_tumor.png')
    return render_template('report.html', name=name, input=input_image, liver=liver_mask, tumor=tumor_mask)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        name = request.form['name']

        email = request.form['email']

        dob = request.form['birthday']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Liver Extraction
        model = keras.models.load_model(os.path.join('../Model_Development/models', 'liver_model.h5'), compile=False)
        model.compile()
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        img = []
        image_size = 256
        #
        dicom_image = pydicom.dcmread(image_path, force=True)
        print(type(dicom_image.pixel_array))
        print(dicom_image.pixel_array.shape)
        out_dir = '/media/danish/Mint/FYP/Web/static/Upload/Abdominal/'
        input_image = Image.fromarray(dicom_image.pixel_array)
        input_image = input_image.convert("L")
        input_image.save(out_dir + filename + '.png')
        image = datagen.preprocess_scans(dicom_image.pixel_array)
        image = datagen.normalize_scans(image)
        image = np.array(Image.fromarray(image).resize([image_size, image_size])).astype(image_datatype)
        image = np.stack((image,) * 3, axis=-1)
        img.append(image)
        image = np.array(img)

        weights = model.get_weights()
        new_model = resUnet.resUnet()
        adam = keras.optimizers.Adam()
        new_model.compile(optimizer=adam, loss=resUnet.dice_coef_loss, metrics=["acc", resUnet.dice_coef])
        new_model.set_weights(weights)
        pred = new_model.predict(image)

        pred = np.reshape(pred * 255, (256, 256))
        liver_path = '/media/danish/Mint/FYP/Web/static/Upload/Detected_Liver/'
        file_name = filename + '_liver'
        np.savetxt(liver_path + file_name + '.csv', pred, delimiter=',')
        im = Image.fromarray(pred)
        im = im.convert("L")
        im.save(liver_path + file_name + '.png')

        # Tumor Extraction
        liver_mask = genfromtxt(liver_path + file_name + '.csv', delimiter=',')
        img = []
        image_path = '/media/danish/Mint/FYP/Web/static/Upload/' + filename
        dicom_image = pydicom.dcmread(image_path)
        image = datagen.preprocess_scans(dicom_image.pixel_array)

        image = resize(image, [256, 256])

        image = np.multiply(image, np.clip(liver_mask, 0, 1))

        image = np.array(Image.fromarray(image).resize([image_size, image_size])).astype(float)

        image = np.stack((image,) * 3, axis=-1)
        img.append(image)
        image = np.array(img)

        model1 = resUnet.resUnet()
        adam = keras.optimizers.Adam()
        model1.compile(optimizer=adam, loss=resUnet.dice_coef_loss, metrics=["acc", resUnet.dice_coef])
        # model1.summary()

        model1.load_weights(os.path.join('../Model_Development/models', 'tumor_weights_final_50epochs.h5'))

        model1.compile()
        weights = model1.get_weights()
        new_model = resUnet.resUnet()
        adam = keras.optimizers.Adam()
        new_model.compile(optimizer=adam, loss=resUnet.dice_coef_loss, metrics=["acc", resUnet.dice_coef])

        new_model.set_weights(weights)

        pred = new_model.predict(image)

        pred = np.reshape(pred * 255, (256, 256))
        tumor_path = '/media/danish/Mint/FYP/Web/static/Upload/Detected_Tumors/'
        file_name = filename + '_tumor'
        im = Image.fromarray(pred)
        im = im.convert("L")
        im.save(tumor_path + file_name + '.png')

        session['name'] = name
        session['uemail'] = email
        session['dob'] = dob
        session['filename'] = filename
        print(session['name'])
        return redirect(url_for('report'))


if __name__ == '__main__':
    app.run(debug=True)
