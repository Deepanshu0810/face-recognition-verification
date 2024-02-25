from flask import render_template, Blueprint, request, Response
from training.facenet import FacenetModel
import os
import cv2 as cv

views = Blueprint('views', __name__)

model = FacenetModel()

@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@views.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'POST':
        img = request.files['image']
        print(img)
        if(img):
            img.save(os.path.join(os.getcwd(),'app','static','input',img.filename))
            img_array,name,dis = model.recognize(os.path.join(os.getcwd(),'app','static','input',img.filename))
            cv.imwrite(os.path.join(os.getcwd(),'app','static','results',img.filename),img_array)
            context = {
                'name':name,
                'dis':dis,
                'img':os.path.join('static','results',img.filename)
            }   
            print(context['name'])
            print(context['img'])
            return render_template("recognition.html",context=context)

    
    return render_template("recognition.html")

@views.route('/face-verification', methods=['GET', 'POST'])
def face_verification():
    if request.method == 'POST':
        img1 = request.files['image1']
        img2 = request.files['image2']
        if(img1 and img2):
            img1.save(os.path.join(os.getcwd(),'app','static','input',img1.filename))
            img2.save(os.path.join(os.getcwd(),'app','static','input',img2.filename))
            img1_arr, img2_arr = model.verify(os.path.join(os.getcwd(),'app','static','input',img1.filename),os.path.join(os.getcwd(),'app','static','input',img2.filename))
            cv.imwrite(os.path.join(os.getcwd(),'app','static','results',img1.filename),img1_arr)
            cv.imwrite(os.path.join(os.getcwd(),'app','static','results',img2.filename),img2_arr)
            context = {
                'img1':os.path.join('static','results',img1.filename),
                'img2':os.path.join('static','results',img2.filename)
            }
            return render_template("verification.html",context=context)

    return render_template("verification.html")

@views.route('/live', methods=['GET', 'POST'])
def live():
    context = True
    return render_template("live.html",context=context)

@views.route('/video', methods=['GET', 'POST'])
def video():
    return Response(model.recognize_live(), mimetype='multipart/x-mixed-replace; boundary=frame')