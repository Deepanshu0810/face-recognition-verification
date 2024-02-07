from flask import Flask
from .views import views
import os

def create_app():
    app = Flask(__name__)
    app.config['upload_folder'] = os.path.join(os.getcwd(),'data','test') 
    app.register_blueprint(views)
    return app