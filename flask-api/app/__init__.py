from flask import Flask, app, request, jsonify
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bycrypt import Bcrypt



app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bycrpt = Bcrypt(app)

from app.model import user
from app import routes


