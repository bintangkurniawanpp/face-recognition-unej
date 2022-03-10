from app import db, app, bcrypt
##import jwt
import datetime


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    NIM = db.Column(db.Integer(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password = db.Column(db.String(128), nullable=False)
    admin = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, name, NIM, email, password, admin=False):
        self.name = name
        self.NIM = NIM
        self.email = email
        # self.password = bcrypt.generate_password_hash(password).decode('utf-8')
        self.admin = admin

    # def encode_auth_token(self, user_id):
    #     try:
    #         payload = {
    #             'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5),
    #             'iat': datetime.datetime.utcnow(),
    #             'sub': user_id
    #         }
    #         return jwt.encode(
    #             payload,
    #             app.config.get('SECRET_KEY'),
    #             algorithm='HS256'
    #         )
    #     except Exception as e:
    #         return e