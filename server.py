import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from flask_sqlalchemy import SQLAlchemy
import numpy as np

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
import uuid
import os 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'public/images'

engine = create_engine('sqlite:///database.db', echo=True)
Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()

model = load_model('Plant_disease_model.h5')

labels = ['Healthy', 'Powdery', 'Rust']

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    predictions = relationship('Prediction', back_populates='user')
    
class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    image = Column(String, nullable=False)
    created_at = Column(Integer, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship('User', back_populates='predictions')
    
    powdery_percent = Column(String, nullable=True)
    healthy_percent = Column(String, nullable=True)
    rust_percent = Column(String, nullable=True)
    label = Column(String, nullable=True)

Base.metadata.create_all(engine)

def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

@app.post('/predict')
def predict():
    image = request.files['image']
    user_id = request.form['user_id']
    
    filename = str(uuid.uuid4()) + '.jpg'
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    processed_image = preprocess_image(os.path.join('public/images', filename))
    prediction = model.predict(processed_image)

    index = np.argmax(prediction[0])
    session.add(
        Prediction(
            image=filename,
            created_at=int(round(time.time() * 1000)),
            powdery_percent='{:.2f}%'.format(prediction[0][1] * 100),
            healthy_percent='{:.2f}%'.format(prediction[0][0] * 100),
            rust_percent='{:.2f}%'.format(prediction[0][2] * 100),
            user_id=user_id,
            label=labels[index]
        )
    )
    session.commit()
    
    return jsonify({
        'prediction': str(labels[index]),
        'percentages': {
            'healthy': '{:.2f}%'.format(prediction[0][0] * 100),
            'powdery': '{:.2f}%'.format(prediction[0][1] * 100),
            'rust': '{:.2f}%'.format(prediction[0][2] * 100)
        }
    })

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data['name']
    username = data['username']
    password = generate_password_hash(data['password'])
    user = session.query(User).filter_by(username=username).first()
    if user:
        return jsonify({'message': 'User already exists'}), 400
    user = User(username=username, password=password, name=name)
    session.add(user)
    session.commit()
    return jsonify({'message': 'User registered successfully'})



@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    search_user = session.query(User).filter_by(username=username).first()
    if search_user and check_password_hash(search_user.password, password):
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': search_user.id,
                'name': search_user.name,
                'username': search_user.username
            }
        })
    else:
        return jsonify({'message': 'Invalid username or password'})

@app.get('/users/<int:user_id>/predictions')
def get_predictions(user_id):
    print(user_id)
    predictions = session.query(Prediction).filter_by(user_id=user_id).all()
    print(predictions)
    return jsonify([{
            'image': prediction.image,
            'created_at': prediction.created_at,
            'label': prediction.label,
            'powdery_percent': prediction.powdery_percent,
            'healthy_percent': prediction.healthy_percent,
            'rust_percent': prediction.rust_percent
        } for prediction in predictions])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)