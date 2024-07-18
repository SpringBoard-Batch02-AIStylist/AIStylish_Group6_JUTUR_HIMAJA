from flask import Flask, request, render_template, redirect, url_for
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential

app = Flask(__name__)

# Define the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = Sequential([base_model, GlobalMaxPooling2D()])

# Load image features and filenames
with open('features.pkl', 'rb') as f:
    image_features = pkl.load(f)

with open('images.pkl', 'rb') as f:
    filenames = pkl.load(f)

# Function to recommend similar images
def recommend_similar_images(query_features, image_features, filenames, top_n=5):
    similarities = cosine_similarity(query_features, image_features)
    similarity_indices = similarities[0].argsort()[-top_n:][::-1]  # Top N similar images
    similar_images = [(filenames[idx], similarities[0, idx]) for idx in similarity_indices]
    return similar_images

# Function to display the query image and similar images
def display_query_and_similar_images(query_image_path, similar_images):
    query_image = imread(query_image_path)
    
    top_n = len(similar_images)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis('off')

    # Display similar images
    for i, (image_path, similarity_score) in enumerate(similar_images, start=2):
        similar_image = imread(image_path)
        plt.subplot(1, top_n + 1, i)
        plt.imshow(similar_image)
        plt.title(f"Similarity: {similarity_score * 100:.2f}%")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot to a temporary file
    plot_path = 'static/temp_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

# Define the directory for saving the model
model_directory = 'models'
os.makedirs(model_directory, exist_ok=True)
model_path = os.path.join(model_directory, 'my_model.h5')

# Check if the file exists and remove it (optional)
if os.path.exists(model_path):
    os.remove(model_path)

# Save the model in Keras format
model.save(model_path)

print(f'Model saved at {model_path}')

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and recommendation display
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select a file, browser also submit an empty file without filename
        if file.filename == '':
            return redirect(request.url)
        
        # Save uploaded image
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, file.filename)
        file.save(img_path)
        
        # Process uploaded image for recommendation
        query_image = image.load_img(img_path, target_size=(224, 224))
        query_image = image.img_to_array(query_image)
        query_image = np.expand_dims(query_image, axis=0)
        query_image = preprocess_input(query_image)
        
        # Get recommendations
        query_image_features = model.predict(query_image).reshape(1, -1)
        similar_images = recommend_similar_images(query_image_features, image_features, filenames, top_n=4)
        
        # Display the query image and similar images
        plot_path = display_query_and_similar_images(query_image_path=img_path, similar_images=similar_images)
        
        return render_template('result.html', query_image=img_path, similar_images=[img[0].replace('static', '/static') for img in similar_images], plot=plot_path.replace('static', '/static'))

if __name__ == '__main__':
    app.run(debug=True)
