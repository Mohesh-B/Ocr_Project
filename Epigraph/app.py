import os
from flask import Flask, render_template, request
from google.cloud import vision
from google.oauth2 import service_account
from werkzeug.utils import secure_filename
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify ,url_for
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import unicodedata
import random

# Configure upload folder


app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
SEGMENT_FOLDER = os.path.join(app.static_folder, 'segmented')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENT_FOLDER'] = SEGMENT_FOLDER

@app.route('/')
def index():
    return render_template('index.html')



# ✅ Function to preprocess and segment characters

def preprocess_and_segment(image_path, output_path):
    """Preprocesses an image and segments individual letters using connected components."""
    
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to highlight text
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # Remove small noise by morphological opening
    kernel_noise = np.ones((2,2), np.uint8)  
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)

    # Use a small vertical kernel to preserve letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Draw bounding boxes around detected letters
    output_image = image.copy()
    for i in range(1, num_labels):  # Ignore background (i=0)
        x, y, w, h, area = stats[i]

        # Condition to filter out noise and large merged components
        if 10 < w < 150 and 10 < h < 150:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save segmented image
    cv2.imwrite(output_path, output_image)

# ✅ Function to extract text using Google Cloud Vision API
def extract_text_from_image(image_path):
    credentials_path = "C:/Users/Mohesh B/Desktop/Final_year_project/stone-cathode-452715-r6-060743faa38e.json"
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    extracted_text = response.text_annotations[0].description if response.text_annotations else "No text detected"

    return extracted_text

@app.route('/upload/doc', methods=['GET', 'POST'])
def upload_image_doc():
    extracted_text = None
    uploaded_filename = None
    segmented_filename = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            uploaded_filename = filename  # ✅ Ensure this is set properly

            # Perform OCR
            extracted_text = extract_text_from_image(filepath)

            # Perform Character Segmentation
            segmented_filename = f"segmented_{filename}"
            segmented_filepath = os.path.join(app.config['SEGMENT_FOLDER'], segmented_filename)
            preprocess_and_segment(filepath, segmented_filepath)

    return render_template('uploaddoc.html', 
                           extracted_text=extracted_text, 
                           uploaded_filename=uploaded_filename,  # ✅ Ensure this is passed
                           segmented_filename=segmented_filename)

@app.route('/upload/stone', methods=['GET', 'POST'])
def upload_stone():
    extracted_text = None
    uploaded_filename = None
    segmented_filename = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            uploaded_filename = filename  # ✅ Ensure this is set properly

            # Perform OCR
            extracted_text = extract_text_from_image(filepath)

            # Perform Character Segmentation
            segmented_filename = f"segmented_{filename}"
            segmented_filepath = os.path.join(app.config['SEGMENT_FOLDER'], segmented_filename)
            preprocess_and_segment(filepath, segmented_filepath)

    return render_template('upload_stone.html', 
                           extracted_text=extracted_text, 
                           uploaded_filename=uploaded_filename,  # ✅ Ensure this is passed
                           segmented_filename=segmented_filename)

# ✅ Route for Upload Page@app.route('/upload', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    extracted_text = None
    uploaded_filename = None
    segmented_filename = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            uploaded_filename = filename  # ✅ Ensure this is set properly

            # Perform OCR
            extracted_text = extract_text_from_image(filepath)

            # Perform Character Segmentation
            segmented_filename = f"segmented_{filename}"
            segmented_filepath = os.path.join(app.config['SEGMENT_FOLDER'], segmented_filename)
            preprocess_and_segment(filepath, segmented_filepath)

    return render_template('upload.html', 
                           extracted_text=extracted_text, 
                           uploaded_filename=uploaded_filename,  # ✅ Ensure this is passed
                           segmented_filename=segmented_filename)
def transliterate_tamil(text, target_script):
    """Transliterates Tamil text to a given script (English, Malayalam, Telugu, Hindi)."""
    normalized_text = unicodedata.normalize('NFKD', text)
    
    # Apply custom mapping for specific characters
    normalized_text = normalized_text.replace('ன', 'na')

    if target_script == "en":
        return transliterate(normalized_text, sanscript.TAMIL, sanscript.ITRANS)
    elif target_script == "ml":
        return transliterate(normalized_text, sanscript.TAMIL, sanscript.MALAYALAM)
    elif target_script == "te":
        return transliterate(normalized_text, sanscript.TAMIL, sanscript.TELUGU)
    elif target_script == "hi":
        return transliterate(normalized_text, sanscript.TAMIL, sanscript.DEVANAGARI)
    else:
        return "Invalid script selection."

# API endpoint for transliteration
@app.route("/transliterate", methods=["GET"])
def transliterate_text():
    text = request.args.get("text", "")
    target_lang = request.args.get("lang", "en")  # Default is English
    transliterated_text = transliterate_tamil(text, target_lang)
    return jsonify({"transliterated_text": transliterated_text})


@app.route('/palmscripts')
def palmscripts():
    # Define the folder path for palm script images
    image_folder = os.path.join(app.static_folder, 'palmscripts', 'images')
    # Get a list of image filenames from the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    # Build a list of dictionaries for each palm script image
    palm_scripts = []
    for i, filename in enumerate(image_files):
        # Use the filename (without extension) as the unique id.
        script_id = os.path.splitext(filename)[0]
        palm_scripts.append({
            'id': script_id,
            'filename': filename,
            'description': f'Palm Script {i + 1}'
        })
    
    return render_template('palmscripts.html', palm_scripts=palm_scripts)

CLUSTER_DIR = os.path.join(app.static_folder, 'images')
@app.route('/cluster/<cluster_name>')
def show_cluster(cluster_name):
    folder_path = os.path.join(CLUSTER_DIR, cluster_name)
    if not os.path.isdir(folder_path):
        abort(404, description="Cluster not found")
    
    all_images = [img for img in os.listdir(folder_path)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    # Select 12 images if available
    images = random.sample(all_images, min(12, len(all_images)))

    # Ensure 12 images if possible
    while len(images) < 12 and len(all_images) > len(images):
        extra = list(set(all_images) - set(images))
        images.append(random.choice(extra))

    image_urls = [url_for('static', filename=f'images/{cluster_name}/{img}') for img in images]
    
    return render_template('cluster.html', cluster_name=cluster_name, image_urls=image_urls)

# @app.route('/indus')
# def indus():
#     try:
#         cluster_folders = [folder for folder in os.listdir(CLUSTER_DIR)
#                            if os.path.isdir(os.path.join(CLUSTER_DIR, folder))]
#     except FileNotFoundError:
#         cluster_folders = []

#     cluster_folders.sort()
#     clusters_with_preview = []
#     for folder in cluster_folders:
#         folder_path = os.path.join(CLUSTER_DIR, folder)
#         images = [img for img in os.listdir(folder_path)
#                   if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
#         preview_url = None
#         if images:
#             preview_img = random.choice(images)
#             preview_url = url_for('static', filename=f'images/{folder}/{preview_img}')
#         clusters_with_preview.append({'name': folder, 'preview_url': preview_url})
    
#     return render_template('index.html', clusters=clusters_with_preview)
@app.route('/indus')
def indus():
    try:
        cluster_folders = [folder for folder in os.listdir(CLUSTER_DIR)
                           if os.path.isdir(os.path.join(CLUSTER_DIR, folder)) and folder.lower() != "landing"]
    except FileNotFoundError:
        cluster_folders = []

    cluster_folders.sort()
    clusters_with_preview = []
    for folder in cluster_folders:
        folder_path = os.path.join(CLUSTER_DIR, folder)
        images = [img for img in os.listdir(folder_path)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        preview_url = None
        if images:
            preview_img = random.choice(images)
            preview_url = url_for('static', filename=f'images/{folder}/{preview_img}')
        clusters_with_preview.append({'name': folder, 'preview_url': preview_url})
    
    return render_template('indus.html', clusters=clusters_with_preview)

@app.route('/cluster/<cluster_name>/load_more')
def load_more(cluster_name):
    folder_path = os.path.join(CLUSTER_DIR, cluster_name)
    if not os.path.isdir(folder_path):
        abort(404, description="Cluster not found")

    all_images = [img for img in os.listdir(folder_path)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    exclude = request.args.get('exclude', '')
    exclude = exclude.split(',') if exclude else []
    
    available = [img for img in all_images if img not in exclude]
    
    new_images = random.sample(available, min(12, len(available))) if available else []
    
    image_urls = [url_for('static', filename=f'images/{cluster_name}/{img}') for img in new_images]
    
    return jsonify(image_urls=image_urls)

@app.route('/palmscripts/<script_id>')
def palm_script_detail(script_id):
    # Prepare detail for a given palm script.
    detail = {}
    # Use a naming convention for detail images:
    # Full image is assumed to be stored as <script_id>_detail.jpg in palmscripts/images.
    # Segmented image is stored as <script_id>_ocr.jpg in palmscripts/segments.
    detail['image'] = f"{script_id}.jpg"  # e.g., palm1_detail.jpg
    detail['ocr_image'] = f"{script_id}_segment.jpg"   # e.g., palm1_ocr.jpg
    
    # Read the extracted text from the corresponding text file in palmscripts/text.
    text_folder = os.path.join(app.static_folder, 'palmscripts', 'text')
    text_file = os.path.join(text_folder, f"{script_id}.txt")
    extracted_text = ""
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
    detail['extracted_text'] = extracted_text
    
    return render_template('palm_script_detail.html', detail=detail, script_id=script_id)

@app.route('/stoneinscriptions')
def stoneinscriptions():
    # Define the folder path for stone inscription images
    image_folder = os.path.join(app.static_folder, 'stoneinscriptions', 'images')
    # Get a list of image filenames from the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    stone_inscriptions = []
    for i, filename in enumerate(image_files):
        inscription_id = os.path.splitext(filename)[0]
        stone_inscriptions.append({
            'id': inscription_id,
            'filename': filename,
            'description': f'Stone Inscription {i + 1}'
        })
    
    return render_template('stoneinscriptions.html', stone_inscriptions=stone_inscriptions)

@app.route('/stoneinscriptions/<inscription_id>')
def stone_inscription_detail(inscription_id):
    # Prepare detail for a given stone inscription.
    detail = {}
    detail['image'] = f"{inscription_id}.jpeg"  # e.g., stone1_detail.jpg
    detail['ocr_image'] = f"processed_inscription {inscription_id}.png"   # e.g., stone1_ocr.jpg
    
    text_folder = os.path.join(app.static_folder, 'stoneinscriptions', 'text')
    text_file = os.path.join(text_folder, f"{inscription_id}.txt")
    extracted_text = ""
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
    detail['extracted_text'] = extracted_text
    
    return render_template('stone_inscription_detail.html', detail=detail, inscription_id=inscription_id)


@app.route('/documents')
def documents():
    # Define the folder path for palm script images
    image_folder = os.path.join(app.static_folder, 'documents', 'images')
    # Get a list of image filenames from the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    # Build a list of dictionaries for each palm script image
    documents = []
    for i, filename in enumerate(image_files):
        # Use the filename (without extension) as the unique id.
        script_id = os.path.splitext(filename)[0]
        documents.append({
            'id': script_id,
            'filename': filename,
            'description': f'Document {i + 1}'
        })
    
    return render_template('documents.html', documents=documents)
@app.route('/documents/<script_id>')
def documents_detail(script_id):
    # Possible image extensions
    possible_extensions = ['.png', '.jpg', '.jpeg']

    # Function to find the actual file
    def find_file(directory, script_id):
        for ext in possible_extensions:
            file_path = os.path.join(directory, f"{script_id}{ext}")
            if os.path.exists(file_path):
                return f"{script_id}{ext}"  # Return filename with extension
        return None  # If no file is found

    # Paths for image directories
    image_folder = os.path.join(app.static_folder, 'documents', 'images')
    segment_folder = os.path.join(app.static_folder, 'documents', 'segments')

    # Get the correct file names for the images
    image_file = find_file(image_folder, script_id) or "default.png"  # Use a default if missing
    ocr_image_file = find_file(segment_folder, script_id) or "default.png"

    # Read the extracted text
    text_folder = os.path.join(app.static_folder, 'documents', 'text')
    text_file = os.path.join(text_folder, f"{script_id}.txt")
    extracted_text = ""
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()

    if(extracted_text==""):
        image_path=os.path.join(app.static_folder, image_folder, image_file)
        extracted_text=extract_text_from_image(image_path)

    # Prepare detail dictionary

    detail = {
        'image': image_file,
        'ocr_image': ocr_image_file,
        'extracted_text': extracted_text
    }

    return render_template('documents_detail.html', detail=detail, script_id=script_id)

if __name__ == '__main__':
    app.run(debug=True)
