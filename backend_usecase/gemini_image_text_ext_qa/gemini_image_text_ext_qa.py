import os
import uuid
import time
import threading
import json
from flask import Blueprint, request, render_template, session, flash, url_for, jsonify, current_app
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
import easyocr
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from datetime import datetime, timedelta
import re
import shutil

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not all([GOOGLE_API_KEY, MONGO_URI, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing required environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration
CONFIG = {
    "MAX_FILE_SIZE_MB": 10,
    "ALLOWED_EXTENSIONS": {'png', 'jpg', 'jpeg', 'gif'},
    "OCR_TIMEOUT": 30,
    "EMBEDDING_BATCH_SIZE": 50,
    "TRASH_EXPIRY_SECONDS": 30,
    "MAX_QUERY_HISTORY": 5,
    "SESSION_EXPIRY_MINUTES": 30,
    "PREVIEW_SIZE": (100, 100),
}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

gemini_image_text_ext_qa = Blueprint('gemini_image_text_ext_qa', __name__, template_folder='../templates', static_folder='static')

embedding_model = "models/embedding-001"
generation_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Initialize MongoDB and Qdrant clients
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['gemini_image_text_ext_qa']
files_collection = db['files']
query_history_collection = db['query_history']
trash_collection = db['trash']

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    verify=False  # Temporary workaround for SSL issue; replace with CA certificate path later
)
collection_name = "image_text_embeddings"
dimension = 768

# Ensure Qdrant collection exists
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

class OCRManager:
    def __init__(self):
        self.reader = None
        self.lock = threading.Lock()

    def get_reader(self):
        with self.lock:
            if self.reader is None or not self.is_valid():
                self.reader = easyocr.Reader(['en'], gpu=False)
            return self.reader

    def is_valid(self):
        return hasattr(self.reader, 'readtext')

    def cleanup(self):
        with self.lock:
            if self.reader is not None:
                self.reader = None

ocr_manager = OCRManager()

UPLOAD_FOLDER = os.path.join(gemini_image_text_ext_qa.static_folder, "uploads")
TRASH_FOLDER = os.path.join(gemini_image_text_ext_qa.static_folder, "trash")
PREVIEW_FOLDER = os.path.join(gemini_image_text_ext_qa.static_folder, "previews")

for folder in [UPLOAD_FOLDER, TRASH_FOLDER, PREVIEW_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG["ALLOWED_EXTENSIONS"]

def initialize_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["last_activity"] = time.time()
    if "query_history" not in session:
        session["query_history"] = []
    if "deleted_files" not in session:
        session["deleted_files"] = []
    if time.time() - session.get("last_activity", 0) > CONFIG["SESSION_EXPIRY_MINUTES"] * 60:
        session.clear()
        session["session_id"] = str(uuid.uuid4())
        session["last_activity"] = time.time()
    session["last_activity"] = time.time()

def generate_preview(image_path):
    filename = os.path.basename(image_path)
    preview_path = os.path.join(PREVIEW_FOLDER, f"thumb_{filename}")
    if not os.path.exists(preview_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail(CONFIG["PREVIEW_SIZE"])
                img.save(preview_path)
            with current_app.app_context():
                return url_for('gemini_image_text_ext_qa.static', filename=f'previews/thumb_{filename}')
        except Exception as e:
            with current_app.app_context():
                return url_for('gemini_image_text_ext_qa.static', filename=f'uploads/{filename}')
    with current_app.app_context():
        return url_for('gemini_image_text_ext_qa.static', filename=f'previews/thumb_{filename}')

def extract_text_from_image(image_path, timeout=CONFIG["OCR_TIMEOUT"]):
    def _extract():
        ocr = ocr_manager.get_reader()
        ocr_result = ocr.readtext(image_path, detail=1)
        text = "\n".join([res[1] for res in ocr_result])
        confidence = [res[2] for res in ocr_result]
        return text, confidence

    for attempt in range(3):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_extract)
                text, confidence = future.result(timeout=timeout)
                return text, confidence
        except Exception as e:
            if attempt == 2:
                return "", []
            time.sleep(1)

def extract_image_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        file_size = os.path.getsize(image_path) // 1024
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        preview_url = generate_preview(image_path)
        return {
            "dimensions": f"{width}x{height}",
            "file_size_kb": file_size,
            "upload_time": upload_time,
            "preview_url": preview_url
        }
    except Exception as e:
        return {"dimensions": "Unknown", "file_size_kb": 0, "upload_time": "Unknown", "preview_url": ""}

def chunk_text(text, chunk_size=512):
    if not text.strip():
        return []
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - 50)]

def add_chunks_to_qdrant(filename, text_chunks, batch_size=CONFIG["EMBEDDING_BATCH_SIZE"]):
    if not text_chunks:
        return
    points = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            response = genai.embed_content(model=embedding_model, content=batch)
            embeddings = response['embedding']
            for j, embedding in enumerate(embeddings):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"filename": filename, "chunk": batch[j]}
                ))
        except Exception as e:
            return
    qdrant_client.upsert(collection_name=collection_name, points=points)

def get_relevant_chunks(query, top_k=5):
    try:
        response = genai.embed_content(model=embedding_model, content=query)
        query_embedding = response['embedding']
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        relevant_chunks = [hit.payload["chunk"] for hit in search_result]
        contributing_files = set(hit.payload["filename"] for hit in search_result)
        metadata = {
            filename: files_collection.find_one({"filename": filename}, {"metadata": 1})["metadata"]
            for filename in contributing_files if files_collection.find_one({"filename": filename})
        }
        distances = [hit.score for hit in search_result]
        return relevant_chunks, metadata, distances
    except Exception as e:
        return [], {}, []

def move_to_trash(image_path):
    filename = os.path.basename(image_path)
    trash_path = os.path.join(TRASH_FOLDER, f"{int(time.time())}_{filename}")
    shutil.move(image_path, trash_path)
    return trash_path

def delete_image(image_path, permanent=False):
    filename = os.path.basename(image_path)
    file_doc = files_collection.find_one({"path": image_path})
    if not file_doc:
        return False
    try:
        if permanent:
            if os.path.exists(image_path):
                os.remove(image_path)
                preview_path = os.path.join(PREVIEW_FOLDER, f"thumb_{filename}")
                if os.path.exists(preview_path):
                    os.remove(preview_path)
            files_collection.delete_one({"path": image_path})
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector={"filter": {"must": [{"key": "filename", "match": {"value": filename}}]}}
            )
        else:
            trash_path = move_to_trash(image_path)
            trash_doc = {
                "path": trash_path,
                "original_path": image_path,
                "timestamp": time.time(),
                "original_filename": filename
            }
            trash_collection.insert_one(trash_doc)
            files_collection.delete_one({"path": image_path})
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector={"filter": {"must": [{"key": "filename", "match": {"value": filename}}]}}
            )
            session["deleted_files"].append(trash_doc)
        return True
    except PyMongoError as e:
        return False
    except Exception as e:
        return False

def delete_selected_images(image_paths, permanent=False):
    success = True
    for image_path in image_paths:
        if not delete_image(image_path, permanent):
            success = False
    return success

def undo_deletion():
    if "deleted_files" not in session or not session["deleted_files"]:
        return False
    deleted_file = session["deleted_files"].pop()
    trash_doc = trash_collection.find_one({"path": deleted_file["path"]})
    if trash_doc and time.time() - deleted_file["timestamp"] <= CONFIG["TRASH_EXPIRY_SECONDS"]:
        original_path = deleted_file["original_path"]
        filename = deleted_file["original_filename"]
        shutil.move(deleted_file["path"], original_path)
        text, confidence = extract_text_from_image(original_path)
        metadata = extract_image_metadata(original_path)
        metadata["ocr_confidence"] = f"{sum(confidence) / len(confidence):.2f}%" if confidence else "N/A"
        files_collection.insert_one({"path": original_path, "filename": filename, "text": text, "metadata": metadata})
        text_chunks = chunk_text(text)
        if text_chunks:
            add_chunks_to_qdrant(filename, text_chunks)
        trash_collection.delete_one({"path": deleted_file["path"]})
        return True
    return False

def clean_response(text):
    text = re.sub(r'\*\*|\*|_', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

@gemini_image_text_ext_qa.route('/gemini_image_text_ext_qa_generate', methods=['GET', 'POST'])
def gemini_image_text_ext_qa_handler():
    initialize_session()
    upload_time = session.get('upload_time', None)
    uploaded_files = [doc["path"] for doc in files_collection.find()]
    file_data = {doc["filename"]: {"text": doc["text"], "metadata": doc["metadata"]} 
                 for doc in files_collection.find()}
    query_history = session["query_history"]

    if "deleted_files" in session:
        session["deleted_files"] = [f for f in session["deleted_files"] if time.time() - f["timestamp"] <= CONFIG["TRASH_EXPIRY_SECONDS"]]

    if request.method == 'POST':
        if 'image_files' in request.files:
            start_time = time.time()
            flash("Image upload processing started...", 'info')
            image_files = request.files.getlist('image_files')
            total_files = len([f for f in image_files if f and f.filename])
            if total_files == 0:
                flash("No files selected for upload.", 'warning')
            else:
                processed_files = 0
                new_files_uploaded = False
                already_uploaded_images = []
                failed_uploads = []

                def process_image(image_file):
                    if not image_file or not image_file.filename:
                        return None, None
                    filename = secure_filename(image_file.filename)
                    if files_collection.find_one({"filename": filename}):
                        return filename, "already_uploaded"
                    if not allowed_file(filename):
                        return filename, "invalid_type"
                    image_path = os.path.join(UPLOAD_FOLDER, filename)
                    image_file.save(image_path)
                    if os.path.getsize(image_path) > CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024:
                        os.remove(image_path)
                        return filename, "too_large"
                    text, confidence = extract_text_from_image(image_path)
                    metadata = extract_image_metadata(image_path)
                    metadata["ocr_confidence"] = f"{sum(confidence) / len(confidence):.2f}%" if confidence else "N/A"
                    return filename, {"path": image_path, "text": text, "metadata": metadata}

                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(process_image, image_files))

                for filename, result in results:
                    if result == "already_uploaded":
                        already_uploaded_images.append(filename)
                    elif result in ["invalid_type", "too_large"]:
                        failed_uploads.append(filename)
                    elif result:
                        files_collection.insert_one({"path": result["path"], "filename": filename, "text": result["text"], "metadata": result["metadata"]})
                        text_chunks = chunk_text(result["text"])
                        if text_chunks:
                            add_chunks_to_qdrant(filename, text_chunks)
                            new_files_uploaded = True
                        processed_files += 1

                upload_time = time.time() - start_time
                session['upload_time'] = upload_time
                message_parts = []
                if new_files_uploaded:
                    message_parts.append(f"Processed {processed_files}/{total_files} images in {upload_time:.2f} seconds!")
                if already_uploaded_images:
                    message_parts.append(f"Skipped: {', '.join(already_uploaded_images)}.")
                if failed_uploads:
                    message_parts.append(f"Failed: {', '.join(failed_uploads)}.")
                flash("\n".join(message_parts) or "No valid images uploaded.", 'success' if new_files_uploaded else 'info')

        elif 'query' in request.form:
            user_question = request.form.get('user_question')
            if not uploaded_files:
                flash("Please upload at least one image first.", 'error')
            elif not user_question:
                flash("Please enter a question.", 'error')
            else:
                flash("Query processing started...", 'info')
                start_time = time.time()
                query_cache = next((q for q in query_history if q["query"] == user_question), None)
                if query_cache:
                    answer, confidence, metadata = query_cache["answer"], query_cache["confidence"], query_cache["metadata"]
                    query_time = 0
                    flash("Answer retrieved from cache.", 'info')
                else:
                    relevant_chunks, metadata, distances = get_relevant_chunks(user_question)
                    if not relevant_chunks:
                        flash("No relevant content found in the extracted text.", 'warning')
                        answer, confidence = "No relevant content found in the provided images.", 0
                    else:
                        context = " ".join(relevant_chunks)
                        prompt = (
                            f"You are an expert assistant skilled at interpreting text extracted from images. "
                            f"Your task is to provide a clear, concise, and accurate answer to the user's question based solely on the text below. "
                            f"Avoid speculation or external knowledge; focus only on the provided content. "
                            f"If the answer is not fully clear from the text, indicate uncertainty and explain why. "
                            f"Format your response in plain text, avoiding markdown unless explicitly requested.\n\n"
                            f"**Extracted Text from Images:**\n{context}\n\n"
                            f"**User Question:**\n{user_question}\n\n"
                            f"Provide your answer below:"
                        )
                        try:
                            response = generation_model.generate_content(prompt)
                            answer = clean_response(response.text.strip())
                            avg_similarity = sum(distances) / len(distances) if distances else 0
                            confidence = min(1, max(0, avg_similarity))
                            if not answer or len(answer) < 10:
                                raise ValueError("Response too short or empty")
                        except Exception as e:
                            answer = f"Could not generate a precise answer based on the extracted text. The available content may not fully address: '{user_question}'."
                            confidence = 0.5
                        session["query_history"].insert(0, {
                            "query": user_question,
                            "answer": answer,
                            "confidence": confidence,
                            "metadata": metadata
                        })
                        session["query_history"] = session["query_history"][:CONFIG["MAX_QUERY_HISTORY"]]
                    query_time = time.time() - start_time
                    flash(f"Answer generated in {query_time:.2f} seconds.", 'success')
                return render_template(
                    'usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
                    answer=answer,
                    upload_time=upload_time,
                    query_time=query_time,
                    uploaded_files=uploaded_files,
                    metadata=metadata,
                    query_history=query_history,
                    confidence=confidence,
                    CONFIG=CONFIG,
                    file_data=file_data
                )

        elif 'clear' in request.form:
            files_collection.delete_many({})
            query_history_collection.delete_many({})
            trash_collection.delete_many({})
            qdrant_client.delete_collection(collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            for folder in [UPLOAD_FOLDER, TRASH_FOLDER, PREVIEW_FOLDER]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
            session.pop('upload_time', None)
            session.pop('query_history', None)
            session.pop('deleted_files', None)
            flash("All uploaded files and data cleared.", 'success')

        elif 'delete_image' in request.form:
            image_path = request.form.get('delete_image')
            if delete_image(image_path):
                flash(f"Image '{os.path.basename(image_path)}' moved to trash. Undo available for {CONFIG['TRASH_EXPIRY_SECONDS']} seconds.", 'success')
            else:
                flash(f"Failed to delete image '{os.path.basename(image_path)}'.", 'error')

        elif 'delete_selected' in request.form:
            image_paths = request.form.getlist('delete_images')
            if not image_paths:
                flash("No images selected for deletion.", 'warning')
            elif delete_selected_images(image_paths):
                flash(f"Moved {len(image_paths)} selected images to trash. Undo available for {CONFIG['TRASH_EXPIRY_SECONDS']} seconds.", 'success')
            else:
                flash("Failed to delete some or all selected images.", 'error')

        elif 'undo_delete' in request.form:
            if undo_deletion():
                flash("Last deletion undone successfully.", 'success')
            else:
                flash("No deletion available to undo or time expired.", 'warning')

    return render_template(
        'usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
        upload_time=upload_time,
        uploaded_files=uploaded_files,
        query_history=query_history,
        CONFIG=CONFIG,
        file_data=file_data
    )