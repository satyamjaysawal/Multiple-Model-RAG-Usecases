import os
import uuid
from flask import Blueprint, request, render_template, flash, session, redirect, url_for
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
import easyocr
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not all([GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing required environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration
CONFIG = {
    "MAX_FILE_SIZE_MB": 10,
    "ALLOWED_EXTENSIONS": {'png', 'jpg', 'jpeg', 'gif'},
    "OCR_TIMEOUT": 30,
    "EMBEDDING_BATCH_SIZE": 50,
    "MAX_CONVERSATION_HISTORY": 5,
}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

gemini_image_text_ext_qa = Blueprint('gemini_image_text_ext_qa', __name__, template_folder='../templates', static_folder='static')

embedding_model = "models/embedding-001"
generation_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, verify=False)
collection_name = "image_text_embeddings"
dimension = 768

# Recreate Qdrant collection on each run (no persistence)
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
)

class OCRManager:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_path, timeout=CONFIG["OCR_TIMEOUT"]):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.reader.readtext, image_path, detail=1)
                ocr_result = future.result(timeout=timeout)
                text = "\n".join([res[1] for res in ocr_result])
                return text
        except Exception:
            return ""

ocr_manager = OCRManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG["ALLOWED_EXTENSIONS"]

def chunk_text(text, chunk_size=512):
    if not text.strip():
        return []
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - 50)]
    return chunks

def add_chunks_to_qdrant(text_chunks, full_text, batch_size=CONFIG["EMBEDDING_BATCH_SIZE"]):
    if not text_chunks:
        return
    points = []
    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            response = genai.embed_content(model=embedding_model, content=batch)
            embeddings = response['embedding']
            for j, embedding in enumerate(embeddings):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"chunk": batch[j], "full_text": full_text}
                ))
        qdrant_client.upsert(collection_name=collection_name, points=points)
    except Exception:
        pass

def get_relevant_chunks_and_full_text(query, top_k=5):
    try:
        response = genai.embed_content(model=embedding_model, content=query)
        query_embedding = response['embedding']
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        chunks = [hit.payload["chunk"] for hit in search_result]
        full_text = search_result[0].payload["full_text"] if search_result else ""
        return chunks, full_text
    except Exception:
        return [], ""

def clean_response(text):
    text = re.sub(r'\*\*|\*|_', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def get_conversation_history():
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return session['conversation_history']

def add_to_conversation_history(question, answer):
    history = get_conversation_history()
    history.append({"question": question, "answer": answer})
    if len(history) > CONFIG["MAX_CONVERSATION_HISTORY"]:
        history = history[-CONFIG["MAX_CONVERSATION_HISTORY"]:]
    session['conversation_history'] = history
    session.modified = True

def format_conversation_history():
    history = get_conversation_history()
    if not history:
        return ""

    formatted = "**Previous Conversation:**\n\n"
    for i, qa in enumerate(history):
        formatted += f"Q{i+1}: {qa['question']}\n"
        formatted += f"A{i+1}: {qa['answer']}\n\n"

    return formatted

def clear_qdrant_data():
    try:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
        return True
    except Exception:
        return False

@gemini_image_text_ext_qa.route('/gemini_image_text_ext_qa_delete_history', methods=['POST'])
def delete_history():
    session['conversation_history'] = []
    session['extracted_full_text'] = None
    session.modified = True

    success = clear_qdrant_data()

    if success:
        flash("History and stored data cleared successfully.", 'success')
    else:
        flash("Failed to clear vector data. Session data cleared.", 'warning')

    return redirect(url_for('gemini_image_text_ext_qa.gemini_image_text_ext_qa_handler'))

@gemini_image_text_ext_qa.route('/gemini_image_text_ext_qa_generate', methods=['GET', 'POST'])
def gemini_image_text_ext_qa_handler():
    answer = None
    uploaded_filename = None
    extracted_text = None
    conversation_history = get_conversation_history()

    if request.method == 'POST':
        if 'image_files' in request.files:
            session['conversation_history'] = []
            session['extracted_full_text'] = None

            image_file = request.files['image_files']
            if not image_file or not image_file.filename:
                flash("No file selected.", 'error')
                return render_template('usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
                                       CONFIG=CONFIG,
                                       conversation_history=conversation_history)

            filename = secure_filename(image_file.filename)
            if not allowed_file(filename):
                flash("Invalid file type. Allowed: png, jpg, jpeg, gif.", 'error')
                return render_template('usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
                                       CONFIG=CONFIG,
                                       conversation_history=conversation_history)

            image_path = os.path.join("/tmp", filename)
            image_file.save(image_path)
            if os.path.getsize(image_path) > CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024:
                os.remove(image_path)
                flash("File too large.", 'error')
                return render_template('usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
                                       CONFIG=CONFIG,
                                       conversation_history=conversation_history)

            text = ocr_manager.extract_text(image_path)
            if not text:
                os.remove(image_path)
                flash("No text extracted from the image.", 'warning')
                return render_template('usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
                                       CONFIG=CONFIG,
                                       conversation_history=conversation_history)

            session['extracted_full_text'] = text

            text_chunks = chunk_text(text)
            add_chunks_to_qdrant(text_chunks, text)
            uploaded_filename = filename
            extracted_text = text

            os.remove(image_path)
            flash(f"Image '{filename}' processed successfully.", 'success')

        elif 'query' in request.form:
            user_question = request.form.get('user_question')
            if not user_question:
                flash("Please enter a question.", 'error')
            else:
                full_text = session.get('extracted_full_text', '')

                if not full_text:
                    _, full_text = get_relevant_chunks_and_full_text(user_question)

                if not full_text:
                    answer = "No relevant content found. Please upload an image first."
                    flash("No relevant content found.", 'warning')
                else:
                    conversation_context = format_conversation_history()

                    prompt = (
                        f"You are an expert assistant skilled at interpreting text extracted from images. "
                        f"Your task is to provide a clear, concise, and accurate answer to the user's question based solely on the text below. "
                        f"Avoid speculation or external knowledge; focus only on the provided content. "
                        f"If the answer is not fully clear from the text, indicate uncertainty and explain why.\n\n"
                        f"**Extracted Text from Images:**\n{full_text}\n\n"
                        f"{conversation_context}"
                        f"**Current Question:**\n{user_question}\n\n"
                        f"Provide your answer below:"
                    )

                    try:
                        response = generation_model.generate_content(prompt)
                        answer = clean_response(response.text.strip())

                        add_to_conversation_history(user_question, answer)
                        conversation_history = get_conversation_history()

                        flash("Answer generated successfully.", 'success')
                    except Exception:
                        answer = "Could not generate an answer based on the extracted text."
                        flash("Error generating answer.", 'error')

    return render_template(
        'usecase/gemini_image_text_ext_qa/gemini_image_text_ext_qa.html',
        answer=answer,
        uploaded_filename=uploaded_filename,
        extracted_text=extracted_text,
        conversation_history=conversation_history,
        CONFIG=CONFIG
    )
