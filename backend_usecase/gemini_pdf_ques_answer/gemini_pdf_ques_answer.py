import os
import asyncio
from flask import Blueprint, request, render_template, flash, Response, send_file
from werkzeug.utils import secure_filename
import numpy as np
import fitz  # PyMuPDF
import google.generativeai as genai
import easyocr
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
import certifi
from bson.objectid import ObjectId
import datetime
import uuid
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import aiohttp

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL not found in environment variables")

genai.configure(api_key=API_KEY)

# Initialize Blueprint
gemini_pdf_ques_answer = Blueprint('gemini_pdf_ques_answer', __name__, template_folder='../templates')

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
mongo_db = mongo_client.get_database("gemini_chatbot_db")
pdf_collection = mongo_db["pdf_documents"]

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, verify=False)
async_qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, verify=False)
COLLECTION_NAME = "pdf_chunks"

# Initialize EasyOCR reader with lazy loading
ocr_reader = None
def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'])
    return ocr_reader

embedding_model = "models/embedding-001"
generation_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Ensure Qdrant collection exists
def initialize_qdrant_collection():
    try:
        collections = qdrant_client.get_collections().collections
        if COLLECTION_NAME not in [col.name for col in collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
    except Exception as e:
        raise

initialize_qdrant_collection()

async def extract_text_from_pdf(pdf_stream, filename):
    try:
        async with aiohttp.ClientSession() as session:
            with fitz.open(stream=pdf_stream, filetype="pdf") as pdf:
                selectable_text = ""
                for page in pdf:
                    selectable_text += page.get_text()
                if selectable_text.strip():
                    return selectable_text, "non-scanned"
                else:
                    text = await extract_text_with_easyocr(pdf, filename)
                    return text, "scanned"
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {str(e)}")

async def extract_text_with_easyocr(pdf_doc, filename):
    text = ""
    ocr = get_ocr_reader()
    for page_num in range(len(pdf_doc)):
        pix = pdf_doc[page_num].get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        ocr_result = await asyncio.to_thread(ocr.readtext, img_bytes)
        text += " ".join([result[1] for result in ocr_result]) + " "
    return text.strip()

def extract_pdf_metadata(pdf_stream, filename):
    """Extract metadata from a PDF."""
    try:
        with fitz.open(stream=pdf_stream, filetype="pdf") as pdf:
            metadata = pdf.metadata
            page_count = len(pdf)
            file_size = len(pdf_stream) // 1024  # Size in KB
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "title": metadata.get("title", "Untitled") or "Untitled",
            "author": metadata.get("author", "Unknown") or "Unknown",
            "page_count": page_count,
            "file_size_kb": file_size,
            "upload_time": upload_time
        }
    except Exception as e:
        return {
            "title": "Untitled",
            "author": "Unknown",
            "page_count": 0,
            "file_size_kb": 0,
            "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def store_pdf_in_mongo(filename, text, pdf_type, pdf_stream, tags=[]):
    metadata = extract_pdf_metadata(pdf_stream, filename)
    pdf_doc = {
        "filename": filename,
        "text": text,
        "pdf_type": pdf_type,
        "upload_time": datetime.datetime.now().isoformat(),
        "qa_history": [],
        "preview": pdf_stream,
        "tags": tags,
        "metadata": metadata
    }
    result = pdf_collection.insert_one(pdf_doc)
    return str(result.inserted_id)

def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

async def add_chunks_to_qdrant(text_chunks, pdf_id):
    points = []
    for idx, chunk in enumerate(text_chunks):
        try:
            response = await asyncio.to_thread(genai.embed_content, model=embedding_model, content=chunk)
            if 'embedding' not in response:
                raise ValueError("Embedding generation failed")
            embedding = response['embedding']
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": chunk, "pdf_id": pdf_id}
                )
            )
        except Exception as e:
            raise

    await async_qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

async def get_relevant_chunks(query, pdf_ids, top_k=5):
    try:
        response = await asyncio.to_thread(genai.embed_content, model=embedding_model, content=query)
        if 'embedding' not in response:
            raise ValueError("Failed to generate embedding for query")
        query_embedding = response['embedding']
        
        search_result = await async_qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="pdf_id", match=models.MatchAny(any=pdf_ids))]
            )
        )
        contributing_pdf_ids = set(hit.payload["pdf_id"] for hit in search_result)
        metadata = {pdf_id: pdf_collection.find_one({"_id": ObjectId(pdf_id)})["metadata"] for pdf_id in contributing_pdf_ids}
        return [hit.payload["text"] for hit in search_result], metadata
    except Exception as e:
        raise

def delete_pdf(pdf_id):
    try:
        result = pdf_collection.delete_one({"_id": ObjectId(pdf_id)})
        if result.deleted_count == 0:
            raise ValueError("PDF not found in MongoDB")
        
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="pdf_id", match=models.MatchValue(value=pdf_id))]
                )
            )
        )
        return True
    except Exception as e:
        raise

@gemini_pdf_ques_answer.route('/gemini_pdf_ques_answer_generate', methods=['GET', 'POST'])
async def gemini_pdf_ques_answer_generate():
    message, answer, error, metadata = None, None, None, {}
    selected_pdf_ids = request.args.getlist('pdf_id') or []
    search_query = request.args.get('search', '').lower()
    pdf_type_filter = request.args.get('pdf_type', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    page = int(request.args.get('page', 1))
    per_page = 10

    # Build MongoDB query
    query = {}
    if search_query:
        query["$or"] = [
            {"filename": {"$regex": search_query, "$options": "i"}},
            {"tags": {"$regex": search_query, "$options": "i"}}
        ]
    if pdf_type_filter:
        query["pdf_type"] = pdf_type_filter
    if date_from or date_to:
        query["upload_time"] = {}
        if date_from:
            query["upload_time"]["$gte"] = date_from
        if date_to:
            query["upload_time"]["$lte"] = date_to + "T23:59:59"
    
    total_pdfs = pdf_collection.count_documents(query)
    pdfs = list(pdf_collection.find(query).sort("upload_time", -1).skip((page - 1) * per_page).limit(per_page))
    total_pages = (total_pdfs + per_page - 1) // per_page

    if request.method == 'POST':
        if 'upload' in request.form:
            pdf_file = request.files.get('pdf_file')
            tags = request.form.get('tags', '').split(',')
            if not pdf_file or not pdf_file.filename:
                error = "No file selected for upload."
                flash(error, 'error')
            elif not allowed_file(pdf_file.filename):
                error = "Invalid file type. Only PDF files are allowed."
                flash(error, 'error')
            else:
                filename = secure_filename(pdf_file.filename)
                pdf_stream = pdf_file.read()
                try:
                    text, pdf_type = await extract_text_from_pdf(pdf_stream, filename)
                    pdf_id = store_pdf_in_mongo(filename, text, pdf_type, pdf_stream, [tag.strip() for tag in tags if tag.strip()])
                    text_chunks = chunk_text(text)
                    await add_chunks_to_qdrant(text_chunks, pdf_id)
                    message = f"PDF '{filename}' uploaded successfully! Type: {pdf_type}"
                    flash(message, 'success')
                    selected_pdf_ids = [pdf_id]
                except ValueError as ve:
                    error = str(ve)
                    flash(error, 'error')
                except Exception as e:
                    error = f"Failed to process PDF: {str(e)}"
                    flash(error, 'error')

        elif 'query' in request.form:
            user_question = request.form.get('user_question')
            if not selected_pdf_ids:
                error = "Please select at least one PDF first."
                flash(error, 'error')
            elif not user_question:
                error = "Please enter a question."
                flash(error, 'error')
            else:
                try:
                    relevant_chunks, metadata = await get_relevant_chunks(user_question, selected_pdf_ids)
                    if not relevant_chunks:
                        error = "No relevant content found in the selected PDFs."
                        flash(error, 'error')
                    else:
                        context = " ".join(relevant_chunks)
                        prompt = (
                            f"You are an intelligent assistant with expertise in summarizing and answering questions based on provided content. "
                            f"Below is text extracted from one or more documents. Answer the question clearly and concisely in plain text.\n\n"
                            f"Document Content:\n{context}\n\n"
                            f"Question:\n{user_question}"
                        )
                        response = await asyncio.to_thread(generation_model.generate_content, prompt)
                        answer = response.text.strip()
                        for pdf_id in selected_pdf_ids:
                            pdf_collection.update_one(
                                {"_id": ObjectId(pdf_id)},
                                {"$push": {"qa_history": {"question": user_question, "answer": answer, "timestamp": datetime.datetime.now().isoformat()}}}
                            )
                except Exception as e:
                    error = f"Failed to generate answer: {str(e)}"
                    flash(error, 'error')

        elif 'delete' in request.form:
            pdf_ids_to_delete = request.form.getlist('pdf_id')
            if not pdf_ids_to_delete:
                error = "No PDFs selected for deletion."
                flash(error, 'error')
            else:
                try:
                    for pdf_id in pdf_ids_to_delete:
                        delete_pdf(pdf_id)
                        if pdf_id in selected_pdf_ids:
                            selected_pdf_ids.remove(pdf_id)
                    message = f"Deleted {len(pdf_ids_to_delete)} PDF(s) successfully."
                    flash(message, 'success')
                except Exception as e:
                    error = f"Failed to delete PDFs: {str(e)}"
                    flash(error, 'error')

        elif 'export' in request.form:
            pdf_ids = request.form.getlist('pdf_id')
            export_format = request.form.get('export_format', 'txt')
            if not pdf_ids:
                error = "No PDFs selected for export."
                flash(error, 'error')
            else:
                if export_format == 'txt':
                    output = io.StringIO()
                    export_count = 0
                    for pdf_id in pdf_ids:
                        pdf = pdf_collection.find_one({"_id": ObjectId(pdf_id)})
                        if pdf and pdf.get('qa_history'):
                            export_count += 1
                            output.write(f"QA History for {pdf['filename']} (Exported on {datetime.datetime.now().isoformat()[:19]})\n\n")
                            for qa in pdf['qa_history']:
                                output.write(f"Q: {qa['question']}\n")
                                output.write(f"A: {qa['answer']}\n")
                                output.write(f"Timestamp: {qa['timestamp'][:19]}\n\n")
                    if export_count > 0:
                        filename = f"qa_history_bulk_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        return Response(
                            output.getvalue(),
                            mimetype="text/plain",
                            headers={"Content-Disposition": f"attachment; filename={filename}"}
                        )
                    else:
                        error = "No QA history available for selected PDFs."
                        flash(error, 'error')
                elif export_format == 'pdf':
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    export_count = 0
                    for pdf_id in pdf_ids:
                        pdf = pdf_collection.find_one({"_id": ObjectId(pdf_id)})
                        if pdf and pdf.get('qa_history'):
                            export_count += 1
                            story.append(Paragraph(f"QA History for {pdf['filename']} (Exported on {datetime.datetime.now().isoformat()[:19]})", styles['Heading1']))
                            story.append(Spacer(1, 12))
                            for qa in pdf['qa_history']:
                                story.append(Paragraph(f"Q: {qa['question']}", styles['BodyText']))
                                story.append(Paragraph(f"A: {qa['answer']}", styles['BodyText']))
                                story.append(Paragraph(f"Timestamp: {qa['timestamp'][:19]}", styles['BodyText']))
                                story.append(Spacer(1, 12))
                    if export_count > 0:
                        doc.build(story)
                        buffer.seek(0)
                        filename = f"qa_history_bulk_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name=filename)
                    else:
                        error = "No QA history available for selected PDFs."
                        flash(error, 'error')

    selected_pdfs = [pdf_collection.find_one({"_id": ObjectId(pid)}) for pid in selected_pdf_ids if pid]
    return render_template(
        'usecase/gemini_pdf_ques_answer/gemini_pdf_ques_answer.html',
        message=message,
        answer=answer,
        error=error,
        selected_pdf_ids=selected_pdf_ids,
        pdfs=pdfs,
        selected_pdfs=selected_pdfs,
        search_query=search_query,
        pdf_type_filter=pdf_type_filter,
        date_from=date_from,
        date_to=date_to,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        metadata=metadata
    )

@gemini_pdf_ques_answer.route('/preview/<pdf_id>')
def preview_pdf(pdf_id):
    pdf = pdf_collection.find_one({"_id": ObjectId(pdf_id)})
    if pdf and pdf.get('preview'):
        pdf_doc = fitz.open(stream=pdf['preview'], filetype="pdf")
        page = pdf_doc[0]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    return "Preview not available", 404

def allowed_file(filename):
    return filename.lower().endswith('.pdf')