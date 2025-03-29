from flask import Flask, render_template  # type: ignore
from backend_usecase.gemini_pdf_ques_answer.gemini_pdf_ques_answer import gemini_pdf_ques_answer
from backend_usecase.gemini_chatbot.gemini_chatbot import gemini_chatbot
from backend_usecase.gemini_image_text_ext_qa.gemini_image_text_ext_qa import gemini_image_text_ext_qa
import os

app = Flask(__name__, template_folder="templates")  # Ensure Flask knows where to find templates

app.secret_key = os.urandom(24)  # Generates a random 24-byte key

# Register Blueprints
app.register_blueprint(gemini_chatbot, url_prefix="/gemini_chatbot")
app.register_blueprint(gemini_pdf_ques_answer, url_prefix="/gemini_pdf_ques_answer")
app.register_blueprint(gemini_image_text_ext_qa, url_prefix="/gemini_image_text_ext_qa")

@app.template_filter('basename')
def basename_filter(file_path):
    """Extract the base name from a file path."""
    return os.path.basename(file_path)

@app.route('/')
def index():
    return render_template('common/index.html')  # Ensure this file exists in /templates/common/

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use environment variable for port
    app.run(debug=False, host="0.0.0.0", port=port)
