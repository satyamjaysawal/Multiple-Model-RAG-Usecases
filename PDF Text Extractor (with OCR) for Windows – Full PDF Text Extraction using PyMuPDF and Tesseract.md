

---

## 📄 `README.md` (For Local Windows Only – with Full Code)

````md
# 📄 PDF Text Extractor (with OCR) – Local Windows System Only

This Python project extracts complete text from any PDF, including:
- ✅ Normal (selectable) text
- ✅ Text inside scanned images using OCR

It is designed to run specifically on **Windows systems only** with proper paths configured for **Tesseract OCR** and **Poppler**.

---

## 🎯 Purpose and Use Cases

- 📚 Extract full text from academic or scanned PDF documents
- 🔍 Process government PDFs where text is image-based (scanned files)
- 📑 Digitize old printed documents or archives
- 💬 Use in NLP preprocessing pipelines for raw textual input

---

## 💻 System Requirement

- ✅ Works only on **Windows**
- Python 3.x installed
- Tesseract OCR installed at: `C:\Program Files\Tesseract-OCR`
- Poppler installed at: `C:\poppler\Library\bin`

---

## 🛠️ Dependencies

Install required libraries:

```bash
pip install pymupdf pdf2image pytesseract
````

Also install:

* [Tesseract for Windows (UB Mannheim build)](https://github.com/UB-Mannheim/tesseract/wiki)
* [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)

Make sure both are extracted and their **paths are correct** (see below in code).

---

## 📜 Full Code: `extract_text.py`

```python
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

# Set path to Tesseract (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# PDF and Poppler paths
pdf_path = "sample.pdf"
poppler_path = r"C:\poppler\Library\bin"  # Make sure this contains pdfinfo.exe

# Step 1: Extract selectable text
doc = fitz.open(pdf_path)
normal_text = ""
for page in doc:
    normal_text += page.get_text()
doc.close()

# Step 2: OCR for scanned image-based text
images = convert_from_path(pdf_path, poppler_path=poppler_path)
ocr_text = ""
for img in images:
    ocr_text += pytesseract.image_to_string(img, lang='eng')  # Use 'hin' for Hindi

# Step 3: Combine all text
complete_text = normal_text + "\n" + ocr_text

# Step 4: Save output
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(complete_text)

print("✅ Text extracted and saved to 'extracted_text.txt'")
```

---

## 📁 Folder Structure

```
project-folder/
│
├── sample.pdf               # Your input PDF file
├── extract_text.py          # Main script
├── extracted_text.txt       # Output text
└── README.md                # Project info
```

---

## ⚠️ Notes

* This project is tested on **Windows 10/11 only**
* For other platforms (Linux/macOS), Poppler and Tesseract need separate setup

---

## 📌 Output

After running the script, you’ll get a file called:

```
extracted_text.txt
```

Which contains **all the text extracted** from the PDF, including OCR-scanned content.

---

## ❌ Limitations

* Doesn’t work without installing Poppler & Tesseract
* Designed only for **local usage**, not web-based
* Doesn’t preserve formatting/tables (plain text only)

---

## 📜 License

MIT License. Free to use and modify.

---

## 🤝 Contributions

Not open for contribution – personal/local tool. But feel free to customize!

```

---

## ✅ Next Step

Would you like me to:

- Generate this `README.md` file as a **downloadable Markdown file**?
- Package it as a zip with the code and sample file?
- Convert this to a `.docx` or `.pdf` documentation file?

Let me know and I’ll prepare it.
```
