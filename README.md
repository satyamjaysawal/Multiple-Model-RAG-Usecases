


Great! If you’ve **cloned a Flask project from GitHub** and want to **run it in VS Code**, follow these steps:

---

## ✅ Steps to Run a Cloned Flask App in VS Code

### **1. Clone the Repository**

In the terminal:

```bash
git clone https://github.com/username/project-name.git
cd project-name
```

> Replace the URL with your actual GitHub repo link.

---

### **2. Open the Project in VS Code**

In terminal:

```bash
code .
```

Or launch VS Code and open the folder manually.

---

### **3. Create and Activate a Virtual Environment**

If the repo doesn’t already include a virtual environment:

```bash
# Create
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

### **4. Install Dependencies**

If there is a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### **5. Set the FLASK\_APP Environment Variable**

Depending on the entry file (often `app.py`, `run.py`, or `wsgi.py`):

```bash
# Windows:
set FLASK_APP=app.py
# macOS/Linux:
export FLASK_APP=app.py
```

If the app folder structure is more complex (like `project_name/__init__.py`), set:

```bash
set FLASK_APP=project_name
# or
export FLASK_APP=project_name
```

---

### **6. Run the Flask App**

```bash
flask run
```

You should see output like:

```bash
 * Running on http://127.0.0.1:5000/
```

---

### **7. Open in Browser**

Visit:
📍 `http://127.0.0.1:5000/`



****

![image](https://github.com/user-attachments/assets/6fc7bb57-86c8-4270-a463-cbc4370c1a1d)

![image](https://github.com/user-attachments/assets/cb5e60c3-075b-436b-9291-4770ed713be0)

![image](https://github.com/user-attachments/assets/3a16efdc-84a3-46a1-b80d-ffd398be995e)

![image](https://github.com/user-attachments/assets/95acef7e-ef3a-4f52-8191-aebc06e253e4)


![image](https://github.com/user-attachments/assets/fd4fd275-b780-4e77-a3f2-6a0ae3b7de39)


![image](https://github.com/user-attachments/assets/4796ef69-36b5-484b-a286-2398328adafb)

![image](https://github.com/user-attachments/assets/44358798-6192-403f-80e9-f137a1f697a8)













