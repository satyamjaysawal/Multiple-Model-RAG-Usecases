from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    # Sample DataFrame
    data = {"Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]}
    df = pd.DataFrame(data)

    # Convert DataFrame to HTML table
    table_html = df.to_html(classes='table table-striped', index=False)

    return render_template('index.html', table=table_html)

# Required for Vercel
def handler(event, context):
    return app

if __name__ == '__main__':
    app.run(debug=True)
