
from flask import Flask, render_template, request, send_file
import companyfinder
from docx import Document
from docx2pdf import convert
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        company = request.form['input_1']
        role = request.form['input_2']
        name = request.form['input_3']
        
        # Get company info
        company_info = companyfinder.getCompany(company)
        
        # Generate cover letter (placeholder text - replace with your generation logic)
        cover_letter = f"Dear Hiring Manager,\n\nI am writing to express my interest in the {role} position at {company}.\n\nBest regards,\n{name}"
        
        # Create Word document
        doc = Document()
        doc.add_paragraph(cover_letter)
        doc.save('temp_cover_letter.docx')
        
        # Convert to PDF
        convert('temp_cover_letter.docx', 'cover_letter.pdf')
        
        return render_template('index.html', paragraph=cover_letter)
    
    return render_template('index.html')

@app.route('/download/<filetype>')
def download(filetype):
    if filetype == 'docx':
        return send_file('temp_cover_letter.docx', as_attachment=True)
    elif filetype == 'pdf':
        return send_file('cover_letter.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
