# 🚀 HyperCover

Generate 100 personalized cover letters in 30 seconds using AI.

## 🌟 Features

- **Bulk Generation**: Upload a CSV with job listings and get personalized cover letters for each position
- **AI-Powered Personalization**: Each letter is uniquely crafted using company research and your profile
- **Smart Company Research**: Automatically gathers information about companies from:
  - Company websites
  - LinkedIn profiles
  - News articles
  - Industry databases
- **Multiple Export Formats**: Get your cover letters in both DOCX and PDF formats
- **Real-time Progress**: Watch your cover letters being generated with a live progress bar
- **Template Customization**: Choose from various templates and customize:
  - Tone (formal, casual, confident, humble)
  - Content sections
  - Structure and formatting
- **LinkedIn Integration**: Import your professional profile directly from LinkedIn
- **Enterprise-Grade Security**: 
  - Rate limiting
  - Input sanitization
  - File validation
  - Secure file handling

## 🛠️ Tech Stack

- **Backend**: Python/Flask with async processing
- **AI**: Groq API with Mixtral and LLaMA models
- **Frontend**: Modern UI with Tailwind CSS
- **Caching**: Redis for high performance
- **Document Processing**: Pandoc for PDF generation
- **Monitoring**: Prometheus metrics

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/plawlost/hypercover.git
cd hypercover
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GROQ_API_KEY=your_groq_api_key
NEWS_API_KEY=your_news_api_key  # Optional
GLASSDOOR_API_KEY=your_glassdoor_api_key  # Optional
```

4. Run the application:
```bash
python main.py
```

The app will be available at `http://localhost:5000`

## 📊 Usage

1. **Import Your Profile**
   - Enter your LinkedIn URL or manually input your professional details
   - The app will extract your experience, skills, and achievements

2. **Choose a Template**
   - Select from various pre-built templates
   - Customize tone, structure, and content sections
   - Preview the result in real-time

3. **Upload Job List**
   - Prepare a CSV file with columns: company_name, position, notes (optional)
   - Download the template CSV for the correct format
   - Upload your file

4. **Generate & Download**
   - Click "Generate Cover Letters"
   - Watch the real-time progress
   - Download your personalized cover letters in a ZIP file

## 🔒 Security

HyperCover takes security seriously:
- All user inputs are sanitized
- Files are validated and securely handled
- Rate limiting prevents abuse
- No sensitive data is stored
- SSL encryption in transit

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for their powerful AI API
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tailwind CSS](https://tailwindcss.com/) for the UI components

---
Made with ❤️ by [plawlost](https://github.com/plawlost) 