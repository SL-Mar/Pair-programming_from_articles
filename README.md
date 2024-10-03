# 🛠️ Algorithms Generation from Quantitative Finance Articles to QuantConnect  🚀 

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-brightgreen.svg)

Coder is a Python-based tool designed to convert quantitative finance research articles into actionable trading algorithms compatible with [QuantConnect](https://www.quantconnect.com/). By utilizing Natural Language Processing (NLP) and OpenAI's language models, this tool automates the extraction of trading strategies and risk management techniques from PDF articles, summarizes the findings, and generates ready-to-use QuantConnect Python code with proper syntax highlighting.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation](#step-by-step-installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [GUI Interaction](#gui-interaction)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Refinement Attempts](#refinement-attempts)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction

This script streamlines the process of transforming quantitative finance research into executable trading algorithms. By automating text extraction, preprocessing, and analysis from PDF documents, the tool facilitates the efficient development of trading strategies within the QuantConnect environment. This automation reduces manual effort, minimizes errors, and accelerates the implementation of complex financial models.

## Features

- **PDF Text Extraction**: Utilizes `pdfplumber` to accurately extract text from complex PDF structures.
  
- **Text Preprocessing**: Cleans extracted text by removing URLs, headers, footers, and irrelevant content.
  
- **Heading Detection**: Identifies section headings using SpaCy's NLP capabilities for structured content organization.
  
- **Keyword Analysis**: Categorizes sentences into trading signals and risk management based on predefined keywords.
  
- **Article Summarization**: Generates concise summaries of extracted strategies and risk management techniques using OpenAI's GPT-4.
  
- **QuantConnect Code Generation**: Automatically generates QuantConnect-compatible Python algorithms based on the extracted data.
  
- **GUI Display**: Presents the article summary and generated code in separate Tkinter windows with syntax highlighting powered by Pygments.
  
- **Error Handling & Validation**: Validates generated code for syntax errors and refines it automatically if necessary.

## Installation

### Prerequisites

- **Python 3.8 or Higher**: Ensure Python is installed on your system. [Download Python](https://www.python.org/downloads/)
  
- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys) to enable AI-driven functionalities.

### Step-by-Step Installation

1. **Clone the Repository**
```bash
git clone https://github.com/SL-Mar/Article_to_Code.git
cd Article_to_Code
```
2. **Create a Virtual Environment**
   
It's recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
```

3. **Activate the Virtual Environment**

For macOS/Linux:
```bash
source venv/bin/activate
```

For Windows:
```bash
venv\Scripts\activate
```
    
4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

5. **Download spaCy Model**
```bash
python -m spacy download en_core_web_sm
```

6. **Configure Environment Variables**

Create a .env file in the root directory and add your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "LOG_LEVEL=INFO" >> .env
```

## Usage

### Running the application

Execute the main script with the path to your PDF article as an argument:
```bash
python article_to_code.py path/to/your/article.pdf
```

### GUI Interaction

1. Load PDF: The application processes the specified PDF, extracting and analyzing its content.
2. View Summary: A concise summary of the trading strategy and risk management sections is displayed.
3. Review Generated Code: The corresponding QuantConnect Python code is showcased with syntax highlighting.
4. Copy and Save: Use the provided buttons to copy the summary or code to your clipboard or save the code to a file.

## Configuration

### Environment Variables:

+ OPENAI_API_KEY: Your OpenAI API key for accessing GPT-4 functionalities.
+ LOG_LEVEL: Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR). Default is INFO.

### Refinement Attempts:

By default, the application attempts to refine the generated code up to 3 times if syntax errors are detected. You can adjust this by modifying the max_refine_attempts parameter in the ArticleProcessor class.

## Dependencies

The project relies on several external libraries. All dependencies are listed in the requirements.txt file.

1. pdfplumber
2. spaCy
3. openai
4. python-dotenv
5. tkinter
6. pygments

## Contributing

Contributions are welcome! Please follow these steps to contribute : 

1. Fork the Repository
2. Create a New Branch
```bash
git checkout -b feature/YourFeatureName
```
3. Commit Your Changes
```bash
git commit -m "Add feature: YourFeatureName"
```
4. Push to the Branch   
```bash
git push origin feature/YourFeatureName
```
5. Open a Pull Request

Provide a clear description of the changes and the problem they address.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software. See the LICENSE file for more details.

## Acknowledgements

+ QuantConnect as the backtest and trading platform
+ pdfplumber for efficient PDF text extraction.
+ spaCy for powerful natural language processing capabilities.
+ OpenAI for providing the GPT-4 model used in AI-driven summarization and code generation.
+ Tkinter for the graphical user interface framework.
+ Pygments for syntax highlighting in the GUI.

## Contact
- **LinkedIn**: S.M. LAIGNEL [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/smrlaignel/)











