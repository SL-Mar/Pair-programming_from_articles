QuantConnect Algorithm Generator from Quantitative Finance Articles

Table of Contents

    Introduction
    Features
    Prerequisites
    Installation
        1. Clone the Repository
        2. Install Python
        3. Install Dependencies
        4. Download SpaCy Model
        5. Setup Environment Variables
    Usage
    Troubleshooting
    Contributing
    License

Introduction

The QuantConnect Algorithm Generator is a Python-based tool designed to streamline the process of converting quantitative finance research articles into actionable trading algorithms compatible with QuantConnect. By leveraging Natural Language Processing (NLP) and OpenAI's powerful language models, this tool automates the extraction of trading strategies and risk management techniques from PDF articles, summarizes the findings, and generates ready-to-use QuantConnect Python code with proper syntax highlighting.
Features

    PDF Text Extraction: Accurately extracts text from complex PDF structures using pdfplumber.
    Text Preprocessing: Cleans extracted text by removing URLs, headers, footers, and irrelevant content.
    Heading Detection: Identifies section headings using SpaCy's NLP capabilities.
    Keyword Analysis: Categorizes sentences into trading signals and risk management based on predefined keywords.
    Article Summarization: Generates concise summaries of extracted strategies and risk management techniques using OpenAI's GPT-4.
    QuantConnect Code Generation: Automatically generates QuantConnect-compatible Python algorithms based on the extracted data.
    GUI Display: Presents the article summary and generated code in separate, user-friendly Tkinter windows with syntax highlighting powered by Pygments.
    Error Handling & Validation: Validates generated code for syntax errors and refines it if necessary.

Prerequisites

Before setting up the project, ensure you have the following:

    Operating System: Windows, macOS, or Linux.
    Python Version: Python 3.6 or higher.
    OpenAI API Key: Access to OpenAI's API for generating summaries and code.
    Tkinter: Standard GUI library for Python (usually included by default).

Installation

Follow the steps below to set up the project on your local machine.
1. Clone the Repository



git clone https://github.com/SL-Mar/Article_to_Code.git
cd Article_to_Code

2. Install Python

Ensure that Python 3.6 or higher is installed on your system.

    Windows & macOS:
        Download the latest Python installer from the official website.
        Run the installer and follow the on-screen instructions.
        Important: During installation, check the box "Add Python to PATH".

    Linux:

        Python is usually pre-installed. To check, run:

        bash

python3 --version

If not installed, use your distribution's package manager. For example, on Ubuntu:

bash

        sudo apt-get update
        sudo apt-get install python3 python3-pip

3. Install Dependencies

Note: tkinter is part of Python's standard library and typically doesn't require separate installation. However, if it's missing, refer to the Troubleshooting section.

    Create a Virtual Environment (Optional but Recommended):

    bash

python -m venv env

Activate the Virtual Environment:

    Windows:

    bash

env\Scripts\activate

macOS/Linux:

bash

    source env/bin/activate

Install Required Python Packages:

bash

pip install -r requirements.txt

If requirements.txt is not provided, install the packages manually:

bash

    pip install pygments pdfplumber spacy openai python-dotenv

4. Download SpaCy Model

The project utilizes SpaCy's English model for NLP tasks.

bash

python -m spacy download en_core_web_sm

5. Setup Environment Variables

Create a .env file in the project root directory to securely store your OpenAI API key.

    Create .env File:

    bash

touch .env

Add Your OpenAI API Key:

Open the .env file in a text editor and add:

env

    OPENAI_API_KEY=your_openai_api_key_here

    Replace your_openai_api_key_here with your actual OpenAI API key.

Usage

    Prepare Your PDF Article:

    Ensure your quantitative finance article is in PDF format. Place it in the project directory or note its path.

    Run the Script:

    bash

    python your_script_name.py

    Replace your_script_name.py with the actual name of your Python script, e.g., main.py.

    Interact with the GUI:
        Summary Window: Displays a concise summary of the article's trading strategies and risk management techniques.
        Code Window: Shows the generated QuantConnect Python algorithm with syntax highlighting for easy reading and verification.

    Review and Utilize the Generated Code:
        Copy the code from the GUI and integrate it into your QuantConnect projects.
        Optionally, use the QuantConnect Lean CLI to validate and test the generated algorithms locally before deployment.

Troubleshooting
tkinter Not Found

If you encounter errors related to tkinter (e.g., No module named tkinter), follow these steps based on your operating system:

    Windows:
        Reinstall Python and ensure that the "tcl/tk and IDLE" option is selected during installation.
    macOS:
        Reinstall Python using the official installer from python.org.
    Linux:

        Install tkinter using your distribution's package manager.

        Debian/Ubuntu:

        bash

sudo apt-get update
sudo apt-get install python3-tk

Fedora:

bash

sudo dnf install python3-tkinter

Arch Linux:

bash

        sudo pacman -S tk

OpenAI API Issues

    Invalid API Key:
        Ensure that your OPENAI_API_KEY in the .env file is correct.
    Rate Limits Exceeded:
        Check your OpenAI dashboard for usage statistics and consider upgrading your plan if necessary.
    Network Issues:
        Ensure you have a stable internet connection.

Other Common Issues

    Missing Dependencies:
        Verify that all required Python packages are installed. Reinstall if necessary.

    SpaCy Model Not Found:

        Ensure you've downloaded the SpaCy model using:

        bash

        python -m spacy download en_core_web_sm

    Script Crashes or Doesn't Display Code:
        Check the console or log output for error messages.
        Ensure that the PDF provided is not corrupted and follows standard formatting.

Contributing

Contributions are welcome! Follow these steps to contribute:

    Fork the Repository.

    Create a New Branch:

    bash

git checkout -b feature/YourFeatureName

Commit Your Changes:

bash

git commit -m "Add some feature"

Push to the Branch:

bash

    git push origin feature/YourFeatureName

    Open a Pull Request.

Please ensure that your contributions adhere to the project's coding standards and include relevant tests and documentation.
License

This project is licensed under the MIT License.

Disclaimer: This tool leverages OpenAI's GPT-4 model to generate code based on the content of quantitative finance articles. While it strives for accuracy, always review and validate the generated algorithms before deploying them in live trading environments.

