import re
import pdfplumber
import spacy
from collections import defaultdict
from typing import Dict, List
import openai
import os
import logging
from dotenv import load_dotenv, find_dotenv
import ast
import tkinter as tk
from tkinter import scrolledtext
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from pygments.token import Token

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArticleProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load the text from a PDF file using pdfplumber for better accuracy.
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logging.info("PDF loaded successfully.")
            return text
        except Exception as e:
            logging.error(f"Failed to load PDF: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by removing headers, footers, references, and unnecessary whitespace.
        """
        try:
            # Remove URLs
            text = re.sub(r'https?://\S+', '', text)
            # Remove specific phrases
            text = re.sub(r'Electronic copy available at: .*', '', text)
            # Remove standalone numbers (e.g., page numbers)
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            # Remove multiple newlines
            text = re.sub(r'\n+', '\n', text)
            # Remove common header/footer patterns (example)
            text = re.sub(r'^\s*(Author|Title|Abstract)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
            # Strip leading and trailing whitespace
            text = text.strip()
            logging.info("Text preprocessed successfully.")
            return text
        except Exception as e:
            logging.error(f"Failed to preprocess text: {e}")
            return ""
    
    def detect_headings(self, text: str) -> List[str]:
        """
        Detect potential headings using NLP techniques.
        """
        try:
            doc = self.nlp(text)
            headings = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                # Simple heuristic: headings are short and title-cased
                if len(sent_text.split()) < 10 and sent_text.istitle():
                    headings.append(sent_text)
            logging.info(f"Detected {len(headings)} headings.")
            return headings
        except Exception as e:
            logging.error(f"Failed to detect headings: {e}")
            return []
    
    def split_into_sections(self, text: str, headings: List[str]) -> Dict[str, str]:
        """
        Split the text into sections based on the detected headings.
        """
        sections = defaultdict(str)
        current_section = "Introduction"  # Default section

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line in headings:
                current_section = line
            else:
                sections[current_section] += line + " "

        logging.info(f"Split text into {len(sections)} sections.")
        return sections
    
    def keyword_analysis(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Categorize sentences into trading signals and risk management based on keywords.
        """
        keyword_map = defaultdict(list)
        
        risk_management_keywords = [
            "drawdown", "volatility", "reduce", "limit", "risk", "risk-adjusted", 
            "maximal drawdown", "market volatility", "bear markets", "stability", 
            "sidestep", "reduce drawdown", "stop-loss", "position sizing", "hedging"
        ]
        trading_signal_keywords = [
            "buy", "sell", "signal", "indicator", "trend", "SMA", "moving average", 
            "momentum", "RSI", "MACD", "bollinger bands", "Rachev ratio", "stay long", 
            "exit", "market timing", "yield curve", "recession", "unemployment", 
            "housing starts", "Treasuries", "economic indicator"
        ]
        
        irrelevant_patterns = [
            r'figure \d+',  
            r'\[\d+\]',     
            r'\(.*?\)',     
            r'chart',       
            r'\bfigure\b',  
            r'performance chart',  
            r'\d{4}-\d{4}',  
            r'^\s*$'        
        ]
        
        processed_sentences = set()
        
        for section, content in sections.items():
            doc = self.nlp(content)
            for sent in doc.sents:
                sent_text = sent.text.lower().strip()
                
                if any(re.search(pattern, sent_text) for pattern in irrelevant_patterns):
                    continue
                if sent_text in processed_sentences:
                    continue
                processed_sentences.add(sent_text)
                
                if any(kw in sent_text for kw in trading_signal_keywords):
                    keyword_map['trading_signal'].append(sent.text.strip())
                elif any(kw in sent_text for kw in risk_management_keywords):
                    keyword_map['risk_management'].append(sent.text.strip())
        
        # Remove duplicates and sort
        for category, sentences in keyword_map.items():
            unique_sentences = list(set(sentences))
            keyword_map[category] = sorted(unique_sentences, key=len)
        
        logging.info("Keyword analysis completed.")
        return keyword_map
    
    def generate_summary(self, extracted_data: Dict[str, List[str]]) -> str:
        """
        Generate a summary of the article using OpenAI's API based on the extracted data.
        """
        trading_signals = '\n'.join(extracted_data.get('trading_signal', []))
        risk_management = '\n'.join(extracted_data.get('risk_management', []))
    
        prompt = f"""
        You are an expert in quantitative finance. Provide a concise summary of the trading strategies and risk management techniques described below.

        ### Trading Signals:
        {trading_signals}

        ### Risk Management:
        {risk_management}

        ### Summary:
        """
    
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in summarizing quantitative finance articles."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3,
                n=1
            )
            summary = response['choices'][0]['message']['content'].strip()
            logging.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logging.error(f"Failed to generate summary: {e}")
            return "Summary could not be generated."
    
    def generate_qc_code(self, extracted_data: Dict[str, List[str]]) -> str:
        """
        Generate QuantConnect Python code based on extracted data.
        """
        trading_signals = '\n'.join(extracted_data.get('trading_signal', []))
        risk_management = '\n'.join(extracted_data.get('risk_management', []))
    
        prompt = f"""
        You are an expert QuantConnect algorithm developer. Convert the following trading strategy and risk management descriptions into a complete, error-free QuantConnect Python algorithm.

        ### Trading Strategy:
        {trading_signals}

        ### Risk Management:
        {risk_management}

        ### Requirements:
        1. **Initialize Method**:
            - Set the start and end dates.
            - Set the initial cash.
            - Define the universe selection logic.
            - Initialize required indicators.
        2. **OnData Method**:
            - Implement buy/sell logic based on indicators.
            - Ensure indicators are updated correctly.
        3. **Risk Management**:
            - Implement drawdown limit of 15%.
            - Apply position sizing or stop-loss mechanisms as described.
        4. **Ensure Compliance**:
            - Use only QuantConnect's supported indicators and methods.
            - The code must be syntactically correct and free of errors.

        ### Example Structure:
        ```python
        from AlgorithmImports import *

        class MyAlgorithm(QCAlgorithm):
            def Initialize(self):
                self.SetStartDate(2020, 1, 1)
                self.SetEndDate(2023, 1, 1)
                self.SetCash(100000)
                # Define universe, indicators, etc.

            def OnData(self, data):
                # Trading logic

            def OnEndOfDay(self):
                # Risk management
        ```

        ### Generated Code:
        ```
        # The LLM will generate the code after this line
        ```
        """
    
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in generating QuantConnect algorithms in Python."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.3,
                n=1
            )
            generated_code = response['choices'][0]['message']['content'].strip()
            # Extract code block
            code_match = re.search(r'```python(.*?)```', generated_code, re.DOTALL)
            if code_match:
                generated_code = code_match.group(1).strip()
            logging.info("Code generated by LLM.")
            return generated_code
        except Exception as e:
            logging.error(f"Failed to generate code: {e}")
            return ""
    
    def validate_code(self, code: str) -> bool:
        """
        Validate the generated code for syntax errors.
        """
        try:
            ast.parse(code)
            logging.info("Generated code is syntactically correct.")
            return True
        except SyntaxError as e:
            logging.error(f"Syntax error in generated code: {e}")
            return False
    
    def refine_code(self, code: str) -> str:
        """
        Ask the LLM to fix syntax errors in the generated code.
        """
        prompt = f"""
        The following QuantConnect Python code has syntax errors. Please fix them and provide the corrected code.

        ```python
        {code}
        ```
        """
    
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in QuantConnect Python algorithms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2,
                n=1
            )
            corrected_code = response['choices'][0]['message']['content'].strip()
            code_match = re.search(r'```python(.*?)```', corrected_code, re.DOTALL)
            if code_match:
                corrected_code = code_match.group(1).strip()
            logging.info("Code refined by LLM.")
            return corrected_code
        except Exception as e:
            logging.error(f"Failed to refine code: {e}")
            return ""
    
    def display_summary_and_code(self, summary: str, code: str):
        """
        Display two windows: one with the summary and one with the generated code with syntax highlighting.
        """
        # Create the main Tkinter root
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Summary Window
        summary_window = tk.Toplevel()
        summary_window.title("Article Summary")
        summary_window.geometry("800x600")

        summary_text = scrolledtext.ScrolledText(summary_window, wrap=tk.WORD, font=("Arial", 12))
        summary_text.pack(expand=True, fill='both')
        summary_text.insert(tk.END, summary)
        summary_text.configure(state='disabled')  # Make it read-only

        # Code Window
        code_window = tk.Toplevel()
        code_window.title("Generated QuantConnect Code")
        code_window.geometry("1000x800")

        code_text = scrolledtext.ScrolledText(code_window, wrap=tk.NONE, font=("Consolas", 12), bg="#2B2B2B", fg="#F8F8F2")
        code_text.pack(expand=True, fill='both')

        # Apply syntax highlighting
        self.apply_syntax_highlighting(code, code_text)

        code_text.configure(state='disabled')  # Make it read-only

        # Start the Tkinter event loop
        root.mainloop()
    
    def apply_syntax_highlighting(self, code: str, text_widget: scrolledtext.ScrolledText):
        """
        Apply syntax highlighting to the code using Pygments and insert it into the Text widget.
        """
        lexer = PythonLexer()
        style = get_style_by_name('default')  # You can choose different styles
        tokens = lex(code, lexer)
        
        # Configure tags based on Pygments token types
        for token, content in tokens:
            token_name = str(token).replace('.', '_')
            if not text_widget.tag_names().__contains__(token_name):
                # Map Pygments token types to Tkinter colors
                if token in Token.Comment:
                    fg_color = "#6A9955"
                elif token in Token.Keyword:
                    fg_color = "#569CD6"
                elif token in Token.Name.Builtin:
                    fg_color = "#4EC9B0"
                elif token in Token.Literal.String:
                    fg_color = "#CE9178"
                elif token in Token.Operator:
                    fg_color = "#D4D4D4"
                elif token in Token.Punctuation:
                    fg_color = "#D4D4D4"
                elif token in Token.Name.Function:
                    fg_color = "#DCDCAA"
                elif token in Token.Name.Class:
                    fg_color = "#4EC9B0"
                else:
                    fg_color = "#D4D4D4"  # Default color
                
                text_widget.tag_config(token_name, foreground=fg_color)
            
            text_widget.insert(tk.END, content, token_name)
    
    def extract_structure(self, pdf_path: str) -> Dict[str, List[str]]:
        """
        Extract text from PDF, detect structure, and perform keyword analysis.
        """
        raw_text = self.load_pdf(pdf_path)
        if not raw_text:
            logging.error("No text extracted from PDF.")
            return {}
        
        preprocessed_text = self.preprocess_text(raw_text)
        headings = self.detect_headings(preprocessed_text)
        sections = self.split_into_sections(preprocessed_text, headings)
        keyword_analysis = self.keyword_analysis(sections)
        
        return keyword_analysis
    
    def extract_structure_and_generate_code(self, pdf_path: str) -> str:
        """
        Extract structure from PDF and generate QuantConnect code.
        """
        extracted_data = self.extract_structure(pdf_path)
        if not extracted_data:
            logging.error("No data extracted for code generation.")
            return ""
        
        # Generate summary
        summary = self.generate_summary(extracted_data)
        
        # Generate QuantConnect code
        qc_code = self.generate_qc_code(extracted_data)
        if not self.validate_code(qc_code):
            logging.info("Attempting to refine generated code.")
            qc_code = self.refine_code(qc_code)
            if not self.validate_code(qc_code):
                logging.error("Refined code is still invalid.")
                return ""
        
        # Display summary and code in separate windows
        self.display_summary_and_code(summary, qc_code)
        
        return qc_code

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Example usage
if __name__ == "__main__":
    # Initialize the processor with your OpenAI API key
    processor = ArticleProcessor()
    
    # Path to your PDF file
    pdf_path = "ssrn-4397638.pdf"  # Replace with your actual PDF path
    
    # Extract structure and generate QuantConnect code
    generated_code = processor.extract_structure_and_generate_code(pdf_path)
    
    if generated_code:
        logging.info("QuantConnect code generation and display completed successfully.")
    else:
        logging.error("Failed to generate and display QuantConnect code.")
