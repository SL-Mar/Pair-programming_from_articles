# quantcli/gui.py

import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, ttk
from .search import search_crossref
import json
import os
import logging
from .processor import ArticleProcessor
import webbrowser
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from pygments.styles import get_style_by_name
from tkinter.font import Font


# Configure a logger for the GUI
logger = logging.getLogger(__name__)

class QuantCLIGUI:
    def __init__(self, master):
        self.master = master
        master.title("Quant Coder v1.0 - SL Mar 2024")
        master.geometry("800x600")

        self.label = tk.Label(master, text="Quantitative research from articles", font=("Helvetica", 16))
        self.label.pack(pady=10)

        # Search Frame
        self.search_frame = tk.Frame(master)
        self.search_frame.pack(pady=10)

        self.search_label = tk.Label(self.search_frame, text="Search Query:")
        self.search_label.pack(side=tk.LEFT, padx=5)

        self.search_entry = tk.Entry(self.search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, padx=5)

        self.num_label = tk.Label(self.search_frame, text="Number of Results:")
        self.num_label.pack(side=tk.LEFT, padx=5)

        self.num_entry = tk.Entry(self.search_frame, width=5)
        self.num_entry.pack(side=tk.LEFT, padx=5)
        self.num_entry.insert(0, "5")

        self.search_button = tk.Button(master, text="Search", command=self.perform_search)
        self.search_button.pack(pady=10)

        # Results Frame
        self.results_frame = tk.Frame(master)
        self.results_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Instructions Label
        self.instructions_label = tk.Label(
            self.results_frame,
            text="Double-click an article to open it in your web browser."
        )
        self.instructions_label.pack(side=tk.TOP, pady=5)

        # Treeview for displaying search results
        self.results_tree = ttk.Treeview(
            self.results_frame,
            columns=('Index', 'Title', 'Authors'),
            show='headings'
        )
        self.results_tree.heading('Index', text='Index')
        self.results_tree.heading('Title', text='Title')
        self.results_tree.heading('Authors', text='Authors')
        self.results_tree.column('Index', width=20)
        self.results_tree.column('Title', width=400)
        self.results_tree.column('Authors', width=200)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind events to change the cursor and handle double-clicks
        self.results_tree.bind("<Enter>", lambda e: self.results_tree.config(cursor="hand2"))
        self.results_tree.bind("<Leave>", lambda e: self.results_tree.config(cursor=""))
        self.results_tree.bind('<Double-1>', self.on_article_double_click)

        self.scrollbar = tk.Scrollbar(self.results_frame, command=self.results_tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_tree.config(yscrollcommand=self.scrollbar.set)

        # Action Buttons
        self.actions_frame = tk.Frame(master)
        self.actions_frame.pack(pady=10)

        self.open_button = tk.Button(
            self.actions_frame,
            text="Open Article",
            command=self.open_selected_article
        )
        self.open_button.pack(side=tk.LEFT, padx=5)

        self.summarize_button = tk.Button(
            self.actions_frame,
            text="Summarize Article",
            command=self.summarize_article
        )
        self.summarize_button.pack(side=tk.LEFT, padx=5)

        self.generate_button = tk.Button(
            self.actions_frame,
            text="Generate Code",
            command=self.generate_code
        )
        self.generate_button.pack(side=tk.LEFT, padx=5)

        # Initialize articles list
        self.articles = []

    def perform_search(self):
        query = self.search_entry.get().strip()
        num = self.num_entry.get().strip()

        if not query:
            messagebox.showwarning("Input Error", "Please enter a search query.")
            return

        try:
            num = int(num)
        except ValueError:
            messagebox.showwarning("Input Error", "Number of results must be an integer.")
            return

        logger.info(f"GUI: Searching for '{query}' with {num} results.")
        articles = search_crossref(query, rows=num)

        if not articles:
            messagebox.showinfo("Search Results", "No articles found or an error occurred during the search.")
            return

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Display results in Treeview
        for idx, article in enumerate(articles):
            self.results_tree.insert('', 'end', values=(idx, article['title'], article['authors']))

        # Store articles for later use
        self.articles = articles

    def on_article_double_click(self, event):
        self.open_selected_article()

    def open_selected_article(self):
        selected_item = self.results_tree.selection()
        if selected_item:
            item = self.results_tree.item(selected_item)
            index = int(item['values'][0])
            self.open_article_by_id(index)
        else:
            messagebox.showwarning("No Selection", "Please select an article to open.")

    def open_article_by_id(self, index):
        try:
            article = self.articles[index]
            webbrowser.open(article["URL"])
        except IndexError:
            messagebox.showwarning("Invalid Index", f"Article at index {index} not found.")

    def summarize_article(self):
        """
        Allows the user to select a PDF file and then summarizes it.
        """
        # Open file dialog to select a PDF file
        filepath = filedialog.askopenfilename(
            title="Select Article PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if not filepath:
            return  # User canceled the file dialog

        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", "The selected file does not exist.")
            return

        processor = ArticleProcessor()
        extracted_data = processor.extract_structure(filepath)
        if not extracted_data:
            messagebox.showerror("Error", "Failed to extract data from the article.")
            return

        summary = processor.openai_handler.generate_summary(extracted_data)
        if summary:
            # Automatically save the summary
            summary_filename = os.path.splitext(filepath)[0] + '.txt'
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary)
            messagebox.showinfo("Summary Saved", f"Summary saved as {summary_filename}")
            # Display the summary in a new window
            self.display_summary(summary)
        else:
            messagebox.showerror("Error", "Failed to generate summary.")

    def generate_code(self):
        """
        Allows the user to select a summary text file and then generates code based on it.
        """
        # Open file dialog to select a summary text file
        summary_path = filedialog.askopenfilename(
            title="Select Summary Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not summary_path:
            return  # User canceled the file dialog

        if not os.path.exists(summary_path):
            messagebox.showerror("File Not Found", "The selected file does not exist.")
            return

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = f.read()

        processor = ArticleProcessor()
        qc_code = processor.openai_handler.generate_qc_code(summary)

        attempt = 0
        max_attempts = processor.max_refine_attempts
        while qc_code and not processor.code_validator.validate_code(qc_code) and attempt < max_attempts:
            qc_code = processor.code_refiner.refine_code(qc_code)
            attempt += 1

        if qc_code and processor.code_validator.validate_code(qc_code):
            # Display the code in a new window with syntax highlighting
            self.display_code(qc_code)
        else:
            messagebox.showerror("Error", "Failed to generate valid QuantConnect code.")

    def display_summary(self, summary):
        """
        Displays the summary in a new window.
        """
        summary_window = tk.Toplevel(self.master)
        summary_window.title("Article Summary")
        summary_text = scrolledtext.ScrolledText(
            summary_window, wrap=tk.WORD, width=100, height=30,
            bg='#FFFFFF', fg='#000000', insertbackground='#000000'
        )
        summary_text.pack(expand=True, fill='both')
        summary_text.insert(tk.END, summary)
        summary_text.configure(state='disabled')

    def display_code(self, code):
        """
        Displays the generated code in a new window with syntax highlighting.
        """
        code_window = tk.Toplevel(self.master)
        code_window.title("Generated QuantConnect Code")
        code_text = scrolledtext.ScrolledText(
            code_window, wrap=tk.NONE, width=100, height=30,
            bg='#282a36', fg='#f8f8f2', insertbackground='#ffffff'
        )
        code_text.pack(expand=True, fill='both')

        code_font = Font(family='Courier New', size=12)
        code_text.configure(font=code_font)

        # Apply syntax highlighting
        self.apply_syntax_highlighting(code, code_text)

    def apply_syntax_highlighting(self, code: str, text_widget: scrolledtext.ScrolledText):
        """
        Apply syntax highlighting to the code using Pygments and insert it into the Text widget.
        """
        logger = logging.getLogger('syntax_highlighting')
        logger.setLevel(logging.DEBUG)
        try:
            lexer = PythonLexer()
            style = get_style_by_name('monokai')  # Choose a Pygments style

            # Set the background color to match the style
            background_color = '#272822'  # Monokai background color
            text_widget.configure(background=background_color)

            # Configure font
            code_font = Font(family='Courier New', size=12)
            text_widget.configure(font=code_font)

            # Clear existing tags
            for tag in text_widget.tag_names():
                text_widget.tag_delete(tag)

            # Define tags based on the Pygments style
            for token, _ in style:
                tag_name = str(token).replace('Token.', '').replace('.', '_')
                tstyle = style.style_for_token(token)
                style_props = {}

                # Set foreground color
                if tstyle['color']:
                    style_props['foreground'] = f"#{tstyle['color']}"
                else:
                    style_props['foreground'] = '#F8F8F2'  # Default foreground color

                # Set background color
                if tstyle['bgcolor']:
                    style_props['background'] = f"#{tstyle['bgcolor']}"

                # Set font weight and slant
                if tstyle['bold']:
                    style_props['font'] = code_font.copy()
                    style_props['font'].configure(weight='bold')
                if tstyle['italic']:
                    if 'font' not in style_props:
                        style_props['font'] = code_font.copy()
                    style_props['font'].configure(slant='italic')

                # Configure tag with style properties
                text_widget.tag_configure(tag_name, **style_props)

            # Tokenize the code using Pygments
            tokens = list(lex(code, lexer))

            # Enable the widget to insert text
            text_widget.configure(state='normal')
            text_widget.delete(1.0, tk.END)  # Clear existing text

            # Insert tokens with appropriate tags
            for token_type, content in tokens:
                tag_name = str(token_type).replace('Token.', '').replace('.', '_')
                text_widget.insert(tk.END, content, tag_name)

            # Re-enable the widget
            text_widget.configure(state='disabled')
        except Exception as e:
            logger.error(f"Failed to apply syntax highlighting: {e}")
            print(f"Exception occurred: {e}")
            text_widget.configure(state='normal')
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, code)  # Fallback: insert without highlighting
            text_widget.configure(state='disabled')


def launch_gui():
    """
    Launches the full GUI for the QuantConnect CLI.
    """
    root = tk.Tk()
    app = QuantCLIGUI(root)
    root.mainloop()

