# quantcli/search.py

import requests
import json
import logging
import webbrowser
import os

logger = logging.getLogger(__name__)

def search_crossref(query: str, rows: int = 5):
    """
    Search the CrossRef API for articles matching the query.

    Args:
        query (str): The search query.
        rows (int): Number of results to return.

    Returns:
        list: A list of articles with relevant details.
    """
    logger.info(f"Searching CrossRef for query: '{query}' with {rows} results.")
    api_url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("message", {}).get("items", [])
        articles = []
        for index, item in enumerate(items, start=1):
            article = {
                "id": str(index),
                "title": item.get("title", ["No title available"])[0],
                "authors": ", ".join(
                    f"{author.get('given', '')} {author.get('family', '')}".strip()
                    for author in item.get("author", [])
                ) or "No authors available",
                "published": (
                    (item.get("published-print") or item.get("published-online") or {})
                    .get("date-parts", [[None]])[0][0]
                ),
                "URL": item.get("URL", "#"),
                "DOI": item.get("DOI", None),
                "abstract": item.get("abstract", "No abstract available."),
            }
            articles.append(article)
        logger.info(f"Found {len(articles)} articles.")
        return articles
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching articles from CrossRef: {e}")
        return []

def save_to_html(articles: list, filename: str = "output.html"):
    """
    Save the list of articles to an HTML file and open it in the default web browser.

    Args:
        articles (list): List of articles to save.
        filename (str): The name of the HTML file.
    """
    logger.info(f"Saving articles to HTML file: {filename}")
    html_content = """
    <html>
        <head>
            <title>QuantCLI Search Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                ul { list-style-type: none; padding: 0; }
                li { margin-bottom: 15px; }
                a { text-decoration: none; color: #1a0dab; }
                a:hover { text-decoration: underline; }
                .authors { color: #555; }
                .published { color: #777; }
            </style>
        </head>
        <body>
            <h1>Search Results</h1>
            <ul>
    """
    for article in articles:
        html_content += f"""
                <li>
                    <a href="{article['URL']}" target="_blank">{article['title']}</a><br/>
                    <span class="authors">{article['authors']}</span><br/>
                    <span class="published">{article['published']}</span>
                </li>
        """
    html_content += """
            </ul>
        </body>
    </html>
    """

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Articles saved to {filename}")
        # Open the HTML file in the default web browser
        filepath = os.path.abspath(filename)
        webbrowser.open(f"file://{filepath}")
        logger.info(f"Opened {filename} in the web browser.")
    except Exception as e:
        logger.error(f"Failed to save or open HTML file: {e}")

