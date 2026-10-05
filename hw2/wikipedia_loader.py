"""Wikipedia document loader using the MediaWiki API directly.

The older ``wikipedia`` PyPI package often fails against Wikimedia's API
from cloud IPs. This loader uses HTTPS requests with an explicit User-Agent.
"""

from __future__ import annotations

import requests
from langchain_core.documents import Document


class WikipediaLoader:
    """Load Wikipedia pages matching a search query."""

    def __init__(self, query: str, load_max_docs: int = 4) -> None:
        """Store the search query and validate the requested document count.

        Args:
            query: Wikipedia search string.
            load_max_docs: Maximum number of pages to fetch.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if load_max_docs < 1:
            raise ValueError("load_max_docs must be at least 1")

        self.query = query
        self.load_max_docs = load_max_docs

    def load(self) -> list[Document]:
        """Search Wikipedia and return up to ``load_max_docs`` page documents.

        Returns:
            A list of LangChain ``Document`` objects with page text and metadata.
        """
        headers = {
            "User-Agent": (
                "PDXAcademicClient/gensec "
                "(Educational; contact: alongo2022@fau.edu)"
            )
        }
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": self.query,
                "srlimit": self.load_max_docs,
                "format": "json",
            },
            headers=headers,
            timeout=30,
        ).json()

        documents: list[Document] = []
        for hit in search.get("query", {}).get("search", []):
            title = hit["title"]
            page = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts|info",
                    "explaintext": True,
                    "titles": title,
                    "inprop": "url",
                    "format": "json",
                },
                headers=headers,
                timeout=30,
            ).json()
            for pdata in page.get("query", {}).get("pages", {}).values():
                extract = pdata.get("extract", "") or ""
                documents.append(
                    Document(
                        page_content=extract,
                        metadata={
                            "source": pdata.get(
                                "fullurl",
                                f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            ),
                            "title": pdata.get("title", title),
                            "summary": extract[:500],
                        },
                    )
                )
        return documents
