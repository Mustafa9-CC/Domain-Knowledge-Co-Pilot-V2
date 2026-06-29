import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self._token: str | None = None
        self._client = httpx.Client(timeout=120.0)

    @property
    def token(self) -> str | None:
        return self._token

    @token.setter
    def token(self, value: str | None):
        self._token = value

    @property
    def _headers(self) -> dict:
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def health(self) -> dict:
        r = self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def signup(self, email: str, username: str, password: str) -> dict:
        r = self._client.post(f"{self.base_url}/auth/signup", json={
            "email": email,
            "username": username,
            "password": password,
        })
        r.raise_for_status()
        return r.json()

    def login(self, email: str, password: str) -> dict:
        r = self._client.post(f"{self.base_url}/auth/login", json={
            "email": email,
            "password": password,
        })
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        return data

    def me(self) -> dict:
        r = self._client.get(f"{self.base_url}/auth/me", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def create_corpus(self, name: str, description: str = "") -> dict:
        r = self._client.post(f"{self.base_url}/corpora", json={
            "name": name,
            "description": description,
        }, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def list_corpora(self) -> list:
        r = self._client.get(f"{self.base_url}/corpora", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def get_corpus(self, corpus_id: int) -> dict:
        r = self._client.get(f"{self.base_url}/corpora/{corpus_id}", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def delete_corpus(self, corpus_id: int):
        r = self._client.delete(f"{self.base_url}/corpora/{corpus_id}", headers=self._headers)
        r.raise_for_status()

    def upload_document(self, corpus_id: int, file_path: str) -> dict:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            r = self._client.post(
                f"{self.base_url}/corpora/{corpus_id}/documents",
                files=files,
                headers=self._headers,
            )
        r.raise_for_status()
        return r.json()

    def list_documents(self, corpus_id: int) -> list:
        r = self._client.get(f"{self.base_url}/corpora/{corpus_id}/documents", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def get_document(self, document_id: int) -> dict:
        r = self._client.get(f"{self.base_url}/documents/{document_id}", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def delete_document(self, document_id: int):
        r = self._client.delete(f"{self.base_url}/documents/{document_id}", headers=self._headers)
        r.raise_for_status()

    def preview_document(self, document_id: int) -> dict:
        r = self._client.get(f"{self.base_url}/documents/{document_id}/preview", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def list_sessions(self, corpus_id: int) -> list:
        r = self._client.get(f"{self.base_url}/corpora/{corpus_id}/sessions", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def create_session(self, corpus_id: int, title: str = "") -> dict:
        r = self._client.post(f"{self.base_url}/corpora/{corpus_id}/sessions", json={
            "title": title,
        }, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def delete_session(self, session_id: int):
        r = self._client.delete(f"{self.base_url}/sessions/{session_id}", headers=self._headers)
        r.raise_for_status()

    def get_messages(self, session_id: int) -> list:
        r = self._client.get(f"{self.base_url}/sessions/{session_id}/messages", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def chat(self, corpus_id: int, question: str, session_id: int | None = None, debug: bool = False) -> dict:
        r = self._client.post(f"{self.base_url}/corpora/{corpus_id}/chat", json={
            "question": question,
            "session_id": session_id,
            "debug": debug,
        }, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def chat_stream(self, corpus_id: int, question: str, session_id: int | None = None):
        """Stream chat tokens via SSE.

        Yields:
            tuple[str, dict]: (event_type, data) pairs.
                event_type is 'token', 'done', or 'error'.
        """
        import json as _json
        with self._client.stream(
            "POST",
            f"{self.base_url}/corpora/{corpus_id}/chat/stream",
            json={"question": question, "session_id": session_id},
            headers=self._headers,
        ) as response:
            if response.status_code == 501:
                # Streaming disabled, fall back
                response.read()
                return
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_type = line[len("event: "):]
                elif line.startswith("data: "):
                    try:
                        data = _json.loads(line[len("data: "):])
                        yield event_type, data
                    except _json.JSONDecodeError:
                        continue

    def list_all_sessions(self) -> list:
        r = self._client.get(f"{self.base_url}/sessions", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def create_cross_corpus_session(self, corpus_ids: list[int] | None = None, title: str = "") -> dict:
        url = f"{self.base_url}/sessions"
        if corpus_ids:
            query = "&".join(f"corpus_ids={cid}" for cid in corpus_ids)
            url = f"{url}?{query}"
        r = self._client.post(url, json={"title": title}, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def cross_corpus_chat(self, question: str, corpus_ids: list[int] | str | None = None, session_id: int | None = None, debug: bool = False, retrieval_mode: str = "hybrid") -> dict:
        r = self._client.post(f"{self.base_url}/chat", json={
            "question": question,
            "corpus_ids": corpus_ids,
            "session_id": session_id,
            "debug": debug,
            "retrieval_mode": retrieval_mode,
        }, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def cross_corpus_chat_stream(self, question: str, corpus_ids: list[int] | str | None = None, session_id: int | None = None, retrieval_mode: str = "hybrid"):
        import json as _json
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/stream",
            json={
                "question": question,
                "corpus_ids": corpus_ids,
                "session_id": session_id,
                "retrieval_mode": retrieval_mode,
            },
            headers=self._headers,
        ) as response:
            if response.status_code == 501:
                response.read()
                return
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_type = line[len("event: "):]
                elif line.startswith("data: "):
                    try:
                        data = _json.loads(line[len("data: "):])
                        yield event_type, data
                    except _json.JSONDecodeError:
                        continue


api_client = ApiClient()

