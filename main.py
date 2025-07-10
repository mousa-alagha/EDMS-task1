# File: app.py
"""
Agentic RAG POC Application
- Streamlit chat UI
- Planning agent (JSON-based CoT)
- Query Rewriter agent
- Local PDF retrieval agent with page-level metadata
- Memory (SQLite-based)
- Quality Check agent verifying citations against PDFs
- Final LLM response with inline citations and citation list
- Logs fetched docs to incrementing text files
- Displays before and after QC responses
"""

import os
import re
import json
import sqlite3
from typing import List, Dict, Any, Tuple
import streamlit as st
from PyPDF2 import PdfReader

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from openai import OpenAI

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
openai = OpenAI()


# ---------- Memory Manager ----------
class MemoryManager:
    """Persistent conversation memory stored in SQLite"""
    def __init__(self, db_path: str = "memory1.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self.conn.commit()

    def add(self, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content)
        )
        self.conn.commit()

    def get_last(self, n: int = 10) -> List[Dict[str, str]]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = cur.fetchall()[::-1]
        return [{"role": r, "content": c} for r, c in rows]

# ---------- PDF Loader & Retriever ----------
@st.cache_resource
def init_vectorstore(pdf_dir: str = "docs") -> FAISS:
    """Load all PDFs by page, embed and index with FAISS"""
    embeddings = OpenAIEmbeddings()
    docs: List[Document] = []
    for fn in os.listdir(pdf_dir):
        if fn.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fn)
            try:
                reader = PdfReader(path, strict=False)
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": fn, "page": i}
                        ))
            except Exception as e:
                st.warning(f"Skipping PDF {fn}: {e}")
                continue
    return FAISS.from_documents(docs, embeddings)

# ---------- Agentic RAG Logic ----------
class AgenticRAG:
    def __init__(self):
        self.memory = MemoryManager()
        self.vstore = init_vectorstore()
        os.makedirs("fetched_data", exist_ok=True)

    def plan(self, query: str) -> Dict[str, Any]:
        """Generate a JSON plan for the query"""
        system = "You are a planning assistant. Output a JSON with key 'steps' listing agent tasks."
        example = {
            "steps": [
                {"agent": "rewriter", "action": "rewrite_query", "args": {}},
                {"agent": "local", "action": "retrieve", "args": {"top_k": 3}}
            ]
        }
        prompt = f"User query: {query}\nPlan format example: {json.dumps(example)}"
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return example

    def rewrite_query(self, query: str) -> str:
        """Refine user question into a concise retrieval query"""
        system = "You are a query rewriter. Convert the user question into a keyword-based search query, removing stop words and question phrasing."
        prompt = f"Original question: {query}\nRewritten query:"
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    def fetch_local(self, query: str, top_k: int) -> List[Document]:
        """Retrieve top_k relevant pages from local PDFs using refined query"""
        return self.vstore.similarity_search(query, k=top_k)

    def generate_answer(self, query: str, docs: List[Document], history: List[Dict[str, str]]) -> str:
        """Produce final answer with inline citations and citation list"""
        system = "You are a helpful assistant. Cite sources inline using [filename, page]."
        memory_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        docs_text = "\n---\n".join(
            f"[{d.metadata['source']}, page {d.metadata['page']}]{d.page_content}"
            for d in docs
        )
        user_content = f"Memory:\n{memory_text}\nDocs:\n{docs_text}\nQuery: {query}"
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        answer = resp.choices[0].message.content
        # Post-process citations
        cites = re.findall(r"\[([^,\]]+),\s*page\s*(\d+)\]", answer)
        unique = []
        seen = set()
        for file, pg in cites:
            entry = f"{file}, page {pg}"
            if entry not in seen:
                seen.add(entry)
                unique.append(entry)
        if unique:
            sources_list = "\n".join(f"- {u}" for u in unique)
            answer += f"\n\n**Sources:**\n{sources_list}"
        return answer

    def quality_check(self, answer: str, docs: List[Document]) -> str:
        """Verify inline citations against PDF content; correct if needed"""
        corrected = answer
        for d in docs:
            tag = f"[{d.metadata['source']}, page {d.metadata['page']}]"
            if tag in corrected:
                path = os.path.join("docs", d.metadata['source'])
                try:
                    reader = PdfReader(path)
                    page_text = reader.pages[d.metadata['page']-1].extract_text() or ""
                    snippet = d.page_content.strip()[:50]
                    if snippet not in page_text:
                        for i, page in enumerate(reader.pages, start=1):
                            text = page.extract_text() or ""
                            if snippet in text:
                                corrected = corrected.replace(tag, f"[{d.metadata['source']}, page {i}]")
                                break
                except Exception:
                    continue
        return corrected

    def handle(self, query: str) -> Tuple[str, str]:
        self.memory.add("user", query)
        plan = self.plan(query)
        refined = query
        docs: List[Document] = []
        for step in plan.get("steps", []):
            if step.get("agent") == "rewriter":
                refined = self.rewrite_query(query)
                self.memory.add("system", f"Refined query: {refined}")
            elif step.get("agent") == "local":
                k = step["args"].get("top_k", 3)
                docs.extend(self.fetch_local(refined, k))
        if docs:
            os.makedirs("fetched_data", exist_ok=True)
            dump = "\n---\n".join(f"[{d.metadata['source']}, page {d.metadata['page']}]{d.page_content}"
                                  for d in docs)
            existing = [f for f in os.listdir("fetched_data") if f.startswith("fetched_data_")]
            inds = [int(re.search(r"fetched_data_(\d+)", f).group(1)) for f in existing] if existing else []
            next_i = max(inds) + 1 if inds else 1
            with open(os.path.join("fetched_data", f"fetched_data_{next_i}.txt"), "w", encoding="utf-8") as fout:
                fout.write(dump)
        history = self.memory.get_last(10)
        raw_answer = self.generate_answer(refined, docs, history)
        final_answer = self.quality_check(raw_answer, docs)
        self.memory.add("assistant", final_answer)
        return raw_answer, final_answer


# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Agentic RAG POC")
    st.title("🦜🔗 Agentic RAG POC")

    if "agent" not in st.session_state:
        st.session_state.agent = AgenticRAG()

    # Display past chat
    for msg in st.session_state.agent.memory.get_last(50):
        st.chat_message(msg["role"]).write(msg["content"])

    # New input
    if prompt := st.chat_input("Ask me anything..."):
        with st.spinner("Thinking..."):
            raw, final = st.session_state.agent.handle(prompt)
        # Show both versions
        st.chat_message("assistant").write(f"**Before Quality Check:**\n{raw}")
        st.chat_message("assistant").write(f"**After Quality Check:**\n{final}")


if __name__ == "__main__":
    main()
