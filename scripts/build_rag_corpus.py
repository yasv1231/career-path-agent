from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_resources import load_rag_resources

try:
    from langsmith_integration import get_chat_model
except Exception:
    get_chat_model = None


SENTENCE_RE = re.compile(r"[^。！？!?;\n]+[。！？!?;]?")
URL_RE = re.compile(r"https?://[^\s)]+")

DOC_TYPE_KEYWORDS = {
    "job_profile": [
        "job profile",
        "job summary",
        "role overview",
        "responsibility",
        "responsibilities",
        "岗位",
        "职责",
        "工作内容",
    ],
    "competency": [
        "competency",
        "competencies",
        "skill model",
        "skill matrix",
        "能力模型",
        "能力要求",
        "技能要求",
        "胜任力",
    ],
    "resource": [
        "resource",
        "resources",
        "course",
        "courses",
        "tutorial",
        "learning path",
        "学习资源",
        "课程",
        "项目",
    ],
}

RESOURCE_TYPE_KEYWORDS = {
    "course": ["course", "课程", "tutorial", "教程", "training", "certification"],
    "project": ["project", "项目", "case study", "实战", "portfolio"],
}


def _try_parse_json(text: str) -> Any:
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    return None


def split_into_chunks(
    text: str,
    max_chars: int = 420,
    overlap: int = 80,
    min_chunk_chars: int = 40,
) -> List[str]:
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return []

    paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
    units = paragraphs
    if len(paragraphs) <= 1:
        sentences = [item.strip() for item in SENTENCE_RE.findall(raw) if item.strip()]
        if sentences:
            units = sentences

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        add_len = len(unit) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunk = " ".join(current).strip()
            if len(chunk) >= min_chunk_chars:
                chunks.append(chunk)

            if overlap > 0 and chunk:
                tail = chunk[-overlap:]
                current = [tail]
                current_len = len(tail)
            else:
                current = []
                current_len = 0

        current.append(unit)
        current_len += add_len

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return [raw]
    return chunks


def parse_front_matter(raw: str) -> Tuple[Dict[str, str], str]:
    text = raw.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    header = text[4:end]
    body = text[end + 5 :]
    meta: Dict[str, str] = {}
    for line in header.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def infer_from_filename(path: Path) -> Dict[str, str]:
    stem = path.stem
    if "__" not in stem:
        return {}

    parts = stem.split("__")
    result: Dict[str, str] = {}
    if len(parts) >= 1 and parts[0]:
        result["career"] = parts[0].replace("_", " ")
    if len(parts) >= 2 and parts[1]:
        result["type"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        result["title"] = parts[2].replace("_", " ")
    return result


def _keyword_score(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword and keyword.lower() in lowered)


def _extract_title(body: str, path: Path) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:80]

    first = next((line.strip() for line in body.splitlines() if line.strip()), "")
    if first:
        return first[:80]
    return path.stem


def _infer_career(body: str, aliases: Dict[str, List[str]]) -> str:
    lowered = body.lower()
    best_key = "general"
    best_score = 0
    for career, alias_list in aliases.items():
        score = 0
        for alias in alias_list:
            alias_norm = alias.strip().lower()
            if alias_norm and alias_norm in lowered:
                score += 1
        if score > best_score:
            best_score = score
            best_key = career
    return best_key


def _infer_doc_type(body: str, has_url: bool) -> str:
    lowered = body.lower()
    scores = {
        doc_type: _keyword_score(lowered, keywords)
        for doc_type, keywords in DOC_TYPE_KEYWORDS.items()
    }
    if has_url:
        scores["resource"] += 1
    best = max(scores.items(), key=lambda item: item[1])[0]
    if scores[best] <= 0:
        return "resource"
    return best


def _infer_resource_type(body: str) -> str:
    lowered = body.lower()
    scores = {
        resource_type: _keyword_score(lowered, keywords)
        for resource_type, keywords in RESOURCE_TYPE_KEYWORDS.items()
    }
    best = max(scores.items(), key=lambda item: item[1])[0]
    if scores[best] <= 0:
        return ""
    return best


def _llm_semantic_meta(
    body: str,
    path: Path,
    aliases: Dict[str, List[str]],
    llm: Any,
) -> Dict[str, str]:
    if not llm:
        return {}

    career_options = sorted(aliases.keys())
    prompt = (
        "You are a semantic document classifier for RAG ingestion.\n"
        "Return JSON only with keys: career, type, title, resource_type.\n"
        "career must be one of: "
        + ", ".join(career_options + ["general"])
        + "\n"
        "type must be one of: job_profile, competency, resource.\n"
        "resource_type must be empty or one of: course, project.\n\n"
        f"Filename: {path.name}\n"
        f"Text:\n{body[:4000]}"
    )

    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
    except Exception:
        return {}

    data = _try_parse_json(text)
    if not isinstance(data, dict):
        return {}

    result: Dict[str, str] = {}
    for key in ("career", "type", "title", "resource_type"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    return result


def _semantic_infer(path: Path, body: str, aliases: Dict[str, List[str]], use_llm: bool, llm: Any) -> Dict[str, str]:
    has_url = bool(URL_RE.search(body))
    meta = {
        "career": _infer_career(body, aliases),
        "type": _infer_doc_type(body, has_url=has_url),
        "title": _extract_title(body, path),
        "resource_type": _infer_resource_type(body),
    }

    if use_llm:
        llm_meta = _llm_semantic_meta(body, path, aliases, llm)
        for key, value in llm_meta.items():
            meta[key] = value

    return meta


def _sanitize_id(raw: str) -> str:
    lowered = raw.lower().replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    if cleaned:
        return cleaned
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"doc_{digest}"


def parse_doc(
    path: Path,
    aliases: Dict[str, List[str]],
    use_llm_semantic: bool,
    llm: Any,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8-sig")
    meta, body = parse_front_matter(raw)
    fallback = infer_from_filename(path)
    semantic = _semantic_infer(path, body, aliases, use_llm=use_llm_semantic, llm=llm)

    career = meta.get("career") or fallback.get("career") or semantic.get("career") or "general"
    doc_type = meta.get("type") or fallback.get("type") or semantic.get("type") or "resource"
    title = meta.get("title") or fallback.get("title") or semantic.get("title") or path.stem
    resource_type = meta.get("resource_type") or semantic.get("resource_type", "")
    url = meta.get("url") or (URL_RE.search(body).group(0) if URL_RE.search(body) else "")
    base_id = meta.get("id") or f"{doc_type}_{career}_{path.stem}"
    doc_id = _sanitize_id(base_id)

    chunks = [
        {"text": chunk}
        for chunk in split_into_chunks(
            body,
            max_chars=chunk_size,
            overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
    ]

    return {
        "id": doc_id,
        "type": doc_type,
        "career": career,
        "title": title,
        "resource_type": resource_type,
        "url": url,
        "chunks": chunks,
    }


def collect_docs(docs_dir: Path) -> List[Path]:
    files: List[Path] = []
    for suffix in ("*.md", "*.txt"):
        files.extend(sorted(docs_dir.rglob(suffix)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/rag_corpus.json from free-form .md/.txt docs with semantic auto-inference."
    )
    parser.add_argument("--docs-dir", default="data/knowledge_docs", help="Directory with raw text docs.")
    parser.add_argument("--output", default="data/rag_corpus.json", help="Target corpus JSON path.")
    parser.add_argument("--use-llm-semantic", action="store_true", help="Use LLM to refine semantic metadata.")
    parser.add_argument("--chunk-size", type=int, default=420, help="Target max characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Overlap characters between chunks.")
    parser.add_argument("--min-chunk-chars", type=int, default=40, help="Minimum chunk length to keep.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    resources = load_rag_resources()
    aliases = resources.get("career_aliases", {})
    llm = get_chat_model() if (args.use_llm_semantic and get_chat_model) else None

    corpus: List[Dict[str, Any]] = []
    for path in collect_docs(docs_dir):
        if path.name.lower() == "readme.md":
            continue
        record = parse_doc(
            path=path,
            aliases=aliases,
            use_llm_semantic=args.use_llm_semantic,
            llm=llm,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_chars=args.min_chunk_chars,
        )
        if record.get("chunks"):
            corpus.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

    by_type: Dict[str, int] = {}
    by_career: Dict[str, int] = {}
    for item in corpus:
        by_type[item["type"]] = by_type.get(item["type"], 0) + 1
        by_career[item["career"]] = by_career.get(item["career"], 0) + 1

    print(f"Built corpus: {output}")
    print(f"Documents: {len(corpus)}")
    print(f"By type: {json.dumps(by_type, ensure_ascii=False)}")
    print(f"By career: {json.dumps(by_career, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
