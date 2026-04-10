import re
from urllib.parse import urlparse
import emoji
import pandas as pd
from chat_parser import parse_whatsapp_chat
from config import WHATSAPP_SYSTEM_PATTERNS, URL_DOMAIN_LABELS

_SYSTEM_MSG_RE = re.compile("|".join(WHATSAPP_SYSTEM_PATTERNS), re.IGNORECASE)

_DOC_ATTACHMENT_RE = re.compile(r"(.+?)\s*•\s*\d+\s*(?:sayfa|page)\s*belge dahil edilmedi", re.IGNORECASE)

_HTTP_URL_RE = re.compile(r"https?://\S+")

_LAUGHTER_RE = re.compile(r'\b[A-ZĞÜŞİÖÇa-zğüşıöç]{6,}\b')

def _is_laughter(token: str) -> bool:
    if not token.isupper():
        return False
    if len(token) >= 3 and len(set(token[-3:])) == 1:
        return False
    if len(set(token)) / len(token) >= 0.75:
        return False
    vowels = set("AEIİOUÖÜaeiıouöü")
    vowel_count = sum(1 for c in token if c in vowels)
    return vowel_count / len(token) < 0.25

def _replace_laughter(text: str) -> str:
    """Replace keyboard-smash laughter tokens with [kahkaha]."""
    def replacer(m):
        return "[kahkaha]" if _is_laughter(m.group(0)) else m.group(0)
    # Collapse multiple consecutive [kahkaha] into one
    result = _LAUGHTER_RE.sub(replacer, text)
    result = re.sub(r'(\[kahkaha\]\s*){2,}', '[kahkaha] ', result)
    return result.strip()


_MAX_URL_LEN = 80  # URLs longer than this are shortened to domain only

def _label_for_url(url: str) -> str:
    """Return a Turkish semantic label + URL.

    Short URLs (<=80 chars) are kept in full.
    Long URLs are shortened to scheme://domain to save token budget.
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    display_url = url if len(url) <= _MAX_URL_LEN else f"{parsed.scheme}://{domain}"
    for domains, label in URL_DOMAIN_LABELS:
        if domain in domains:
            return f"{label} {display_url}"
    return f"[link paylaşıldı] {display_url}"


def _replace_urls(text: str) -> str:
    """Replace each URL in text with label + URL."""
    return _HTTP_URL_RE.sub(lambda m: _label_for_url(m.group(0)), text)


def _is_emoji_only(text: str) -> bool:
    """Return True if message contains nothing but emojis/punctuation/whitespace."""
    return len(emoji.replace_emoji(text, replace="").strip()) == 0


def _relabel_doc(text: str) -> str:
    """Replace document attachment lines with a readable label + filename."""
    return _DOC_ATTACHMENT_RE.sub(lambda m: f"[belge paylaşıldı: {m.group(1).strip()}]", text)


def clean_chat_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove system messages, drop emoji-only rows, and annotate URLs and documents."""
    # 1. Drop system rows
    mask = df["Message"].str.contains(_SYSTEM_MSG_RE, na=False)
    cleaned = df[~mask].copy().reset_index(drop=True)
    print(f"[clean] Dropped {mask.sum()} system rows, {len(cleaned)} rows remaining.")

    # 2. Drop emoji-only messages
    emoji_mask = cleaned["Message"].apply(_is_emoji_only)
    cleaned = cleaned[~emoji_mask].reset_index(drop=True)
    print(f"[clean] Dropped {emoji_mask.sum()} emoji-only rows, {len(cleaned)} rows remaining.")

    # 3. Relabel document attachments
    cleaned["Message"] = cleaned["Message"].apply(_relabel_doc)

    # 4. Annotate URLs with semantic labels
    cleaned["Message"] = cleaned["Message"].apply(_replace_urls)

    # 5. Replace keyboard-smash laughter with [kahkaha]
    cleaned["Message"] = cleaned["Message"].apply(_replace_laughter)

    return cleaned


def chunk_messages(
    df: pd.DataFrame,
    source: str = "unknown",
    window_size: int = 8,
    overlap: int = 2,
) -> list[dict]:
    """
    Group sequential messages into overlapping context windows.

    Each chunk combines `window_size` consecutive messages formatted as
    "Sender: Message" lines, preserving conversational context.
    """
    if df.empty:
        return []

    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce",
    )

    records = df.to_dict("records")
    step = window_size - overlap
    chunks = []

    for start in range(0, len(records), step):
        window = records[start : start + window_size]
        if not window:
            break

        lines = [f"{r['Sender']}: {r['Message']}" for r in window]
        chunk_text = "\n".join(lines)

        datetimes = [r["datetime"] for r in window if pd.notna(r["datetime"])]
        start_dt = min(datetimes) if datetimes else None
        end_dt   = max(datetimes) if datetimes else None

        participants = list(dict.fromkeys(r["Sender"] for r in window))

        # If chunk exceeds 500 chars, split into sub-chunks of 500 chars each.
        # First tries line boundaries; if a single line is still >500 chars
        # (e.g. long self-notes with no newlines), force-splits at word boundaries.
        if len(chunk_text) > 500:
            sub_lines = chunk_text.split("\n")

            # Force-split any single line that exceeds 500 chars at word boundaries
            expanded_lines = []
            for line in sub_lines:
                if len(line) <= 500:
                    expanded_lines.append(line)
                else:
                    words = line.split(" ")
                    current_part = ""
                    for word in words:
                        if len(current_part) + len(word) + 1 > 500 and current_part:
                            expanded_lines.append(current_part.strip())
                            current_part = word
                        else:
                            current_part += (" " if current_part else "") + word
                    if current_part:
                        expanded_lines.append(current_part.strip())

            current_sub, current_len = [], 0
            for line in expanded_lines:
                if current_len + len(line) > 500 and current_sub:
                    chunks.append({
                        "chunk_text": "\n".join(current_sub),
                        "start_datetime": start_dt,
                        "end_datetime": end_dt,
                        "participants": participants,
                        "source": source,
                    })
                    current_sub, current_len = [], 0
                current_sub.append(line)
                current_len += len(line) + 1
            if current_sub:
                chunks.append({
                    "chunk_text": "\n".join(current_sub),
                    "start_datetime": start_dt,
                    "end_datetime": end_dt,
                    "participants": participants,
                    "source": source,
                })
        else:
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "start_datetime": start_dt,
                    "end_datetime": end_dt,
                    "participants": participants,
                    "source": source,
                }
            )

    print(f"[chunk] {source}: {len(chunks)} chunks (window={window_size}, overlap={overlap}).")
    return chunks


def process_all_chats(
    data_dir: str = "data",
    window_size: int = 8,
    overlap: int = 2,
) -> list[dict]:
    """
    Parse, clean, and chunk every .txt file found in data_dir.
    """
    from pathlib import Path

    chat_files = sorted(Path(data_dir).glob("*.txt"))
    if not chat_files:
        print(f"[warn] No .txt files found in {data_dir}/")
        return []

    all_chunks = []
    for path in chat_files:
        source = path.stem
        print(f"\n--- Processing: {source} ---")
        df_raw = parse_whatsapp_chat(str(path))
        print(f"  Raw rows: {len(df_raw)}")
        df_clean = clean_chat_data(df_raw)
        chunks = chunk_messages(df_clean, source=source, window_size=window_size, overlap=overlap)
        all_chunks.extend(chunks)

    print(f"\n[done] Total chunks across all chats: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    all_chunks = process_all_chats()

    # Show one sample chunk from each source
    seen = set()
    for chunk in all_chunks:
        src = chunk["source"]
        if src not in seen:
            seen.add(src)
            print(f"\n=== Sample chunk — source: {src} ===")
            print(f"Start       : {chunk['start_datetime']}")
            print(f"End         : {chunk['end_datetime']}")
            print(f"Participants: {chunk['participants']}")
            print(f"Text:\n{chunk['chunk_text']}")
