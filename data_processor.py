import re
import pandas as pd
from chat_parser import parse_whatsapp_chat

# System message patterns to filter out (Turkish + English)
_SYSTEM_PATTERNS = [
    r"<medya dahil edilmedi>",
    r"<media omitted>",
    r"görüntü dahil edilmedi",
    r"video dahil edilmedi",
    r"ses dahil edilmedi",
    r"sticker dahil edilmedi",
    r"ki[sş]i(yi|sini) [çc][ıi]kard[ıi]n[ıi]z",    # removed a person
    r"ki[sş]isini ekledi",                           # added a person
    r"grubu?.*(olu[sş]turdu|olu[sş]turdunuz)",       # group created
    r"ayr[ıi]ld[ıi]",                               # left the group
    r"kat[ıi]ld[ıi]",                               # joined via link
    r"u[çc]tan uca [sş]ifreli",                     # end-to-end encrypted warning
]

_SYSTEM_RE = re.compile("|".join(_SYSTEM_PATTERNS), re.IGNORECASE)


def clean_chat_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove WhatsApp system messages that carry no semantic value.
    """
    mask = df["Message"].str.contains(_SYSTEM_RE, na=False)
    cleaned = df[~mask].reset_index(drop=True)
    print(f"[clean] Dropped {mask.sum()} system rows, {len(cleaned)} rows remaining.")
    return cleaned


def chunk_messages(
    df: pd.DataFrame,
    window_size: int = 8,
    overlap: int = 2,
) -> list[dict]:
    """
    Group sequential messages into overlapping context windows.

    Each chunk combines `window_size` consecutive messages formatted as
    "Sender: Message" lines, preserving conversational context.
    Adjacent chunks share `overlap` messages so no context is lost at borders.
    """
    if df.empty:
        return []

    # Build a combined datetime column for metadata
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,   # handles EU/TR format: day.month.year
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

        participants = list(dict.fromkeys(r["Sender"] for r in window))  # ordered, unique

        chunks.append(
            {
                "chunk_text": chunk_text,
                "start_datetime": start_dt,
                "end_datetime": end_dt,
                "participants": participants,
            }
        )

    print(f"[chunk] Created {len(chunks)} chunks (window={window_size}, overlap={overlap}).")
    return chunks


if __name__ == "__main__":
    df_raw = parse_whatsapp_chat("data/_chat_2.txt")
    print(f"Raw rows: {len(df_raw)}\n")

    # Step 1 — clean
    df_clean = clean_chat_data(df_raw)

    # Step 2 — chunk (default window=8, overlap=2)
    chunks = chunk_messages(df_clean)

    # Print the first chunk and its metadata
    if chunks:
        first = chunks[0]
        print("\n--- First Chunk ---")
        print(f"Start : {first['start_datetime']}")
        print(f"End   : {first['end_datetime']}")
        print(f"Participants: {first['participants']}")
        print(f"Text:\n{first['chunk_text']}")
