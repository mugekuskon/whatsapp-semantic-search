import re
import pandas as pd

_STRIP = re.compile(r"[\u200e\u200f\ufeff]")

# Matches three common WhatsApp export formats:

#   Android (US):  12/31/23, 14:30 - Sender: Message
#   iOS (US):      [12/31/23, 2:30:00 PM] Sender: Message
#   iOS (EU/TR):   [9.04.2019 18:53:21] Sender: Message

_PATTERN = re.compile(
    r"""
    (?:
        # Android
        (\d{1,2}/\d{1,2}/\d{2,4}),\s
        (\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s-\s
    |
        # iOS US
        \[(\d{1,2}/\d{1,2}/\d{2,4}),\s
        (\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s
    |
        # iOS EU/TR
        \[(\d{1,2}\.\d{1,2}\.\d{2,4})\s
        (\d{2}:\d{2}:\d{2})\]\s
    )
    ([^:]+?):\s   
    (.*)          
    """,
    re.VERBOSE,
)


def parse_whatsapp_chat(file_path: str) -> pd.DataFrame:
    """
    Parse a WhatsApp .txt export into a DataFrame.

    Handles Android (US), iOS (US), and iOS (EU/TR) export formats and
    correctly joins multi-line messages.
    """
    records = []
    current: dict | None = None

    with open(file_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            line = _STRIP.sub("", line) 

            m = _PATTERN.match(line)
            if m:
                if current:
                    records.append(current)

                date = m.group(1) or m.group(3) or m.group(5)
                time = m.group(2) or m.group(4) or m.group(6)
                sender = m.group(7).strip()
                message = m.group(8)
                current = {"Date": date, "Time": time, "Sender": sender, "Message": message}
            else:
                # Continuation line — append to previous message
                if current is not None:
                    current["Message"] += "\n" + line

    if current:
        records.append(current)

    return pd.DataFrame(records, columns=["Date", "Time", "Sender", "Message"])


if __name__ == "__main__":
    df = parse_whatsapp_chat("data/_chat_2.txt")
    print(df.head())
