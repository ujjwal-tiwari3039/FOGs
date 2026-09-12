import re
import pandas as pd

def parse_text(text):
    pairs = []
    current_q = ""
    current_a = []
    q_num = 1
    for line in text.split('\n'):
        match = re.match(r'^(\d+)\.\s*(.*)', line.strip())
        if match:
            if current_q:
                pairs.append({"#": q_num, "Question": current_q, "Answer": " ".join(current_a).strip()})
                q_num += 1
            current_q = match.group(2)
            current_a = []
        elif line.strip():
            current_a.append(line.strip())
    if current_q:
        pairs.append({"#": q_num, "Question": current_q, "Answer": " ".join(current_a).strip()})
    return pd.DataFrame(pairs)

print(parse_text("1. What is it?\nIt is a thing.\n2. Why?\nBecause."))
