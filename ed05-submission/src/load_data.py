"""
XML data loader for SemEval 2013 Task 7 (3-way classification).
Loads training and test data from the official repo structure.
HARD CONSTRAINT: This module only loads training/3way/**. Test data is loaded
by a separate function used ONLY in local_eval.py.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# Default base path to the extracted semeval-3way data
DEFAULT_BASE = Path(__file__).parent.parent / "data" / "semeval-2013-task7" / "semeval-3way"

LABEL_CLASSES = ['correct', 'contradictory', 'incorrect']


def _parse_xml_file(xml_path: Path, domain: str, condition: str = None):
    """Parse a single XML question file and yield one record per student answer."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    question_id = root.get('id', xml_path.stem)
    question_text_el = root.find('questionText')
    question_text = question_text_el.text.strip() if question_text_el is not None and question_text_el.text else ""

    # Collect ALL reference answers (regardless of category attribute)
    ref_answers = []
    ref_answers_el = root.find('referenceAnswers')
    if ref_answers_el is not None:
        for ref in ref_answers_el.findall('referenceAnswer'):
            if ref.text and ref.text.strip():
                ref_answers.append(ref.text.strip())

    # Parse each student answer
    student_answers_el = root.find('studentAnswers')
    if student_answers_el is None:
        return

    for sa in student_answers_el.findall('studentAnswer'):
        label = sa.get('accuracy', '').lower().strip()
        sa_id = sa.get('id', '')
        sa_text = sa.text.strip() if sa.text else ""

        if not sa_text or label not in LABEL_CLASSES:
            continue

        record = {
            'question_id': question_id,
            'question_text': question_text,
            'reference_answers': ref_answers,  # list of strings, may be empty
            'student_answer': sa_text,
            'label': label,
            'student_answer_id': sa_id,
            'domain': domain,
        }
        if condition is not None:
            record['condition'] = condition
        yield record


def load_training_data(base_path=None):
    """Load all training/3way data. Returns list of dicts."""
    base = Path(base_path) if base_path else DEFAULT_BASE
    train_dir = base / "training" / "3way"
    data = []

    for domain in ['beetle', 'sciEntsBank']:
        domain_dir = train_dir / domain
        if not domain_dir.exists():
            print(f"WARNING: {domain_dir} not found")
            continue
        for xml_file in sorted(domain_dir.glob("*.xml")):
            for record in _parse_xml_file(xml_file, domain):
                data.append(record)

    return data


def load_test_data(base_path=None):
    """Load all test/3way data with condition labels. Returns list of dicts.
    WARNING: This function is ONLY to be called from local_eval.py.
    Its output must NEVER feed back into training, features, or thresholds.
    """
    base = Path(base_path) if base_path else DEFAULT_BASE
    test_dir = base / "test" / "3way"
    data = []

    test_conditions = {
        'beetle': ['test-unseen-answers', 'test-unseen-questions'],
        'sciEntsBank': ['test-unseen-answers', 'test-unseen-questions', 'test-unseen-domains'],
    }

    for domain, conditions in test_conditions.items():
        for condition in conditions:
            cond_dir = test_dir / domain / condition
            if not cond_dir.exists():
                print(f"WARNING: {cond_dir} not found")
                continue
            for xml_file in sorted(cond_dir.glob("*.xml")):
                for record in _parse_xml_file(xml_file, domain, condition):
                    data.append(record)

    return data


def make_train_val_split(data, val_fraction=0.2, random_seed=42):
    """Stratified split by label. Returns (train_data, val_data)."""
    labels = [d['label'] for d in data]
    train_data, val_data = train_test_split(
        data, test_size=val_fraction, random_state=random_seed,
        stratify=labels
    )
    return list(train_data), list(val_data)


def print_label_stats(data, name='Dataset'):
    """Print label distribution and domain breakdown."""
    print(f"\n{'='*50}")
    print(f"  {name}: {len(data)} total student answers")
    print(f"{'='*50}")

    label_counts = Counter(d['label'] for d in data)
    total = len(data)
    for label in LABEL_CLASSES:
        c = label_counts.get(label, 0)
        pct = 100 * c / total if total > 0 else 0
        print(f"  {label:15s}: {c:5d}  ({pct:5.1f}%)")

    domain_counts = Counter(d['domain'] for d in data)
    print(f"  {'---':15s}")
    for domain, c in sorted(domain_counts.items()):
        print(f"  {domain:15s}: {c:5d}")

    # Reference answer stats
    ref_counts = [len(d['reference_answers']) for d in data]
    no_ref = sum(1 for r in ref_counts if r == 0)
    multi_ref_qs = set()
    for d in data:
        if len(d['reference_answers']) > 1:
            multi_ref_qs.add(d['question_id'])
    print(f"  {'---':15s}")
    print(f"  No ref answers: {no_ref}")
    print(f"  Questions w/ multiple refs: {len(multi_ref_qs)}")
    if ref_counts:
        print(f"  Max refs per question: {max(ref_counts)}")


if __name__ == '__main__':
    print("Loading training data...")
    train_data = load_training_data()
    print_label_stats(train_data, "Training (full)")

    print("\nMaking train/val split...")
    train_split, val_split = make_train_val_split(train_data)
    print_label_stats(train_split, "Train split")
    print_label_stats(val_split, "Val split")

    print("\nLoading test data...")
    test_data = load_test_data()
    print_label_stats(test_data, "Test (all conditions)")

    # Per-condition breakdown
    conditions = set(d.get('condition', 'unknown') for d in test_data)
    for cond in sorted(conditions):
        subset = [d for d in test_data if d.get('condition') == cond]
        print_label_stats(subset, f"Test: {cond}")
