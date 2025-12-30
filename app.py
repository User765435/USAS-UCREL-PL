from flask import Flask, request, jsonify, render_template_string
from ufal.morphodita import Tagger, Forms, TaggedLemmas, TokenRanges
import os
import csv
import re
import difflib

app = Flask(__name__)

# Lemma-to-tag dictionary from CSV
lemma_to_tag = {}
reflexive_verbs = {}
all_verb_lemmas = set()
multiword_to_tag = {}
multiword_entries = []  # will hold dicts with 'tokens', 'phrase', 'tag', plus lemma sequence
# Mapping from lemma to a dictionary of POS->tag values.  Used to handle
# homonymous lemmas (e.g. "koło" as a noun vs. preposition).  Keys are
# lemma strings, values are dictionaries mapping lower‑cased POS labels
# (like 'noun', 'preposition') to the tag string from the CSV.  Populated in
# load_tag_data().
lemma_pos_to_tag = {}
# Trie nodes for surface and lemma MWEs
class MWETreeNode:
    def __init__(self):
        self.children = {}
        self.entry = None  # holds entry when phrase ends here

mwe_trie_root_surface = MWETreeNode()
mwe_trie_root_lemma = MWETreeNode()

# Heuristic custom tag → POS
CUSTOM_TAG_TO_POS = {
    'N': 'noun',
    'B': 'noun',
    'X': 'noun',
    'S': 'noun',
    'F': 'noun',
    'G': 'noun',
    'W': 'noun',
    'Q': 'noun',
    'A': 'adjective',
    'O': 'preposition',
    'Z': 'preposition',
    'M': 'verb',
    'E': 'verb',
}

# Characters that should bind words together (e.g. H&M should be treated as a single unit)
SPECIAL_DELIMS = set("-'°&")

# Regular expression used to remove spaces around special delimiters in joined phrases
_delim_pattern = re.compile(r"\s*([" + re.escape("-'°&") + r"])\s*")

def split_phrase_to_tokens(phrase: str):
    """Split a phrase into tokens, keeping special delimiters as their own tokens.

    Whitespace is used to separate tokens. Any character in SPECIAL_DELIMS
    is treated as a separate token so that multi-word expressions containing
    characters like '-', '\'', '°', '&' can be matched against tokenized
    input where those characters may be separated.
    The input should already be lowered/normalized before splitting.
    """
    tokens = []
    current = ''
    for ch in phrase:
        if ch.isspace():
            if current:
                tokens.append(current)
                current = ''
        elif ch in SPECIAL_DELIMS:
            if current:
                tokens.append(current)
                current = ''
            tokens.append(ch)
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens

def join_form_tokens(tokens: list):
    """Join a sequence of tokens into a display string.

    Tokens are joined with spaces, but any spaces around special delimiters
    (e.g. '-', '\'', '°', '&') are removed so that forms like "H & M" are
    displayed as "H&M". This improves the readability of multi-word
    expressions containing such characters.
    """
    # First join with spaces
    phrase = ' '.join(tokens)
    # Then collapse spaces around special delimiters
    phrase = _delim_pattern.sub(lambda m: m.group(1), phrase)
    return phrase

def decode_pos(tag):
    if not tag:
        return ''
    first = tag[0].lower()
    return {
        'n': 'noun',
        'a': 'adjective',
        'p': 'pronoun',
        'v': 'verb',
        'd': 'adverb',
        'r': 'preposition',
        'c': 'conjunction',
        'i': 'interjection',
        't': 'particle',
        'x': 'unknown',
        'z': 'punctuation'
    }.get(first, 'unknown')

def split_custom_tags(tag_str):
    if not tag_str:
        return []
    parts = []
    for part in tag_str.strip().split():
        for sub in part.split('/'):
            parts.append(sub)
    return parts

def infer_custom_tag_pos(custom_tag_piece):
    if not custom_tag_piece:
        return ''
    prefix = custom_tag_piece[0].upper()
    return CUSTOM_TAG_TO_POS.get(prefix, '')

def choose_best_custom_tag(raw_custom_tag, decoded_pos):
    candidates = split_custom_tags(raw_custom_tag)
    if not candidates:
        return raw_custom_tag, []
    for cand in candidates:
        cand_pos = infer_custom_tag_pos(cand)
        if cand_pos and decoded_pos and cand_pos == decoded_pos:
            return cand, candidates
    return raw_custom_tag, candidates

# Helper functions for approximate lemma matching
def find_closest_lemma(lemma: str) -> str | None:
    """
    Attempt to find the closest sounding lemma from the known lemma list using
    simple string similarity. This implementation uses Python's difflib to
    compute close matches based on spelling similarity as a proxy for
    phonetic proximity. Only candidates with a similarity ratio above 0.8
    are considered.

    :param lemma: the lemma string for which to find a similar known lemma
    :return: the closest matching lemma or None if no suitable match is found
    """
    if not lemma:
        return None
    # Consider all known lemmas from the dictionary
    candidates = list(lemma_to_tag.keys())
    # Use difflib to find the best match
    matches = difflib.get_close_matches(lemma, candidates, n=1, cutoff=0.8)
    return matches[0] if matches else None

def get_custom_tag_for_lemma(lemma: str, pos: str, use_phonetic: bool):
    """
    Retrieve the best custom tag and all candidate tags for a given lemma.

    If an exact lemma is not present in the dictionary, optionally fall back
    to a similar lemma (approximate matching). When no suitable match is found
    the unknown tag ``Z99`` is returned as the best custom tag.  Returning
    ``Z99`` allows downstream logic to distinguish unknown tokens and ensure
    they do not influence high‑level category recommendations.

    :param lemma: the lemma for which to fetch a custom tag
    :param pos: the part of speech inferred for the lemma
    :param use_phonetic: if True, attempt to find a similar lemma via
                         approximate matching when no exact match exists
    :return: tuple (best_custom_tag, candidate_tags)
    """
    # Normalize POS string to lowercase for matching
    pos_lower = pos.lower() if pos else ''
    raw_custom = ''
    # Attempt to find a POS-specific tag for the lemma
    if lemma in lemma_pos_to_tag and pos_lower:
        raw_custom = lemma_pos_to_tag[lemma].get(pos_lower, '')
    # Fallback to the general lemma mapping if no POS-specific tag was found
    if not raw_custom:
        raw_custom = lemma_to_tag.get(lemma, '')
    # If still nothing and approximate matching is enabled, try to find a close lemma
    if not raw_custom and use_phonetic:
        closest = find_closest_lemma(lemma)
        if closest:
            if closest in lemma_pos_to_tag and pos_lower:
                raw_custom = lemma_pos_to_tag[closest].get(pos_lower, '')
            if not raw_custom:
                raw_custom = lemma_to_tag.get(closest, '')
    # If still nothing, return Z99 as the best tag and an empty candidate list
    if not raw_custom:
        return 'Z99', []
    # Finally choose the best candidate for this POS from within the raw tag string
    return choose_best_custom_tag(raw_custom, pos)

# Mapping from tag prefix letters to high-level domain categories. This mapping is
# used when recommending an overall category for a piece of text based on the
# distribution of tags associated with its tokens. The first letter of a
# tag (e.g. 'B' in 'B1' or 'G' in 'G2.1') indicates the broad semantic
# domain of the word. We group these prefixes into user-friendly categories.
TAG_CATEGORY_MAP = {
    'A': 'general & abstract terms',
    'B': 'medical & health',
    'C': 'arts & crafts',
    'E': 'emotional & psychological',
    'F': 'food & agriculture',
    'G': 'government & law',
    'H': 'housing & architecture',
    'I': 'finance & business',
    'K': 'entertainment & leisure',
    'L': 'life & living things',
    'M': 'movement & transport',
    'N': 'numbers & measurement',
    'O': 'objects & materials',
    'P': 'education',
    'Q': 'communication & media',
    'S': 'social & people',
    'T': 'time',
    'W': 'world & environment',
    'X': 'psychology & mental processes',
    'Y': 'science & technology',
    'Z': 'proper nouns & miscellaneous',
}

def compute_classification(results, ignore_prefixes={'A', 'N', 'Z', 'T', 'S', 'X'}):
    """
    Determine the two most salient high‑level semantic categories for a set of
    tagging results and provide frequency counts for every category. This
    function examines the primary `custom_tag` and all candidate tags for
    each token. Tags are grouped by their leading prefix letter (e.g. 'B'
    for medical-related terms). Prefixes included in `ignore_prefixes` are
    excluded from the final category selection but are still counted in the
    returned frequency tally.

    :param results: iterable of dictionaries with keys 'custom_tag' and
                    'all_custom_tags'
    :param ignore_prefixes: iterable of tag prefix letters to ignore when
                            determining the dominant categories. These
                            prefixes will never be returned as one of the
                            main categories. By default we exclude prefixes
                            ``'A'`` (general & abstract terms), ``'N'`` (numbers &
                            measurement), ``'Z'`` (proper nouns & miscellaneous),
                            ``'T'`` (time), ``'S'`` (social & people) and
                            ``'X'`` (psychology & mental processes), as
                            these categories often dominate the distribution
                            without being informative.
    :return: tuple (categories: list[str], category_counts: dict[str, int])
             where `categories` is always a list of two category names (the
             most frequent non-ignored categories; the same name may appear
             twice if only one distinct non-ignored prefix is present) and
             `category_counts` maps human‑readable category names to their
             aggregated counts.
    """
    # Tally counts by tag prefix
    prefix_counts: dict[str, int] = {}
    for res in results or []:
        # Count the main custom tag
        ct = res.get('custom_tag', '')
        # Skip unknown tags ('Z99') entirely so they do not influence category counts
        if ct and ct != 'Z99':
            pre = ct[0].upper()
            prefix_counts[pre] = prefix_counts.get(pre, 0) + 1
        # Count all candidate tags, ignoring Z99 candidates
        for cand in (res.get('all_custom_tags', []) or []):
            if not cand:
                continue
            cand_str = str(cand)
            # Skip unknown tags in candidates
            if not cand_str or cand_str == 'Z99':
                continue
            pre = cand_str[0].upper()
            prefix_counts[pre] = prefix_counts.get(pre, 0) + 1

    # Build category counts by mapping prefixes to category names
    category_counts: dict[str, int] = {}
    for pre, cnt in prefix_counts.items():
        name = TAG_CATEGORY_MAP.get(pre, TAG_CATEGORY_MAP.get('A', 'general & abstract terms'))
        category_counts[name] = category_counts.get(name, 0) + cnt

    # Determine the top two prefixes excluding ignored ones
    top_prefixes: list[str] = []
    if prefix_counts:
        # Sort prefixes by count, descending
        sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)
        # Filter out ignored prefixes for selection
        filtered = [p for p, _ in sorted_prefixes if p not in ignore_prefixes]
        if filtered:
            # There is at least one non-ignored prefix
            top_prefixes.append(filtered[0])
            if len(filtered) > 1:
                top_prefixes.append(filtered[1])
            else:
                # Only one non-ignored prefix; duplicate it to always have two
                top_prefixes.append(filtered[0])
        else:
            # If all prefixes are ignored, fall back to the most frequent overall
            if sorted_prefixes:
                first = sorted_prefixes[0][0]
                top_prefixes = [first, first]
    # Map selected prefixes to category names
    categories: list[str] = []
    for pre in top_prefixes:
        categories.append(TAG_CATEGORY_MAP.get(pre, TAG_CATEGORY_MAP.get('A', 'general & abstract terms')))
    return categories, category_counts

# Load tag data and multiword expressions
def load_tag_data():
    global mwe_trie_root_surface, mwe_trie_root_lemma
    try:
        with open('TagListPL.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'word' in row and 'tag' in row:
                    lemma = row['word']
                    tag = row['tag']
                    # Persist the last seen tag for this lemma in the general mapping
                    lemma_to_tag[lemma] = tag
                    # Attempt to detect a POS/homonym column; accept several possible header names
                    pos_val = ''
                    for key_name in ('POS', 'pos', 'homonins', 'homonim', 'homonym'):
                        if key_name in row and row[key_name]:
                            pos_val = row[key_name].strip()
                            break
                    if pos_val:
                        # Normalise POS to lowercase and record the POS-specific tag
                        lemma_pos_to_tag.setdefault(lemma, {})[pos_val.lower()] = tag
                    # Preserve existing behaviour for reflexive verbs and verb lemmas
                    if ' się' in lemma:
                        base_verb = lemma.replace(' się', '')
                        reflexive_verbs[base_verb] = lemma
                        all_verb_lemmas.add(lemma)
                        all_verb_lemmas.add(base_verb)
                    if tag.startswith('cz'):
                        all_verb_lemmas.add(lemma)
        print(f"Loaded {len(lemma_to_tag)} lemma-tag pairs from CSV")
        print(f"Identified {len(reflexive_verbs)} reflexive verbs")
        print(f"Identified {len(all_verb_lemmas)} verb lemmas")
        # Load multiword expressions
        multiword_path = 'Multi2.csv'
        if os.path.exists(multiword_path):
            with open(multiword_path, 'r', encoding='utf-8') as mw_file:
                mw_reader = csv.DictReader(mw_file)
                for mw_row in mw_reader:
                    phrase = mw_row.get('word', '')
                    tag_val = mw_row.get('tag', '')
                    if not phrase:
                        continue
                    # Consider phrases containing spaces or special delimiters as multi-word expressions
                    if (' ' in phrase) or any(ch in phrase for ch in SPECIAL_DELIMS):
                        # Lowercase for matching; original phrase preserved for display
                        lower_phrase = phrase.lower().strip()
                        # Split into tokens while keeping special delimiters separate
                        tokens = split_phrase_to_tokens(lower_phrase)
                        if len(tokens) > 1:
                            entry = {
                                'tokens': tokens,
                                'phrase': phrase,
                                'tag': tag_val,
                                'lemma_tokens': None  # to be filled later
                            }
                            multiword_entries.append(entry)
            # sort by length descending (number of tokens)
            multiword_entries.sort(key=lambda x: len(x['tokens']), reverse=True)
            print(f"Loaded {len(multiword_entries)} multi-word expressions from {multiword_path}")
        else:
            print(f"Multi-word file '{multiword_path}' not found; no multi-word expressions loaded")
    except Exception as e:
        print(f"Error loading CSV or MWE file: {e}")

    # Build lemma versions of MWEs using the tagger if available
    if tagger:
        for entry in multiword_entries:
            # run the tagger on the phrase to get lemmas
            forms = Forms()
            tagged_lemmas = TaggedLemmas()
            tokenizer = tagger.newTokenizer()
            tokenizer.setText(entry['phrase'])
            lemma_seq = []
            while tokenizer.nextSentence(forms, TokenRanges()):
                tagger.tag(forms, tagged_lemmas)
                for i in range(tagged_lemmas.size()):
                    lemma_seq.append(tagged_lemmas[i].lemma.lower())
            # Flatten lemma sequence by splitting each lemma on special delimiters
            if lemma_seq:
                lemma_tokens = []
                for lem in lemma_seq:
                    lemma_tokens.extend(split_phrase_to_tokens(lem))
                entry['lemma_tokens'] = lemma_tokens if lemma_tokens else entry['tokens']
            else:
                entry['lemma_tokens'] = entry['tokens']
    else:
        for entry in multiword_entries:
            # Fallback: use token-level splitting for lemma tokens
            lemma_tokens = []
            for tok in entry['tokens']:
                lemma_tokens.extend(split_phrase_to_tokens(tok))
            entry['lemma_tokens'] = lemma_tokens if lemma_tokens else entry['tokens']

    # Build tries for surface tokens and lemma tokens
    def build_trie(entries, root, key_field):
        for entry in entries:
            node = root
            for token in entry[key_field]:
                if token not in node.children:
                    node.children[token] = MWETreeNode()
                node = node.children[token]
            node.entry = entry

    mwe_trie_root_surface = MWETreeNode()
    mwe_trie_root_lemma = MWETreeNode()
    build_trie(multiword_entries, mwe_trie_root_surface, 'tokens')
    build_trie(multiword_entries, mwe_trie_root_lemma, 'lemma_tokens')

# Load tagger model
tagger_path = 'pl.tagger'
if not os.path.exists(tagger_path):
    print(f"WARNING: Tagger model file '{tagger_path}' not found!")

try:
    tagger = Tagger.load(tagger_path)
    if not tagger:
        print("Cannot load the tagger model!")
except Exception as e:
    print(f"Error loading tagger: {e}")
    tagger = None

# Initialize data
load_tag_data()

class ReflexiveVerbFinder:
    def __init__(self, tokens, lemmas, token_tags, use_phonetic=False):
        self.tokens = tokens
        self.lemmas = lemmas
        self.tags = token_tags
        self.results = []
        self.size = len(tokens)
        self.processed = [False] * self.size
        # Flag controlling whether approximate lemma matching should be used
        self.use_phonetic = use_phonetic

    def find_all_reflexive_verbs(self):
        self.find_adjacent_reflexive_verbs()
        self.find_reversed_reflexive_verbs()
        self.find_separated_reflexive_verbs()
        self.process_remaining_tokens()
        return self.results

    def find_adjacent_reflexive_verbs(self):
        for i in range(self.size - 1):
            if self.processed[i] or self.processed[i + 1]:
                continue
            if (self.lemmas[i] in all_verb_lemmas and self.tokens[i + 1].lower() == 'się'):
                reflexive_lemma = None
                if self.lemmas[i] in reflexive_verbs:
                    reflexive_lemma = reflexive_verbs[self.lemmas[i]]
                else:
                    potential_reflexive = f"{self.lemmas[i]} się"
                    if potential_reflexive in lemma_to_tag:
                        reflexive_lemma = potential_reflexive
                if reflexive_lemma:
                    self.add_reflexive_result(i, i + 1, reflexive_lemma, "Adjacent reflexive verb")
                    self.processed[i] = self.processed[i + 1] = True

    def find_reversed_reflexive_verbs(self):
        for i in range(self.size - 1):
            if self.processed[i] or self.processed[i + 1]:
                continue
            if self.tokens[i].lower() == 'się':
                next_lemma = self.lemmas[i + 1]
                reflexive_lemma = None
                if next_lemma in reflexive_verbs:
                    reflexive_lemma = reflexive_verbs[next_lemma]
                if not reflexive_lemma:
                    potential_reflexive = f"{next_lemma} się"
                    if potential_reflexive in lemma_to_tag:
                        reflexive_lemma = potential_reflexive
                if reflexive_lemma:
                    self.add_reflexive_result(i + 1, i, reflexive_lemma, "Reversed reflexive verb (się + verb)")
                    self.processed[i] = self.processed[i + 1] = True

    def find_separated_reflexive_verbs(self):
        verb_indices = []
        sie_indices = []
        for i in range(self.size):
            if self.processed[i]:
                continue
            if self.lemmas[i] in all_verb_lemmas:
                verb_indices.append(i)
            elif self.tokens[i].lower() == 'się':
                sie_indices.append(i)
        for verb_idx in verb_indices:
            if self.processed[verb_idx]:
                continue
            verb_lemma = self.lemmas[verb_idx]
            if verb_lemma not in reflexive_verbs:
                continue
            MAX_DISTANCE = 5
            closest_sie = None
            min_distance = MAX_DISTANCE + 1
            for sie_idx in sie_indices:
                if self.processed[sie_idx]:
                    continue
                distance = abs(verb_idx - sie_idx)
                if distance < min_distance and distance <= MAX_DISTANCE:
                    min_distance = distance
                    closest_sie = sie_idx
            if closest_sie is not None:
                reflexive_lemma = reflexive_verbs[verb_lemma]
                self.add_reflexive_result(
                    verb_idx, closest_sie, reflexive_lemma,
                    f"Separated reflexive verb (distance: {min_distance})"
                )
                self.processed[verb_idx] = self.processed[closest_sie] = True

    def add_reflexive_result(self, verb_idx, sie_idx, reflexive_lemma, note):
        verb_form = self.tokens[verb_idx]
        sie_form = self.tokens[sie_idx]
        if verb_idx < sie_idx:
            display_form = (f"{verb_form} ... {sie_form}" if sie_idx > verb_idx + 1 else f"{verb_form} {sie_form}")
        else:
            display_form = (f"{sie_form} ... {verb_form}" if verb_idx > sie_idx + 1 else f"{sie_form} {verb_form}")
        morpho = self.tags[verb_idx] if verb_idx < len(self.tags) else ''
        pos = decode_pos(morpho)
        # Determine the best custom tag for the reflexive lemma, using
        # approximate matching if enabled
        best_custom, all_cands = get_custom_tag_for_lemma(reflexive_lemma, pos, self.use_phonetic)
        self.results.append({
            "form": display_form,
            "lemma": reflexive_lemma,
            "pos": pos,
            "custom_tag": best_custom,
            "all_custom_tags": all_cands,
            "morpho_tag": morpho,
            "note": note,
            "position": min(verb_idx, sie_idx)
        })

    def process_remaining_tokens(self):
        for i in range(self.size):
            if not self.processed[i]:
                form = self.tokens[i]
                lemma = self.lemmas[i]
                morpho = self.tags[i]
                pos = decode_pos(morpho)
                # Determine the best custom tag for the lemma, using
                # approximate matching if enabled
                best_custom, all_cands = get_custom_tag_for_lemma(lemma, pos, self.use_phonetic)
                self.results.append({
                    "form": form,
                    "lemma": lemma,
                    "pos": pos,
                    "custom_tag": best_custom,
                    "all_custom_tags": all_cands,
                    "morpho_tag": morpho,
                    "note": '',
                    "position": i
                })
                self.processed[i] = True
        self.results.sort(key=lambda x: x.get('position', 0))
        for result in self.results:
            if "position" in result:
                del result["position"]

# HTML template
homepage = '''
<!DOCTYPE html>
<html>
<head>
    <title>Polish Word Tagger</title>
    <style>
        body { font-family: Arial; padding: 20px; line-height: 1.5; }
        .word-info { border-bottom: 1px solid #ccc; padding: 8px; margin-bottom: 6px; }
        .label { font-weight: bold; }
        .note { font-style: italic; color: #006600; }
        .badge { background: #eee; padding: 2px 6px; border-radius: 4px; margin-left: 4px; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>Polish Word Tagger (with MSE/MWE lemma-aware matching)</h1>
    <form id="form">
        <textarea id="sentence" rows="4" cols="70" placeholder="Wpisz zdanie po polsku..."></textarea><br>
        <label><input type="checkbox" id="phoneticToggle"> Use approximate lemma search</label><br>
        <button type="submit">Analyze</button>
    </form>
    <div id="output"></div>
    <script>
        document.getElementById('form').addEventListener('submit', async e => {
            e.preventDefault();
            const sentence = document.getElementById('sentence').value;
            const out = document.getElementById('output');
            out.innerHTML = '<p>Processing...</p>';
            try {
                const res = await fetch('/tag', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({
                        sentence,
                        phonetic: document.getElementById('phoneticToggle').checked
                    })
                });
                const data = await res.json();
                out.innerHTML = '';
                // If category information is provided, display it at the top
                if (data.categories && data.categories.length) {
                    let catHtml = '<div class="word-info">';
                    catHtml += `<div><span class="label">Main categories:</span> ${data.categories.join(', ')}</div>`;
                    if (data.category_counts) {
                        const pairs = [];
                        for (const [name, cnt] of Object.entries(data.category_counts)) {
                            pairs.push(`${name}: ${cnt}`);
                        }
                        if (pairs.length) {
                            catHtml += `<div><span class="label">Tag frequencies:</span> ${pairs.join(', ')}</div>`;
                        }
                    }
                    catHtml += '</div>';
                    out.innerHTML += catHtml;
                }
                if (data.results && data.results.length) {
                    data.results.forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'word-info';
                        let html = `<div><span class="label">Form:</span> ${item.form}</div>`;
                        if (item.lemma) html += `<div><span class="label">Lemma:</span> ${item.lemma}</div>`;
                        if (item.pos) html += `<div><span class="label">POS:</span> ${item.pos}</div>`;
                        if (item.custom_tag) html += `<div><span class="label">Custom tag:</span> ${item.custom_tag}</div>`;
                        if (item.all_custom_tags && item.all_custom_tags.length>1) {
                            html += `<div><span class="label">All candidates:</span> ${item.all_custom_tags.join(', ')}</div>`;
                        }
                        if (item.morpho_tag) html += `<div><span class="label">Full tag:</span> ${item.morpho_tag}</div>`;
                        if (item.note) html += `<div class="note">${item.note}</div>`;
                        div.innerHTML = html;
                        out.appendChild(div);
                    });
                } else if (data.error) {
                    out.innerHTML = `<div style="color:red;">Error: ${data.error}</div>`;
                } else {
                    out.innerHTML = '<div>No results</div>';
                }
            } catch (err) {
                out.innerHTML = `<div style="color:red;">${err.message}</div>`;
            }
        });
    </script>
</body>
</html>
'''

def find_mwes(tokens_lower):
    """Greedy longest-first match on surface forms and lemma forms, return list of matches."""
    n = len(tokens_lower)
    used = [False]*n
    results = []
    i = 0
    # First try surface-form matches
    while i < n:
        node = mwe_trie_root_surface
        j = i
        last_match = None
        while j < n and tokens_lower[j] in node.children:
            node = node.children[tokens_lower[j]]
            if node.entry:
                last_match = (j+1, node.entry)
            j += 1
        if last_match:
            end_idx, entry = last_match
            results.append({"start": i, "end": end_idx, "entry": entry})
            for k in range(i, end_idx):
                used[k] = True
            i = end_idx
        else:
            i += 1
    # Then attempt lemma-based matches on spans not yet used
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue
        # build lemma window (we don't have lemmas here; caller must supply)
        node = mwe_trie_root_lemma
        j = i
        last_match = None
        while j < n and not used[j] and False:  # placeholder: lemma-based matching done separately below
            # this part is handled after tokenization since we need the actual lemma sequence
            break
        i += 1
    return results  # lemma matches integrated later in the tagging function

@app.route('/')
def home():
    return render_template_string(homepage)

@app.route('/tag', methods=['POST'])
def tag_sentence():
    data = request.get_json()
    sentence = data.get('sentence', '')
    # Whether to enable approximate (phonetic) lemma matching
    use_phonetic = bool(data.get('phonetic'))
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    results = []

    # Special case
    if sentence.strip().lower() == "się boi":
        # Handle the special reflexive pattern directly
        reflexive_form = "bać się"
        pos = "verb"
        best_custom, all_cands = get_custom_tag_for_lemma(reflexive_form, pos, use_phonetic)
        single_result = {
            "form": "się boi",
            "lemma": reflexive_form,
            "pos": pos,
            "custom_tag": best_custom,
            "all_custom_tags": all_cands,
            "morpho_tag": '',
            "note": "Reflexive verb (special case)"
        }
        categories, category_counts = compute_classification([single_result])
        return jsonify({
            "results": [single_result],
            "categories": categories,
            "category_counts": category_counts
        })

    if tagger:
        forms = Forms()
        tagged_lemmas = TaggedLemmas()
        tokens_ranges = TokenRanges()
        tokenizer = tagger.newTokenizer()
        if not tokenizer:
            return jsonify({"error": "Could not create tokenizer"}), 500
        tokenizer.setText(sentence)
        while tokenizer.nextSentence(forms, tokens_ranges):
            tokens = [forms[i] for i in range(forms.size())]
            tagger.tag(forms, tagged_lemmas)
            lemmas = [tagged_lemmas[i].lemma.lower() for i in range(tagged_lemmas.size())]
            token_tags = [tagged_lemmas[i].tag for i in range(tagged_lemmas.size())]
            lower_tokens = [tok.lower() for tok in tokens]

            skip_indices = set()
            multiword_results = []

            # First: surface-form greedy MWE match
            surface_matches = []
            i = 0
            n = len(lower_tokens)
            while i < n:
                node = mwe_trie_root_surface
                j = i
                last_match = None
                while j < n and lower_tokens[j] in node.children:
                    node = node.children[lower_tokens[j]]
                    if node.entry:
                        last_match = (j+1, node.entry)
                    j += 1
                if last_match:
                    end_idx, entry = last_match
                    surface_matches.append({"start": i, "end": end_idx, "entry": entry, "type": "surface"})
                    for k in range(i, end_idx):
                        skip_indices.add(k)
                    i = end_idx
                else:
                    i += 1

            # Second: lemma-based greedy MWE match on remaining spans
            lemma_matches = []
            i = 0
            while i < len(lemmas):
                if i in skip_indices:
                    i += 1
                    continue
                node = mwe_trie_root_lemma
                j = i
                last_match = None
                while j < len(lemmas) and lemmas[j] in node.children:
                    node = node.children[lemmas[j]]
                    if node.entry:
                        last_match = (j+1, node.entry)
                    j += 1
                if last_match:
                    end_idx, entry = last_match
                    lemma_matches.append({"start": i, "end": end_idx, "entry": entry, "type": "lemma"})
                    for k in range(i, end_idx):
                        skip_indices.add(k)
                    i = end_idx
                else:
                    i += 1

            # Combine MWE results (surface first, then lemma; avoid duplicates)
            for match in surface_matches + lemma_matches:
                i0 = match['start']
                end = match['end']
                entry = match['entry']
                mtag = token_tags[i0] if i0 < len(token_tags) else ''
                pos = decode_pos(mtag)
                raw_custom = entry.get('tag', '')
                best_custom, all_cands = choose_best_custom_tag(raw_custom, pos)
                # Produce a display form that removes spaces around special delimiters
                display_form = join_form_tokens(tokens[i0:end])
                multiword_results.append({
                    "form": display_form,
                    "lemma": entry['phrase'],
                    "pos": pos,
                    "custom_tag": best_custom,
                    "all_custom_tags": all_cands,
                    "morpho_tag": mtag,
                    "note": "Multi-word expression",
                    "position": i0
                })

            # Filter tokens not covered by MWEs
            filtered_tokens = []
            filtered_lemmas = []
            filtered_tags = []
            orig_index_for_filtered = []
            for idx, (tok, lem, t_tag) in enumerate(zip(tokens, lemmas, token_tags)):
                if idx not in skip_indices:
                    filtered_tokens.append(tok)
                    filtered_lemmas.append(lem)
                    filtered_tags.append(t_tag)
                    orig_index_for_filtered.append(idx)

            # Special reflexive pattern: "się verb"
            if len(filtered_tokens) >= 2 and filtered_tokens[0].lower() == 'się':
                verb_lemma = filtered_lemmas[1]
                if verb_lemma in reflexive_verbs:
                    reflexive_form = reflexive_verbs[verb_lemma]
                    pos = decode_pos(filtered_tags[1])
                    # Determine best custom tag for the reflexive form using optional phonetic search
                    best_custom, all_cands = get_custom_tag_for_lemma(reflexive_form, pos, use_phonetic)
                    results.append({
                        "form": f"{filtered_tokens[0]} {filtered_tokens[1]}",
                        "lemma": reflexive_form,
                        "pos": pos,
                        "custom_tag": best_custom,
                        "all_custom_tags": all_cands,
                        "morpho_tag": filtered_tags[1],
                        "note": "Reflexive verb (się + verb pattern)",
                        "position": orig_index_for_filtered[0]
                    })
                    for j in range(2, len(filtered_tokens)):
                        orig_idx = orig_index_for_filtered[j]
                        mtag = token_tags[orig_idx]
                        pos_j = decode_pos(mtag)
                        # Determine best custom tag for each remaining token using optional phonetic search
                        best_custom_j, all_cands_j = get_custom_tag_for_lemma(filtered_lemmas[j], pos_j, use_phonetic)
                        results.append({
                            "form": tokens[orig_idx],
                            "lemma": filtered_lemmas[j],
                            "pos": pos_j,
                            "custom_tag": best_custom_j,
                            "all_custom_tags": all_cands_j,
                            "morpho_tag": mtag,
                            "note": '',
                            "position": orig_index_for_filtered[j]
                        })
                    # Incorporate any multiword results and sort
                    results.extend(multiword_results)
                    results.sort(key=lambda x: x.get('position', 0))
                    for res in results:
                        if 'position' in res:
                            del res['position']
                    # Compute category recommendations for the assembled results
                    categories, category_counts = compute_classification(results)
                    return jsonify({
                        "results": results,
                        "categories": categories,
                        "category_counts": category_counts
                    })

            # Regular processing with reflexive finder on filtered
            finder = ReflexiveVerbFinder(filtered_tokens, filtered_lemmas, filtered_tags, use_phonetic)
            sentence_results = finder.find_all_reflexive_verbs()
            updated_sentence_results = []
            for res in sentence_results:
                if 'position' in res:
                    pos_idx = res['position']
                    if pos_idx < len(orig_index_for_filtered):
                        res['position'] = orig_index_for_filtered[pos_idx]
                updated_sentence_results.append(res)

            results.extend(multiword_results)
            results.extend(updated_sentence_results)
    else:
        # Fallback if no tagger
        words = sentence.split()
        lower_words = [w.lower() for w in words]
        skip_indices = set()
        multiword_results = []
        # Simple surface MWE match
        i = 0
        while i < len(lower_words):
            node = mwe_trie_root_surface
            j = i
            last_match = None
            while j < len(lower_words) and lower_words[j] in node.children:
                node = node.children[lower_words[j]]
                if node.entry:
                    last_match = (j+1, node.entry)
                j += 1
            if last_match:
                end_idx, entry = last_match
                raw_custom = entry.get('tag', '')
                best_custom, all_cands = choose_best_custom_tag(raw_custom, '')
                # Join words for display and collapse spaces around special delimiters
                display_form = join_form_tokens(words[i:end_idx])
                multiword_results.append({
                    "form": display_form,
                    "lemma": entry['phrase'],
                    "pos": '',
                    "custom_tag": best_custom,
                    "all_custom_tags": all_cands,
                    "morpho_tag": '',
                    "note": "Multi-word expression",
                    "position": i
                })
                for k in range(i, end_idx):
                    skip_indices.add(k)
                i = end_idx
            else:
                i += 1
        for idx, word in enumerate(words):
            if idx in skip_indices:
                continue
            # Determine the best custom tag for the raw word. In the absence of
            # a lemma (no tagger), we treat the word itself as a lemma and
            # optionally perform approximate matching if enabled.
            best_custom, all_cands = get_custom_tag_for_lemma(word, '', use_phonetic)
            results.append({
                "form": word,
                "lemma": "",
                "pos": '',
                "custom_tag": best_custom,
                "all_custom_tags": all_cands,
                "morpho_tag": '',
                "note": '',
                "position": idx
            })
        results.extend(multiword_results)

    # Final sort and cleanup
    results.sort(key=lambda x: x.get('position', 0))
    for res in results:
        if 'position' in res:
            del res['position']
    # Compute the overall category recommendations and counts
    categories, category_counts = compute_classification(results)
    return jsonify({
        "results": results,
        "categories": categories,
        "category_counts": category_counts
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Polish Word Tagger server (lemma-aware MSE/MWE handling)")
    print(f"Loaded {len(lemma_to_tag)} lemma-tag pairs from CSV")
    print(f"Identified {len(reflexive_verbs)} reflexive verbs")
    print(f"Identified {len(all_verb_lemmas)} verb lemmas")
    print("Access the application at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0')


































