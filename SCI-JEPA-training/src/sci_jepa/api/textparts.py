"""
Text side of the API (no model): the task / part tables, the splitter that turns a few sentences
into parts, and the content and safety checks of the specification (section 7b / 7c).

THE SPLITTER IS A PLACEHOLDER. The specification asks for a language model that splits text "like
the training data"; until it exists, each sentence is labelled by cue words (we propose, dataset,
accuracy rose, ...). It is deterministic and cheap, and `understood` always shows what it read, so a
wrong split can be corrected through POST /v1/sessions/{id}/answers.

Thresholds (min_words, 0.6 confidence, 30 % for edits) are the specification's starting values,
except problem.min_words = 6 instead of 8: the specification's own examples ("Image classifiers
overfit on small medical datasets.", 7 words) must pass. Section 11's experiment sets the final ones.
"""
import difflib
import re
import unicodedata

# ---------------------------------------------------------------- tables (GET /v1/tasks)
TASKS = {
    "forward": {"question": "What results can you expect?", "requires": ["problem", "method", "conditions"],
                "answers": ["result"]},
    "inverse": {"question": "Which method could lead to this result?", "requires": ["problem", "result"],
                "answers": ["method", "procedure"]},
    "abduction": {"question": "What could explain these results?", "requires": ["result", "conditions"],
                  "answers": ["claim", "question"]},
    "consistency": {"question": "Is this claim supported by the evidence?", "requires": ["claim", "evidence"],
                    "answers": ["evidence"]},
    "edit_check": {"question": "Does the model tell a finding from an edited copy?", "requires": ["original", "edited"],
                   "answers": []},
    "intervention": {"question": "How would the result change with another method or setup?",
                     "requires": ["problem", "method", "conditions", "change"], "answers": ["result"]},
}
TASK_ORDER = ("forward", "inverse", "abduction", "consistency")     # what /v1/ask can pick from free text
RANKING_TASKS = ("forward", "inverse", "abduction")

PARTS = {   # part -> KU types it becomes, and its limits
    "problem": {"ku_types": ["context", "question"], "min_words": 6, "max_chars": 1500},
    "method": {"ku_types": ["method"], "min_words": 5, "max_chars": 1500},
    "conditions": {"ku_types": ["procedure"], "min_words": 5, "max_chars": 1500},
    "result": {"ku_types": ["result"], "min_words": 6, "max_chars": 1500},
    "claim": {"ku_types": ["claim"], "min_words": 6, "max_chars": 1000},
    "evidence": {"ku_types": ["evidence"], "min_words": 8, "max_chars": 1500},
}
EXTRA_PARTS = {   # given only as follow-up answers (not found in free text)
    "original": {"min_words": 3, "max_chars": 1500}, "edited": {"min_words": 3, "max_chars": 1500},
    "change": {"min_words": 3, "max_chars": 1500},
}
LOW_CONFIDENCE = 0.6
MAX_EDIT_SHARE = 0.30

QUESTIONS = {   # part -> (follow-up question, example)
    "problem": ("What problem or question are you working on?",
                "Predicting reaction yields from written lab protocols."),
    "method": ("Which method or approach do you use?",
               "Fine-tuning a small language model on the protocol text."),
    "conditions": ("Which data or setup do you use?",
                   "About 50,000 Suzuki coupling protocols from US patents, 80/20 split."),
    "result": ("What result did you get?",
               "Top-1 accuracy rose from 71.2% to 76.8% over a ResNet-50 baseline on ImageNet."),
    "claim": ("Which claim do you want to check?", "Dropout reduces overfitting in small transformers."),
    "evidence": ("Which evidence do you have for or against the claim?",
                 "On 5 datasets, validation loss fell by 12% with dropout 0.1 versus none."),
    "original": ("Which finding do you want to compare?", "Adding dropout reduced overfitting by 12%."),
    "edited": ("What is the edited copy of the finding?", "Adding dropout increased overfitting by 12%."),
    "change": ("What would you change: a different method or a different setup?",
               "Training a graph neural network on reaction graphs."),
}
VAGUE_QUESTIONS = {
    "result": "What changed, by how much, and compared with what?",
    "problem": "What exactly is the problem: which quantity, system or question?",
    "method": "What exactly did you do: which model, algorithm or technique?",
    "conditions": "Which data, how much of it, and which setup?",
    "claim": "What exactly do you claim, and about what?",
    "evidence": "What did you measure, on what, and what came out?",
}

# ---------------------------------------------------------------- words
STOP = set("""a an the and or but if then than of in on at to for from by with without about into over under
between through during before after as is are was were be been being am do does did done have has had having
we our us i my me you your they their them it its this that these those there here which who whom whose what
when where why how not no nor so too very can could may might must shall should will would also just only
some any each every all both few more most other such own same one two using use used via per""".split())
GENERIC = set("""experiment experiments experimental result results method methods data study studies work works
paper papers thing things stuff approach approaches analysis way ways something lot lots good bad better
interesting new novel test tests tested testing try tried model models get got make made""".split())
EN_FUNCTION = set("the and of to in is that for with on as by are was be this from at an or which we our".split())
_DIRECTION = re.compile(r"\b(increas\w*|decreas\w*|improv\w*|reduc\w*|rais\w*|rose|rise[sn]?|fell|fall\w*|drop\w*|"
                        r"higher|lower|better|worse|faster|slower|outperform\w*|exceed\w*|gain\w*|loss(?:es)?|"
                        r"boost\w*|degrad\w*|doubl\w*|halv\w*|cut|grew|grow\w*|shrank|declin\w*)\b", re.I)
_COMPARE = re.compile(r"\b(than|compared|versus|vs\.?|relative to|baseline|over (?:a|the)|against)\b", re.I)
_NUMBER = re.compile(r"\d")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-']*|\d+(?:[.,]\d+)*%?")


def words(text):
    return _WORD.findall(text or "")


def content_words(text):
    return [w for w in words(text) if w.lower() not in STOP and w.lower() not in GENERIC]


# ---------------------------------------------------------------- normalisation and safety
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]\d{3,4}(?:[\s-]\d{2,4})?(?!\w)")


def normalise(text):
    """NFC, control characters removed, whitespace runs collapsed (newlines kept)."""
    t = _CTRL.sub(" ", unicodedata.normalize("NFC", text or ""))
    return re.sub(r"[ \t]+", " ", t).strip()


def scrub_pii(text):
    """Remove emails and phone numbers -> (text, removed?). Names are not detected yet."""
    t = _EMAIL.sub("[email removed]", text)
    t = _PHONE.sub(lambda m: "[phone removed]" if sum(c.isdigit() for c in m.group(0)) >= 9 else m.group(0), t)
    return t, t != text


def is_english(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isascii() for c in letters) / len(letters) < 0.85:
        return False
    w = [x.lower() for x in words(text)]
    return len(w) < 6 or sum(x in EN_FUNCTION for x in w) / len(w) >= 0.06


def is_gibberish(text):
    w = words(text)
    letters = [c.lower() for c in text if c.isalpha()]
    if not w or not letters:
        return True
    if re.search(r"(.)\1{5,}", text):                                    # aaaaaa, !!!!!!
        return True
    if sum(c in "aeiouy" for c in letters) / len(letters) < 0.2:         # no vowels: keyboard mash
        return True
    if sum(len(x) for x in w) / len(w) > 15:
        return True
    return len(w) >= 10 and len({x.lower() for x in w}) / len(w) < 0.3   # the same few words repeated


def sentences(text):
    """-> [(start, end, sentence)] with character offsets into text."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{1,}", text):
        seg = text[start:m.start()]
        if seg.strip():
            a = start + len(seg) - len(seg.lstrip())
            out.append((a, a + len(seg.strip()), seg.strip()))
        start = m.end()
    seg = text[start:]
    if seg.strip():
        a = start + len(seg) - len(seg.lstrip())
        out.append((a, a + len(seg.strip()), seg.strip()))
    return out


# ---------------------------------------------------------------- splitter (placeholder)
CUES = {
    "problem": [r"\bwe (?:want|aim|seek|try|ask|wish|need) to\b", r"\b(?:our|the) (?:goal|aim|objective|question)\b",
                r"\bproblem\b", r"\bchallenge\b", r"\bwhether\b", r"\bhow (?:to|can|do|does)\b", r"\bhypothes\w*",
                r"\bmotivat\w*", r"\b(?:is|are|remains?) (?:difficult|hard|unknown|unclear|open|costly|expensive)\b",
                r"\blittle is known\b", r"\?\s*$", r"\bpredict(?:ing)?\b.*\bfrom\b", r"\bwe study\b", r"\bfocus on\b"],
    "method": [r"\bwe (?:propose|use|train|fine-?tune|apply|develop|introduce|design|build|employ|adopt|combine|"
               r"implement|model|learn|pretrain|pre-train|optimi[sz]e|simulate|derive|compute|estimate|fit)\b",
               r"\bour (?:method|approach|model|framework|algorithm|technique|system)\b", r"\busing (?:a|an|the)\b",
               r"\bbased on\b", r"\balgorithm\b", r"\barchitecture\b", r"\bframework\b", r"\bnetwork\b",
               r"\bregression\b", r"\btransformer\b", r"\blanguage model\b", r"\bfine-?tun\w*"],
    "conditions": [r"\bdata ?sets?\b", r"\bdata\b", r"\bsamples?\b", r"\bsplit\b", r"\bbenchmarks?\b", r"\bsetup\b",
                   r"\bcollected\b", r"\bparticipants\b", r"\bcohort\b", r"\bpatients\b", r"\bprotocols\b",
                   r"\b\d[\d,.]*\s*(?:k\b|thousand|million)?\s*(?:examples|images|papers|patients|samples|protocols|"
                   r"reactions|molecules|documents|sentences|subjects|runs|trials|cells|galaxies|events)\b",
                   r"\b\d+/\d+\b", r"\bepochs?\b", r"\bbatch size\b", r"\bhyper-?parameters?\b", r"\btemperature\b",
                   r"\bGPUs?\b", r"\bfrom (?:US )?patents\b", r"\bheld-out\b", r"\btraining set\b", r"\btest set\b"],
    "result": [r"\bwe (?:found|find|observe[d]?|obtain(?:ed)?|achiev\w*|reach\w*|measur\w*|got)\b",
               r"\b(?:results? show|reached|achieved|obtained)\b", r"\baccuracy\b", r"\berror\b", r"\bF1\b",
               r"\b(?:MAE|RMSE|AUC|BLEU|MRR)\b", r"\d+(?:\.\d+)?\s*%", r"\bsignificant\w*", r"\bp\s*[<=]",
               r"\boutperform\w*", r"\b(?:rose|fell|improv\w*|increas\w*|decreas\w*|reduc\w*)\b.*\d"],
    "claim": [r"\bwe (?:claim|argue|conclude|suggest|believe|posit|demonstrate)\b", r"\bthis (?:shows|suggests|"
              r"indicates|implies|proves)\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bcauses?\b",
              r"\b(?:reduces|increases|improves|prevents|leads to)\b(?!.*\d)"],
    "evidence": [r"\bevidence\b", r"\bsupport(?:s|ed|ing)?\b", r"\bin (?:\d+|all|five|three|several) "
                 r"(?:datasets|experiments|trials|studies|settings)\b", r"\b(?:table|figure|fig\.)\s*\d",
                 r"\bon \d+ (?:datasets|tasks|benchmarks)\b", r"\b(?:fell|rose) by\b"],
}
_CUES = {p: [re.compile(c, re.I) for c in cs] for p, cs in CUES.items()}


def _score(sent):
    return {p: sum(1 for c in cs if c.search(sent)) for p, cs in _CUES.items()}


def split_parts(text, goal=None):
    """Label each sentence with the part it most likely is -> [{"part", "text", "confidence", "span", "spans"}].
    goal: the task, if known; evidence and result are swapped to the one the task needs."""
    need = set(TASKS[goal]["requires"]) if goal in TASKS else set()
    labelled, prev = [], None
    for a, b, s in sentences(text):
        sc = _score(s)
        if goal == "consistency":                        # claim + evidence: a measured result is evidence
            sc["evidence"] += sc.pop("result"); sc["result"] = 0
        elif need and "result" in need and "evidence" not in need:
            sc["result"] += sc.pop("evidence"); sc["evidence"] = 0
        if need:                                         # parts the task does not use lose ties
            for p in sc:
                if p not in need and sc[p]:
                    sc[p] -= 0.5
        best = max(sc, key=lambda p: (sc[p], -list(CUES).index(p)))
        tot = sum(max(v, 0) for v in sc.values())
        if sc[best] <= 0:                                # no cue: continues the previous sentence's part
            best = prev or ("problem" if "problem" in need or not need else sorted(need)[0])
            conf = 0.45
        else:
            conf = 0.5 + 0.5 * (sc[best] / tot) * min(1.0, sc[best] / 2)
        labelled.append((best, round(conf, 2), a, b, s))
        prev = best
    parts = {}
    for p, c, a, b, s in labelled:                       # one entry per part, sentences joined in order
        e = parts.setdefault(p, {"part": p, "text": [], "confidence": [], "spans": []})
        e["text"].append(s); e["confidence"].append(c); e["spans"].append([a, b])
    out = []
    for p in PARTS:
        if p in parts:
            e = parts[p]
            out.append({"part": p, "text": " ".join(e["text"]), "confidence": round(min(e["confidence"]), 2),
                        "span": [e["spans"][0][0], e["spans"][-1][1]], "spans": e["spans"]})
    return out


def pick_task(parts):
    """The first task (forward, inverse, abduction, consistency) whose required parts are all there;
    else the one with the fewest missing."""
    have = set(parts)
    miss = {t: [p for p in TASKS[t]["requires"] if p not in have] for t in TASK_ORDER}
    return min(TASK_ORDER, key=lambda t: (len(miss[t]), TASK_ORDER.index(t)))


# ---------------------------------------------------------------- content checks
def _question(part, vague=False):
    q, ex = QUESTIONS.get(part, (f"Please give the {part}.", ""))
    return (VAGUE_QUESTIONS.get(part, q) if vague else q), ex


def check_part(part, text):
    """-> None if fine, else {"reason", "question", "example"}; warnings go in a separate list."""
    lim = PARTS.get(part) or EXTRA_PARTS.get(part) or {"min_words": 3, "max_chars": 1500}
    text = (text or "").strip()
    if not text:
        q, ex = _question(part)
        return {"reason": "missing_part", "question": q, "example": ex}, []
    if len(text) > lim["max_chars"]:
        return {"reason": "too_long", "question": f"Please shorten the {part} to at most {lim['max_chars']} characters.",
                "example": QUESTIONS.get(part, ("", ""))[1]}, []
    if is_gibberish(text):
        return {"reason": "invalid_text", "question": "Please enter a readable description.", "example": ""}, []
    w, cw = words(text), content_words(text)
    if part in PARTS and (len(w) < lim["min_words"] or len(cw) < max(2, (lim["min_words"] + 1) // 2)):
        q, ex = _question(part, vague=True)
        return {"reason": "too_vague", "question": q, "example": ex}, []
    warn = []
    if part == "result":
        specific = bool(_NUMBER.search(text)) or bool(_DIRECTION.search(text)) or bool(_COMPARE.search(text))
        if not specific:                                 # a qualitative result is still a result: warn only
            warn.append({"code": "result_not_specific", "part": "result",
                         "message": _question("result", vague=True)[0]})
    return None, warn


def near_duplicate(a, b, thr=0.8):
    A, B = {x.lower() for x in content_words(a)}, {x.lower() for x in content_words(b)}
    if not A or not B:
        return a.strip().lower() == b.strip().lower()
    return len(A & B) / len(A | B) >= thr


def validate_parts(task, parts, confidences=None):
    """The checks of POST /v1/validate -> (valid, checks, warnings). parts: {part: text}."""
    req = TASKS[task]["requires"]
    checks, warns = [], []
    for p in req:
        if p == "change" and isinstance(parts.get(p), dict):
            txt = parts[p].get("text", "")
        else:
            txt = parts.get(p)
        bad, w = check_part(p, txt)
        warns += w
        checks.append({"part": p, "ok": True} if bad is None else {"part": p, "ok": False, **bad})
    given = [(p, parts[p]) for p in req if isinstance(parts.get(p), str) and parts.get(p)]
    for i in range(len(given)):
        for j in range(i + 1, len(given)):
            (pa, ta), (pb, tb) = given[i], given[j]
            if task == "edit_check":
                continue
            if near_duplicate(ta, tb):
                for c in checks:
                    if c["part"] == pb and c["ok"]:
                        c.update(ok=False, reason="duplicate_part",
                                 question=f"{pa.capitalize()} and {pb} say the same thing; what exactly is the {pb}?",
                                 example=QUESTIONS.get(pb, ("", ""))[1])
    for p, c in (confidences or {}).items():
        if p in req and c < LOW_CONFIDENCE:
            warns.append({"code": "low_confidence", "part": p, "message": f"Did we read your {p} correctly?"})
    if task == "edit_check" and parts.get("original") and parts.get("edited"):
        share = edit_share(parts["original"], parts["edited"])
        if share == 0:
            for c in checks:
                if c["part"] == "edited":
                    c.update(ok=False, reason="invalid_value", question="The edited copy is the same as the original; "
                             "what did you change?", example=QUESTIONS["edited"][1])
        elif share > MAX_EDIT_SHARE:
            warns.append({"code": "not_a_minimal_edit", "part": "edited", "message": "The check works best for small "
                          f"edits; {round(100 * share)}% of the words changed."})
    return all(c["ok"] for c in checks), checks, warns


# ---------------------------------------------------------------- edits
def edit_share(a, b):
    """Share of words changed between two texts (0 = identical)."""
    wa, wb = [x.lower() for x in words(a)], [x.lower() for x in words(b)]
    if wa == wb:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


_NEG_WORDS = {"not", "no", "never", "none", "cannot", "can't", "didn't", "doesn't", "don't", "isn't", "aren't",
              "wasn't", "weren't", "without", "neither", "nor", "fails", "failed"}


def edit_kind(original, edited):
    """direction / negation / number / substitution (the trained kinds), or other."""
    from sci_jepa.data.ku_schema import counterfactuals
    for kind, txt in counterfactuals(original):
        if txt.strip() == edited.strip():
            return kind
    wa, wb = [x.lower() for x in words(original)], [x.lower() for x in words(edited)]
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    changed_a, changed_b = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            changed_a += wa[i1:i2]; changed_b += wb[j1:j2]
    ch = changed_a + changed_b
    if ch and all(re.fullmatch(r"\d+(?:[.,]\d+)*%?", x) for x in ch):
        return "number"
    if set(ch) & _NEG_WORDS:
        return "negation"
    if any(_DIRECTION.fullmatch(x) for x in ch):
        return "direction"
    if changed_a and changed_b and len(ch) <= 6:
        return "substitution"
    return "other"
