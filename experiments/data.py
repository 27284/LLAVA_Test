"""Dataset preparation and VQAv2 evaluation (no model dependencies)."""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('yes_no', 'count', 'color', 'object', 'spatial')
SUBSET = ROOT / 'data/vqav2/subset/vqav2_validation_300.json'
IMAGES = ROOT / 'data/vqav2/images/val2014'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        save_json(Path(path).with_suffix('.empty.json'), {'samples': 0})
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def classify(question, answer_type=''):
    q = question.lower().strip()
    if answer_type == 'yes/no' or re.match(r'^(is|are|was|were|does|do|did|can|could|has|have|will|would)\b', q):
        return 'yes_no'
    if re.search(r'\bcolou?rs?\b', q):
        return 'color'
    if q.startswith('how many'):
        return 'count'
    # Numeric answers such as times, ages, prices and jersey numbers are not counts.
    if re.match(r'^where (is|are)\b', q) or re.search(r'\b(left of|right of|next to|beside|behind|in front of|under|above|below|on top of|between)\b', q):
        return 'spatial'
    if re.match(r'^(what (is|are|kind|type|object|animal)|which object|who is)\b', q):
        return 'object'
    return None


def prepare(questions, annotations, output, per_category=60, seed=42):
    anns = {a['question_id']: a for a in load_json(annotations)['annotations']}
    pools = {c: [] for c in CATEGORIES}
    for q in load_json(questions)['questions']:
        a = anns[q['question_id']]
        category = classify(q['question'], a['answer_type'])
        if category is None:
            continue
        pools[category].append(dict(q, category=category, answer=a['multiple_choice_answer'],
                                    all_answers=[x['answer'] for x in a['answers']],
                                    answer_type=a['answer_type'], question_type_original=a['question_type'],
                                    image_filename=f"COCO_val2014_{q['image_id']:012d}.jpg"))
    rng = random.Random(seed)
    selected = []
    for category, pool in pools.items():
        if len(pool) < per_category:
            raise ValueError(f'{category}: only {len(pool)} available')
        selected.extend(rng.sample(pool, per_category))
    rng.shuffle(selected)
    save_json(output, selected)
    return Counter(x['category'] for x in selected)


# Rules and contraction spellings follow the official VQA evaluator:
# https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
CONTRACTIONS = dict(pair.split('=', 1) for pair in """
aint=ain't arent=aren't cant=can't couldve=could've couldnt=couldn't couldn'tve=couldn't've couldnt've=couldn't've
didnt=didn't doesnt=doesn't dont=don't hadnt=hadn't hadnt've=hadn't've hadn'tve=hadn't've hasnt=hasn't havent=haven't
hed=he'd hed've=he'd've he'dve=he'd've hes=he's howd=how'd howll=how'll hows=how's Id've=I'd've I'dve=I'd've Im=I'm Ive=I've
isnt=isn't itd=it'd itd've=it'd've it'dve=it'd've itll=it'll let's=let's maam=ma'am mightnt=mightn't mightnt've=mightn't've
mightn'tve=mightn't've mightve=might've mustnt=mustn't mustve=must've neednt=needn't notve=not've oclock=o'clock oughtnt=oughtn't
ow's'at='ow's'at 'ows'at='ow's'at 'ow'sat='ow's'at shant=shan't shed've=she'd've she'dve=she'd've she's=she's
shouldve=should've shouldnt=shouldn't shouldnt've=shouldn't've shouldn'tve=shouldn't've somebody'd=somebodyd
somebodyd've=somebody'd've somebody'dve=somebody'd've somebodyll=somebody'll somebodys=somebody's someoned=someone'd
someoned've=someone'd've someone'dve=someone'd've someonell=someone'll someones=someone's somethingd=something'd
somethingd've=something'd've something'dve=something'd've somethingll=something'll thats=that's thered=there'd
thered've=there'd've there'dve=there'd've therere=there're theres=there's theyd=they'd theyd've=they'd've they'dve=they'd've
theyll=they'll theyre=they're theyve=they've twas='twas wasnt=wasn't wed've=we'd've we'dve=we'd've weve=we've werent=weren't
whatll=what'll whatre=what're whats=what's whatve=what've whens=when's whered=where'd wheres=where's whereve=where've
whod=who'd whod've=who'd've who'dve=who'd've wholl=who'll whos=who's whove=who've whyll=why'll whyre=why're whys=why's
wont=won't wouldve=would've wouldnt=wouldn't wouldnt've=wouldn't've wouldn'tve=wouldn't've yall=y'all yall'll=y'all'll
yallll=y'all'll yall'd've=y'all'd've y'alld've=y'all'd've y'all'dve=y'all'd've youd=you'd youd've=you'd've you'dve=you'd've
youll=you'll youre=you're youve=you've
""".split())
NUMBERS = dict(zip('none zero one two three four five six seven eight nine ten'.split(), '0 0 1 2 3 4 5 6 7 8 9 10'.split()))


def normalize(text):
    original = text.replace('\n', ' ').replace('\t', ' ').strip()
    text = original
    for p in '; / [ ] " { } ( ) = + \\ _ - > < @ ` , ? !'.split():
        text = text.replace(p, '' if p + ' ' in original or ' ' + p in original or re.search(r'\d,\d', original) else ' ')
    text = re.sub(r'(?!<=\d)(\.)(?!\d)', '', text)
    return ' '.join(CONTRACTIONS.get(NUMBERS.get(w, w), NUMBERS.get(w, w))
                    for w in text.lower().split() if w not in ('a', 'an', 'the'))


def soft_accuracy(prediction, answers):
    if len(answers) != 10:
        raise ValueError('VQAv2 soft accuracy requires all 10 human answers')
    answers = [a.replace('\n', ' ').replace('\t', ' ').strip() for a in answers]
    prediction = prediction.replace('\n', ' ').replace('\t', ' ').strip()
    # Preserve the official evaluator's conditional normalization.
    if len(set(answers)) > 1:
        answers = [normalize(a) for a in answers]
        prediction = normalize(prediction)
    return sum(min(1.0, sum(prediction == a for j, a in enumerate(answers) if j != i) / 3)
               for i in range(10)) / 10


def evaluate(samples, predictions, threshold=1.0):
    indexed = {str(p['question_id']): p for p in predictions}
    if len(indexed) != len(predictions):
        raise ValueError('Duplicate prediction question_id')
    rows = []
    for s in samples:
        p = indexed.get(str(s['question_id']))
        if p is None:
            continue
        pred = p.get('predicted_answer', p.get('pred_answer', ''))
        score = soft_accuracy(pred, s['all_answers'])
        rows.append(dict(question_id=s['question_id'], image_id=s['image_id'],
                         image_filename=s['image_filename'], category=s['category'], question=s['question'],
                         gt_answer=s['answer'], predicted_answer=pred, soft_accuracy=score,
                         exact_match=int(normalize(pred) == normalize(s['answer'])),
                         correct=int(score >= threshold)))
    return rows
