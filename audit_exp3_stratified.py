import json
from collections import defaultdict
import statistics

def detect_task_type(conv):
    q = ''
    a = ''
    for c in conv:
        if c['from'] == 'human':
            q = c['value'].lower()
        elif c['from'] == 'gpt':
            a = c['value'].strip()

    if 'answer with' in q and ('yes' in q or 'no' in q):
        return 'binary_yn', q, a
    if 'respond with only the letter' in q or 'multiple-choice' in q:
        return 'mcq', q, a
    if any(w in q for w in ['describe', 'what does', 'explain', 'summarize', 'write a']):
        return 'description', q, a
    if any(w in q for w in ['what type', 'what kind', 'what are', 'what is', 'how many', 'which']):
        return 'open_vqa', q, a
    if 'caption' in q:
        return 'caption', q, a
    return 'other', q, a


COUNTRY_NAMES = ['serbia','portugal','finland','india','germany','france','spain','uk','italy','china','belgium','romania']
SEASON_WORDS = ['summer','winter','spring','autumn','fall','season']
MEAS_WORDS = ['square meter','hectare','km2','sqm','sq m','square km']

stats = defaultdict(lambda: {
    'total': 0,
    'answer_lengths': [],
    'single_token': 0,
    'short_le5': 0,
    'medium_6_30': 0,
    'long_gt30': 0,
    'has_country': 0,
    'has_season': 0,
    'has_measurement': 0,
    'sample_answers': [],
    'sample_questions': [],
})

with open('data/training_exp3/train.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        task_type, q, a = detect_task_type(d.get('conversations', []))
        category = d.get('category', 'unknown')
        key = f'{task_type}/{category}'

        s = stats[key]
        s['total'] += 1
        words = len(a.split())
        s['answer_lengths'].append(words)

        if words <= 1:
            s['single_token'] += 1
        if words <= 5:
            s['short_le5'] += 1
        elif words <= 30:
            s['medium_6_30'] += 1
        else:
            s['long_gt30'] += 1

        if any(c in a.lower() for c in COUNTRY_NAMES):
            s['has_country'] += 1
        if any(sv in a.lower() for sv in SEASON_WORDS):
            s['has_season'] += 1
        if any(m in a.lower() for m in MEAS_WORDS):
            s['has_measurement'] += 1

        if len(s['sample_answers']) < 3:
            last_newline = q.rfind('\n')
            q_short = q[last_newline+1:].strip()[:80] if last_newline >= 0 else q[:80]
            s['sample_answers'].append(a[:120])
            s['sample_questions'].append(q_short)

print('STRATIFIED AUDIT — exp3 train.jsonl')
print('=' * 70)
for key in sorted(stats.keys()):
    s = stats[key]
    n = s['total']
    lengths = s['answer_lengths']
    med = statistics.median(lengths)
    mean_l = statistics.mean(lengths)
    print()
    print(f'[{key}]  n={n}')
    print(f'  Median words: {med:.1f}  |  Mean: {mean_l:.1f}  |  Max: {max(lengths)}')
    print(f'  Single token (<=1): {s["single_token"]:>5}  ({100*s["single_token"]/n:.1f}%)')
    print(f'  Short (<=5 words):  {s["short_le5"]:>5}  ({100*s["short_le5"]/n:.1f}%)')
    print(f'  Medium (6-30 w):    {s["medium_6_30"]:>5}  ({100*s["medium_6_30"]/n:.1f}%)')
    print(f'  Long (>30 words):   {s["long_gt30"]:>5}  ({100*s["long_gt30"]/n:.1f}%)')
    if s['has_country'] or s['has_season'] or s['has_measurement']:
        print(f'  Hallucination: country={s["has_country"]}  season={s["has_season"]}  measurement={s["has_measurement"]}')
    if s['sample_questions']:
        print(f'  Sample Q: {s["sample_questions"][0]}')
        print(f'  Sample A: {s["sample_answers"][0]}')
