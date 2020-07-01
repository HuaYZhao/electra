import json
import pickle
from copy import deepcopy
from functional import seq
from finetune.qa.squad_official_eval import compute_f1


def generator_data(split='train'):
    data = json.load(open(f'{split}-v2.0.json', 'r', encoding='utf-8'))
    all_nbest = pickle.load(open(f'{split}_all_nbest.pkl', 'rb'))
    count = 0

    for article in data['data']:
        for p in article['paragraphs']:
            del p['context']
            new_qas = []

            for qa in p['qas']:
                qid = qa['id']
                gold_answers = qa['answers']
                if not gold_answers:
                    continue
                nbest = all_nbest[qid][:5]

                most_text = nbest[0]['text']
                new_qa = []
                max_f1 = 0.
                for i, nb in enumerate(nbest):
                    pred = nb['text']
                    if split == 'train':
                        a = qa['answers'][0]['text']
                        f1 = compute_f1(a, pred)
                    else:
                        f1 = max(compute_f1(a['text'], pred) for a in gold_answers)
                    if pred in most_text or most_text in pred:
                        new_qa.append({"f1_score": f1,
                                       "pred_answer": pred,
                                       "question": qa['question'],
                                       "id": f"{qid}_{i}"})
                        max_f1 = max(max_f1, f1)
                if new_qa[0]["f1_score"] > 0:
                    for qa in new_qa:
                        qa['answer_recall'] = 1 if qa['f1_score'] == max_f1 else 0
                    new_qas.extend(new_qa)
            p['qas'] = new_qas
            count += len(new_qas)

    print(count)

    json.dump(data, open(f'{split}.json', 'w', encoding='utf-8'))


generator_data('train')
generator_data('dev')
