# dedup at document level and tag the first sent in a document with the language id

import random
import time
import sys
import itertools

lang = sys.argv[1]

src_documents, tgt_documents = [], []
with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/en-{lang}.en', 'r') as f:
    src = f.readlines()

with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/en-{lang}.{lang}', 'r') as f:
    tgt = f.readlines()

src_single_doc, tgt_single_doc = [], []
for src_l, tgt_l in zip(src, tgt):
    if src_l == '\n' and tgt_l == '\n':
        src_documents.append(src_single_doc)
        tgt_documents.append(tgt_single_doc)
        src_single_doc, tgt_single_doc = [], []
    else:
        src_single_doc.append(src_l)
        tgt_single_doc.append(tgt_l)

assert len(src_documents) == len(tgt_documents)
print(f'Length of docs: {len(src_documents)}')

start = time.time()
c = list(zip(src_documents, tgt_documents))
random.shuffle(c)
src_documents, tgt_documents = zip(*c)
end = time.time()

print(f'Time to shuffle: {end-start}')

with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/tagged-en-{lang}.en', 'w') as f:
    for doc in src_documents:
        for i, sent in enumerate(doc):
            if i == 0:
                f.write(f'<{lang}> ' + sent.strip()+'\n')
            else:
                f.write(sent.strip()+'\n')
        f.write('\n')

with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/tagged-en-{lang}.{lang}', 'w') as f:
    for doc in tgt_documents:
        for i, sent in enumerate(doc):
            if i == 0:
                f.write(f'<{lang}> ' + sent.strip()+'\n')
            else:
                f.write(sent.strip()+'\n')
        f.write('\n')