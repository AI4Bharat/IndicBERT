# dedup at document level and tag the first sent in a document with the language id

import random
import time
import sys
import itertools

lang = sys.argv[1]

documents = []
with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_paras/{lang}.txt', 'r') as f:
    lines = f.readlines()

single_doc = []
for line in lines:
    if line == '\n':
        documents.append(single_doc)
        single_doc = []
    else:
        single_doc.append(line)

print(f'Length of docs: {len(documents)}')
documents.sort()
deduped = list(k for k,_ in itertools.groupby(documents))
print(f'Length of deduped docs: {len(deduped)}')

start = time.time()
random.shuffle(deduped)
end = time.time()

print(f'Time to shuffle: {end-start}')

with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_paras/tagged-{lang}.txt', 'w') as f:
    for doc in deduped:
        for i, sent in enumerate(doc):
            if i == 0:
                f.write(f'<{lang}> ' + sent.strip()+'\n')
            else:
                f.write(sent.strip()+'\n')
        f.write('\n')