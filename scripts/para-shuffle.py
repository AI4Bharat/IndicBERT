# combine all langauges into 1 file + shuffle + split by N(5M) documents into multiple sub docs each with all language content

import random
import time

langs_str = "as bn gu hi kn ml mr or pa ta te"
langs = langs_str.split()
print(f'Langs: {langs}')

src_documents, tgt_documents = [], []
for lang in langs:
    with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/tagged-en-{lang}.en', 'r') as f:
        src = f.readlines()

    with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/tagged-en-{lang}.{lang}', 'r') as f:
        tgt = f.readlines()

    print(f'Finished reading {lang}')
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

src_doc_count = 0
with open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/tlm-splits/shuf-{src_doc_count//3000000}.en', 'w') as f:
    for doc in src_documents:
        src_doc_count += 1
        for sent in doc:
            f.write(sent.strip()+'\n')
        f.write('\n')

        if src_doc_count % 3000000 == 0:
            f = open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/tlm-splits/shuf-{src_doc_count//3000000}.en', 'w')

tgt_doc_count = 0
with open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/tlm-splits/shuf-{tgt_doc_count//3000000}.lang', 'w') as f:
    for doc in tgt_documents:
        tgt_doc_count += 1
        for sent in doc:
            f.write(sent.strip()+'\n')
        f.write('\n')

        if tgt_doc_count % 3000000 == 0:
            f = open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/tlm-splits/shuf-{tgt_doc_count//3000000}.lang', 'w')