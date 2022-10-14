
# NER wikiann eval
for lang in as bn gu hi kn ml mr or pa ta te ur
do
python ner/ner.py --model_name <MODEL> --do_predict --eval_data wikiann.$lang --benchmark wikiann
done

# NER naamapadam eval
for lang in as bn gu hi kn ml mr or pa ta te
do
python ner/ner.py --model_name <MODEL> --do_predict --eval_data $lang
done

# POS eval
for lang in Hindi Marathi Tamil Telugu Urdu
do
python pos/pos.py --model_name <MODEL> --do_predict --eval_data udpos.$lang
done

# QA eval
for lang in as bn gu hi kn ml mr or pa ta te
do
python qa/qa.py --model_name <MODEL> --do_predict --eval_data indicqa.$lang
done

# XCOPA eval
for lang in as bn en gom gu hi kn mai ml mr ne or pa sa sat sd ta te ur
do
python xcopa/xcopa.py --model_name <MODEL> --do_predict --task_name xcopa --eval_data translation-$lang
done

# XNLI eval
for lang in as bn gu hi kn ml mr or pa ta te ur
do
python xnli/xnli.py --model_name <MODEL> --do_predict --eval_data $lang
done

# FLORES retrieval
for lang in asm_Beng ben_Beng guj_Gujr hin_Deva kan_Knda kas_Arab mai_Deva mal_Mlym mar_Deva mni_Beng npi_Deva ory_Orya pan_Guru san_Deva sat_Olck tam_Taml tel_Telu urd_Arab
do
python retrieval/retrieval.py --model_name <MODEL> --do_predict --src_file $lang
done

# Sentiment eval
for lang in as bn gu hi kn ml mr or pa ta te ur
do
python sentiment/sentiment.py --model_name <MODEL> --do_predict --eval_data $lang
done

# paraphrase eval
for lang in as bn gu hi kn ml mr or pa te
do
python paraphrase/paraphrase.py --model_name <MODEL> --do_predict --eval_data $lang
done