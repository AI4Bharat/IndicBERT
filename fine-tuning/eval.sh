
# NER eval
for lang in as bn gu hi kn ml mr or pa ta te ur
do
python ner/ner.py --model_name <MODEL> --do_predict --eval_data wikiann.$lang
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