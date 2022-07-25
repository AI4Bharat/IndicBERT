# convert paras to sents
for lang in as bd bn dg en gom gu hi kha kn ks mai ml mni mr ne or pa sa sat sd ta te ur
do
python scripts/convert_para2sent.py --input ../monolingual_data/$lang.txt --output ../monolingual_sents/$lang.txt
echo $lang
done