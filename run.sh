# convert paras to sents
for lang in as bd bn dg en gom gu hi kha kn ks mai ml mni mr ne or pa sa sat sd ta te ur
do
python scripts/convert_para2sent.py --input ../monolingual_data/$lang.txt --output ../monolingual_sents/$lang.txt
echo $lang
done


# create tokenizer
python tokenization/build_tokenizer.py --input ../tokenizer_data/ --output ../wordpiece_250k/ --vocab 250000


# sbatch command to create mlm data
sbatch --job-name mlm_data --gres gpu:0 -p cpup --cpus-per-task 128 --nodes 1 \
    --ntasks-per-node 1 --time=07-00:00:00 \
    --wrap 'srun --output mlm_data.log.node%t --error mlm_data.stderr.node%t.%j  \
    bash create_data.sh'