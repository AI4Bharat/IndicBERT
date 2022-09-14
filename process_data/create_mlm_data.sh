# monolingual data creation
start=$1
stop=$2

create_data(){
	python /nlsasfs/home/ai4bharat/gramesh/fine-tuning/IndicBERT/process_data/create_mlm_data.py \
		--input_file=/nlsasfs/home/ai4bharat/gramesh/fine-tuning/splits-0.7/shuf-$1.txt \
		--output_file=/nlsasfs/home/ai4bharat/gramesh/fine-tuning/tfrecords-0.7/shuf-$1.tfrecord \
		--input_file_type=monolingual \
		--tokenizer=/nlsasfs/home/ai4bharat/gramesh/fine-tuning/IndicBERT/tokenization/wp_land_id_250k/config.json \
		--max_seq_length=512 \
		--max_predictions_per_seq=77 \
		--do_whole_word_mask=True \
		--masked_lm_prob=0.15 \
		--random_seed=12345 \
		--dupe_factor=1
}

for i in $( eval echo {$start..$stop} )
do
create_data $i
echo $i
done