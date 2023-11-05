# first stage.
# CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Car-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/"
# CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Ele-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"
# python main.py --file_type="Camera-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda"
python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda"


# second and thrid stage.
# CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Car-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.3
# CUDA_VISIBLE_DEVICES=5,6 python main.py --file_type="Ele-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/" --factor=0.3
# python main.py --file_type="Camera-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.3 --device="cuda"
python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

# infer
# python infer.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM/format'

python main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda"

python main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

# python infer.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM-Segmented/test_no_label.txt' --output='/workspace/nlplab/kienvt/COQE/tmp/test_with_label_segment.txt'
# python infer_segmented.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM-Segmented/format'


python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=16 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" && python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=16 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'


# python new_main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'




################################# Mode extraction ###########################
python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda"

python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

python infer.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM/format'

###

python main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda"

python main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

python infer_segmented.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM-Segmented/format'

############################## bert large ###################################
python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=8 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --bert_size="large"

python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=8 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp' --bert_size="large"

python infer.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=8 --model_type="extraction" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --device="cuda" --bert_size="large" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM/format'


# python new_main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1 --premodel_path="../pretrain_model/" --factor=0.5 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'