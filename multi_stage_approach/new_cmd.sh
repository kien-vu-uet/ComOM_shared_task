###

python main.py --file_type="VLSP23-Addons" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda"

python main.py --file_type="VLSP23-Addons" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --factor=0.5 --penalty=0.25 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

python infer.py --file_type="VLSP23-Addons" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-Addons/private_test'


#####

python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda"

python main.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --factor=0.5 --penalty=0.25 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

python infer.py --file_type="VLSP23-ComOM" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM/private_test'

###

python main.py --file_type="VLSP23-Addons-Segmented" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda"

python main.py --file_type="VLSP23-Addons-Segmented" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --factor=0.5 --penalty=0.25 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

python infer_segmented.py --file_type="VLSP23-Addons-Segmented" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-Addons-Segmented/private_test'
###

python main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda"

python main.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --factor=0.5 --penalty=0.25 --device="cuda" --stage2_clf='mlp' --stage3_clf='mlp'

python infer_segmented.py --file_type="VLSP23-ComOM-Segmented" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=100 --batch=32 --model_type="multitask" --embed_dropout=0.1  --device="cuda" --input='/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM-Segmented/private_test'



