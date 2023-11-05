import torch
import json
import numpy as np

import random
import os
import argparse
import Config

from tqdm import tqdm

from data_utils import shared_utils, kesserl14_utils, coae13_utils, data_loader_utils, vlsp23_utils
from model_utils import train_test_utils, refer_neighbooring_decoder
from eval_utils.base_eval import BaseEvaluation, ElementEvaluation, PairEvaluation
from data_utils import current_program_code as cpc

import transformers
transformers.logging.set_verbosity_error()
# transformers.logging.set_verbosity_info()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.nn.Module.dump_patches = True


def TerminalParser():
    # define parse parameters
    parser = argparse.ArgumentParser()
    parser.description = 'choose train data and test data file path'

    parser.add_argument('--seed', help='random seed', type=int, default=2021)
    parser.add_argument('--batch', help='input data batch size', type=int, default=16)
    parser.add_argument('--epoch', help='the number of run times', type=int, default=25)
    parser.add_argument('--fold', help='the fold of data', type=int, default=5)

    # lstm parameters setting
    parser.add_argument('--input_size', help='the size of encoder embedding', type=int, default=300)
    parser.add_argument('--hidden_size', help='the size of hidden embedding', type=int, default=512)
    parser.add_argument('--num_layers', help='the number of layer', type=int, default=2)

    # program mode choose.
    parser.add_argument('--model_mode', help='bert or norm', default='bert')
    parser.add_argument('--bert_size', help='bert base or large', default='base')
    parser.add_argument('--server_type', help='1080ti or rtx', default='1080ti')
    parser.add_argument('--program_mode', help='debug or run or test', default='run')
    parser.add_argument('--stage_model', help='first or second', default='first')
    parser.add_argument('--model_type', help='bert_crf, bert_crf_mtl', default='crf')
    parser.add_argument('--position_sys', help='BIES or BI or SPAN', default='BMES')

    parser.add_argument('--device', help='run program in device type',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--file_type', help='the type of data set', default='car')
    parser.add_argument('--premodel_path', help='the type of data set', default="../pretrain_model/")
    parser.add_argument('--input', help='the type of data set', default=None)
    # parser.add_argument('--output', help='the type of data set', default=None)

    # model parameters.
    parser.add_argument('--embed_dropout', help='prob of embedding dropout', type=float, default=0.1)
    parser.add_argument('--factor', help='the type of data set', type=float, default=0.4)

    # optimizer parameters.
    parser.add_argument('--bert_lr', help='the type of data set', type=float, default=2e-5)
    parser.add_argument('--linear_lr', help='the type of data set', type=float, default=2e-5)
    parser.add_argument('--crf_lr', help='the type of data set', type=float, default=0.01)

    # stage 2 & 3 model.
    parser.add_argument('--stage2_clf', help='classifier for stage 2', type=str, default='logistic')
    parser.add_argument('--stage3_clf', help='classifier for stage 3', type=str, default='logistic')

    args = parser.parse_args()

    return args


def get_necessary_parameters(args):
    """
    :param args:
    :return:
    """
    param_dict = {"file_type": args.file_type,
                  "model_mode": args.model_mode,
                  "bert_size": args.bert_size,
                  "stage_model": args.stage_model,
                  "model_type": args.model_type,
                  "epoch": args.epoch,
                  "batch_size": args.batch,
                  "program_mode": args.program_mode}

    return param_dict


def main():
    # get program configure
    args = TerminalParser()

    # set random seed
    set_seed(args.seed)

    config = Config.BaseConfig(args)

    config_parameters = get_necessary_parameters(args)

    if args.stage_model == "first":
        model_parameters = {"embed_dropout": args.embed_dropout}
    else:
        model_parameters = {"embed_dropout": args.embed_dropout, "factor": args.factor}

    model_name = shared_utils.parameters_to_model_name(
        {"config": config_parameters, "model": model_parameters}
    )

    print(model_name)

    if config.data_type == "eng":
        data_gene = kesserl14_utils.DataGenerator(config)
    elif config.data_type == 'vie':
        data_gene = vlsp23_utils.DataGenerator(config)
    else:
        data_gene = coae13_utils.DataGenerator(config)

    '''
    LOAD MODEL
    '''
    ### FOR STAGE 1
    if model_name.find("ele") != -1:
            cross_model_name = model_name.replace("ele", "car")
    else:
        cross_model_name = model_name.replace("car", "ele")

    pre_train_model_path = "./PreTrainModel/" + cross_model_name + "/dev_model"

    if not os.path.exists(pre_train_model_path):
        print("[ERROR] pre-train model isn't exist")
        return

    elem_model = torch.load(pre_train_model_path).to(config.device)

    ### FOR STAGE 2 & 3
    dev_pair_parameters = ["./ModelResult/" + cross_model_name + "/dev_pair_result.txt",
                            "./PreTrainModel/" + cross_model_name + "/dev_pair_model", 
                            "./PreTrainModel/" + cross_model_name + "/pair_refer_params"
                            ]

    dev_polarity_parameters = ["./ModelResult/" + cross_model_name + "/dev_polarity_result.txt",
                                "./PreTrainModel/" + cross_model_name + "/dev_polarity_model",
                                "./PreTrainModel/" + cross_model_name + "/polarity_refer_params"
                                ]
    
    predict_pair_model = torch.load(dev_pair_parameters[1]).to(config.device)
    pair_refer_params = shared_utils.read_pickle(dev_pair_parameters[2])
    pair_refer_neighbor_decoder = refer_neighbooring_decoder.ReferNeighbooringStrategy(
        refer_features=pair_refer_params[0],
        refer_labels=pair_refer_params[1],
        num_class=pair_refer_params[2],
        best_result=pair_refer_params[3], 
        device=config.device
    )

    predict_polarity_model = torch.load(dev_polarity_parameters[1]).to(config.device)
    polarity_refer_params = shared_utils.read_pickle(dev_polarity_parameters[2])
    polarity_refer_neighbor_decoder = refer_neighbooring_decoder.ReferNeighbooringStrategy(
        refer_features=polarity_refer_params[0],
        refer_labels=polarity_refer_params[1],
        num_class=polarity_refer_params[2],
        best_result=polarity_refer_params[3], 
        device=config.device
    )
    ###

    if not os.path.exists(args.input + '/result'):
        os.mkdir(args.input + '/result')
    output = args.input + '/result/'

    input_files = os.listdir(args.input + '/no_label')
    for file in input_files:
        with open(args.input + '/raw/' + file, 'r') as f:
            raw_sent_col = f.read()
            raw_sent_col = raw_sent_col.split('\n')
            f.close()

        data_gene.generate_infer_data(args.input + '/no_label/' + file)

        config.bert_tokenizer = data_gene.bert_tokenizer

        test_loader = data_loader_utils.create_first_data_loader(
            data_gene.test_data_dict, config.batch_size
        )


        '''
        INFER STAGE 1
        '''
        generate_second_res_eval = ElementEvaluation(
                config, elem_col=config.val.elem_col,
                ids_to_tags=config.val.invert_norm_id_map
            )
        
        feature_type = 0
        test_candidate_pair_col, test_pair_representation, _, _, _ = \
                    train_test_utils.first_stage_model_test(
                        elem_model, config, test_loader, generate_second_res_eval,
                        eval_parameters=[data_gene.test_data_dict['tuple_pair_col']],
                        test_type="gene", feature_type=feature_type
                    )
        # sent_comparative_label = generate_second_res_eval.predict_sent_label

        '''
        INFER STAGE 2 & 3
        '''
        predict_tuple_pair_col = []
        null_tuple_pair = [(-1, -1)] * 4 + [-1]

        with torch.no_grad():
            for index, data in tqdm(enumerate(test_pair_representation)):
                # if sent_comparative_label[index] == 0:
                #     predict_tuple_pair_col.append([null_tuple_pair])
                #     continue

                candidate_pair = test_candidate_pair_col[index]
                pair_representation = torch.FloatTensor(data).to(config.device)

                # pair_out = predict_pair_model(pair_representation).view(-1)
                pair_out = pair_refer_neighbor_decoder.predict(predict_pair_model,
                                                               pair_representation,
                                                               softmax=False).view(-1)

                if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
                    pair_out = torch.zeros(pair_out.size())

                pair_out = pair_out.cpu()
                pair_out = pair_out == 1

                if pair_out.sum() == 0:
                    predict_tuple_pair_col.append([null_tuple_pair])
                else:
                    pair_representation = pair_representation[pair_out]
                    candidate_pair = [candidate_pair[i] for i, p in enumerate(pair_out.numpy()) if p]
                    # polarity_out = predict_polarity_model(pair_representation).view(-1)
                    polarity_out = polarity_refer_neighbor_decoder.predict(predict_polarity_model,
                                                                           pair_representation,
                                                                           softmax=False).view(-1)

                    polarity_out = polarity_out.cpu().numpy().tolist()
                    new_pair = [candidate_pair[i] + [polarity_out[i]] for i in range(len(candidate_pair))]
                    predict_tuple_pair_col.append(new_pair)

                # '''
                # Decode without stage 2
                # '''
                # polarity_out = predict_polarity_model(pair_representation).view(-1)
                # polarity_out = polarity_out.cpu().numpy().tolist()
                # new_pair = [candidate_pair[i] + [polarity_out[i]] for i in range(len(candidate_pair))]
                # predict_tuple_pair_col.append(new_pair)



        # shared_utils.write_pickle(
        #             predict_tuple_pair_col,
        #             '/workspace/nlplab/kienvt/COQE/tmp/full_tuple.txt'
        #         )
        
        # shared_utils.write_pickle(
        #             data_gene.test_data_dict,
        #             '/workspace/nlplab/kienvt/COQE/tmp/data_dict.txt'
        #         )
        
        
        '''
        FOR DECODE FINAL OUTPUT
        '''
        data_dict = data_gene.test_data_dict
        mapping_col = data_dict['token_mapping_col']
        standard_token_col = data_dict['standard_token']

        standard_token_2_bert_idx = []
        for index in range(len(mapping_col)):
            old_map = mapping_col[index]
            standard_token = standard_token_col[index]
            new_map = []
            for idx, list_bert_token in old_map.items():
                new_map.append((standard_token[idx], list_bert_token))
            standard_token_2_bert_idx.append(new_map)

        polarity_tags = config.val.polarity_col
        result = ''
        elem_tags = ["subject", "object", "aspect", "predicate", "label"]
        for index in range(len(standard_token_2_bert_idx)):
            raw_sent = raw_sent_col[index]
            sent_map = standard_token_2_bert_idx[index]
            idx_map = mapping_col[index]
            all_tuple_pair = predict_tuple_pair_col[index]
            list_token = [t for t, l in sent_map]
            result += raw_sent + "\t" + " ".join(list_token) + "\n"
            for tuple_pair in all_tuple_pair:
                if tuple_pair[:-1] == [[-1, -1]] * 4:
                    result += '\n'
                else:
                    pair_dict = {}
                    
                    for elem in range(4):
                        s_index, e_index = tuple_pair[elem]
                        new_s_index, new_e_index = -1, -1

                        if s_index != -1 and e_index != -1:
                            for i, l in idx_map.items():
                                if s_index in l:
                                    new_s_index = i
                                if e_index in l:
                                    new_e_index = i

                            pair_dict[elem_tags[elem]] = [f"{i+1}&&{list_token[i]}" for i in range(new_s_index, new_e_index)]
                        else:
                            pair_dict[elem_tags[elem]] = []
                    
                    pair_dict["label"] = f"{polarity_tags[tuple_pair[-1]]}"

                    is_superlative = False
                    for tok in pair_dict["predicate"]:
                        if tok.lower().find('nhất') != -1:
                            is_superlative = True
                            break
                    if is_superlative: pair_dict["label"] = pair_dict["label"].replace("COM", "SUP")
                    
                    if len(pair_dict['predicate']) > 0:
                        result += json.dumps(pair_dict, ensure_ascii=False)

                result += '\n'
            result += '\n'

        while result.find('\n\n\n') != -1: result = result.replace('\n\n\n', '\n\n')
        
        if os.path.exists(output + file):
            os.remove(output + file)
        with open(output + file, 'a') as f:
            f.write(result)
            f.close()

    zip_name = 'mode_multitask_with_softmax_before_crf.zip'
    os.remove(f'{output}\*.zip')
    os.chdir(f'{output}')
    os.system(f'zip -r {zip_name} *')

if __name__ == "__main__":
    main()