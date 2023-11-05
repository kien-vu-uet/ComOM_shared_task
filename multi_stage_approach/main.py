import torch
import json
import numpy as np

import random
import os
import argparse
import Config

from data_utils import shared_utils, kesserl14_utils, coae13_utils, data_loader_utils, vlsp23_utils
from model_utils import train_test_utils
from eval_utils.base_eval import BaseEvaluation, ElementEvaluation, PairEvaluation
from eval_utils import create_eval
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

    # model parameters.
    parser.add_argument('--embed_dropout', help='prob of embedding dropout', type=float, default=0.1)
    parser.add_argument('--factor', help='the type of data set', type=float, default=0.4)
    parser.add_argument('--penalty', help='the type of data set', type=float, default=0.4)

    # optimizer parameters.
    parser.add_argument('--bert_lr', help='the type of data set', type=float, default=2e-5)
    parser.add_argument('--linear_lr', help='the type of data set', type=float, default=2e-5)
    parser.add_argument('--crf_lr', help='the type of data set', type=float, default=0.01)

    # stage 2 & 3 model.
    parser.add_argument('--stage2_clf', help='classifier for stage 2', type=str, default='logistic')
    parser.add_argument('--stage3_clf', help='classifier for stage 3', type=str, default='logistic')

    parser.add_argument('-a', '--alpha', help='Cosine similarity score coefficient', \
                        type=lambda s: [float(item) for item in s.split(' ')], default="0. 0.1 0.25 0.4"),
    parser.add_argument('-b', '--beta', help='Euclidean distance score coefficient', 
                        type=lambda s: [float(item) for item in s.split(' ')], default="0. 0.1 0.25 0.4"),
    parser.add_argument('-k', '--topk', help='Number of nearest reference points', 
                        type=lambda s: [int(item) for item in s.split(' ')], default="1 3 5 7 9"),    
    parser.add_argument('--main_metric', help='Main metric to evaluate decoder', default='f1_weighted')

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
                  "program_mode": args.program_mode
                  }

    return param_dict


def main():
    # get program configure
    args = TerminalParser()

    grid_search_params = [args.alpha, args.beta, args.topk, args.main_metric]

    # set random seed
    set_seed(args.seed)

    config = Config.BaseConfig(args)
    config_parameters = get_necessary_parameters(args)

    if args.stage_model == "first":
        model_parameters = {"embed_dropout": args.embed_dropout}
    else:
        model_parameters = {"embed_dropout": args.embed_dropout, "factor": args.factor, "penalty": args.penalty}

    optimizer_parameters = None

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

    data_gene.generate_data((True, False, False))
    config.bert_tokenizer = data_gene.bert_tokenizer

    global_eval = BaseEvaluation(config)
    global_pair_eval = BaseEvaluation(config)

    print("create data loader")
    train_loader = data_loader_utils.create_first_data_loader(
        data_gene.train_data_dict, config.batch_size
    )

    dev_loader = data_loader_utils.create_first_data_loader(
        data_gene.dev_data_dict, config.batch_size
    )

    test_loader = data_loader_utils.create_first_data_loader(
        data_gene.test_data_dict, config.batch_size
    )

    # run first-stage model.(extract four type elements)
    if config.stage_model == "first" and config.program_mode != "test":
        first_data_loader = [train_loader, dev_loader, test_loader]

        dev_comp_eval = create_eval.create_first_stage_eval(
            config,
            (data_gene.dev_data_dict['multi_label'], data_gene.dev_data_dict['result_label']),
            data_gene.dev_data_dict['comparative_label'],
            data_gene.dev_data_dict['attn_mask'],
            save_model=True
        )

        test_comp_eval = create_eval.create_first_stage_eval(
            config,
            (data_gene.test_data_dict['multi_label'], data_gene.test_data_dict['result_label']),
            data_gene.test_data_dict['comparative_label'],
            data_gene.test_data_dict['attn_mask'],
            save_model=False
        )

        comp_eval = [dev_comp_eval, test_comp_eval, global_eval]

        train_test_utils.first_stage_model_main(
            config, data_gene, first_data_loader, comp_eval,
            model_parameters, optimizer_parameters,
            model_name
        )

    elif config.program_mode == "test" and config.stage_model == "first":
        dev_parameters = ["./ModelResult/" + model_name + "/dev_elem_result.txt",
                          "./PreTrainModel/" + model_name + "/dev_model"]

        print("==================test================")
        predicate_model = torch.load(dev_parameters[1])

        test_parameters = ["./ModelResult/" + model_name + "/test_elem_result.txt", None]

        test_comp_eval = create_eval.create_first_stage_eval(
            config,
            (data_gene.test_data_dict['multi_label'], data_gene.test_data_dict['result_label']),
            data_gene.test_data_dict['comparative_label'],
            data_gene.test_data_dict['attn_mask'],
            save_model=False
        )

        train_test_utils.first_stage_model_test(
            predicate_model, config, test_loader, test_comp_eval, test_parameters
        )

        test_comp_eval.print_elem_result(
            data_gene.test_data_dict['input_ids'], data_gene.test_data_dict['attn_mask'],
            "./ModelResult/" + model_name + "/test_result_file" + ".txt", drop_span=False
        )

        # add average measure.
        shared_utils.calculate_average_measure(test_comp_eval, global_eval)

    elif config.program_mode == "test" and config.stage_model == "second":
        # 0: 768 + 5, 1: 5, 2: 768
        feature_type = 0

        # using evaluation to generate index col and pair label.
        generate_second_res_eval = ElementEvaluation(
            config, elem_col=config.val.elem_col,
            ids_to_tags=config.val.invert_norm_id_map
        )

        if model_name.find("ele") != -1:
            cross_model_name = model_name.replace("ele", "car")
        else:
            cross_model_name = model_name.replace("car", "ele")

        pre_train_model_path = "./PreTrainModel/" + cross_model_name + "/dev_model"

        if not os.path.exists(pre_train_model_path):
            print("[ERROR] pre-train model isn't exist")
            return

        elem_model = torch.load(pre_train_model_path)

        test_first_process_data_path = "./ModelResult/" + model_name + "/preproc" + "/test_first_data_" + str(feature_type) + ".txt"

        if not os.path.exists("./ModelResult/" + model_name + "/preproc"):
            os.mkdir("./ModelResult/" + model_name + "/preproc")

        if os.path.exists(test_first_process_data_path):
            test_candidate_pair_col, test_pair_representation, test_make_pair_label = \
                shared_utils.read_pickle(test_first_process_data_path)

        else:
            test_candidate_pair_col, test_pair_representation, test_make_pair_label, _, _ = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, test_loader, generate_second_res_eval,
                    eval_parameters=[data_gene.test_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            # test_candidate_pair_col, test_comp_input, test_pair_encoded_input, test_attn_mask_input, test_make_pair_label = \
            # train_test_utils.first_stage_model_test_new(
            #     elem_model, config, test_loader, generate_second_res_eval,
            #     eval_parameters=[data_gene.test_data_dict['tuple_pair_col']],
            #     test_type="gene", feature_type=feature_type
            # )

            shared_utils.write_pickle(
                [test_candidate_pair_col, test_pair_representation, test_make_pair_label],
                test_first_process_data_path
            )

        dev_pair_parameters = ["./ModelResult/" + cross_model_name + "/dev_pair_result.txt",
                               "./PreTrainModel/" + cross_model_name + "/dev_pair_model", 
                               "./PreTrainModel/" + cross_model_name + "/pair_refer_params"
                               ]

        dev_polarity_parameters = ["./ModelResult/" + cross_model_name + "/dev_polarity_result.txt",
                                   "./PreTrainModel/" + cross_model_name + "/dev_polarity_model",
                                   "./PreTrainModel/" + cross_model_name + "/polarity_refer_params"
                                   ]

        test_pair_parameters = ["./ModelResult/" + cross_model_name + "/test_pair_result.txt", None]        
        test_polarity_parameters = ["./ModelResult/" + cross_model_name + "/test_pair_result.txt", None]

        predict_pair_model = torch.load(dev_pair_parameters[1])
        pair_refer_params = shared_utils.read_pickle(dev_pair_parameters[2])
        
        predict_polarity_model = torch.load(dev_polarity_parameters[1])
        polarity_refer_params = shared_utils.read_pickle(dev_polarity_parameters[2])

        test_pair_eval = PairEvaluation(
            config,
            gold_pair_col=data_gene.test_data_dict['tuple_pair_col'],
            candidate_pair_col=test_candidate_pair_col,
            elem_col=config.val.elem_col,
            ids_to_tags=config.val.norm_id_map,
            save_model=False
        )

        test_pair_loader = data_loader_utils.get_loader([test_pair_representation], 1)

        train_test_utils.pair_stage_model_test(
            predict_pair_model, pair_refer_params, config, test_pair_loader, test_pair_eval,
            test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
        )

        shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)
        global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
        global_pair_eval.store_result_to_csv([model_name], "result.csv")

        shared_utils.clear_global_measure(global_pair_eval)
        shared_utils.clear_optimize_measure(test_pair_eval)

        # create polarity representation and data loader.
        test_polarity_representation = cpc.get_after_pair_representation(test_pair_eval.y_hat, test_pair_representation)
        test_polarity_loader = data_loader_utils.get_loader([test_polarity_representation], 1)

        train_test_utils.pair_stage_model_test(
            predict_polarity_model, polarity_refer_params, config, test_polarity_loader, test_pair_eval,
            test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True)
        )

        # add average measure.
        shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)

    elif config.stage_model == "second":
        # 0: 768 + 5, 1: 5, 2: 768
        feature_type = 0

        # using evaluation to generate index col and pair label.
        generate_second_res_eval = ElementEvaluation(
            config, elem_col=config.val.elem_col,
            ids_to_tags=config.val.invert_norm_id_map
        )

        pre_train_model_path = "./PreTrainModel/" + model_name + "/dev_model"

        if not os.path.exists(pre_train_model_path):
            print("[ERROR] pre-train model isn't exist")
            return

        elem_model = torch.load(pre_train_model_path)

        train_first_process_data_path = "./ModelResult/" + model_name + "/preproc" + "/train_first_data_" + str(feature_type) + ".txt"
        dev_first_process_data_path = "./ModelResult/" + model_name + "/preproc" + "/dev_first_data_" + str(feature_type) + ".txt"
        test_first_process_data_path = "./ModelResult/" + model_name + "/preproc" + "/test_first_data_" + str(feature_type) + ".txt"

        if not os.path.exists("./ModelResult/" + model_name + "/preproc"):
            os.mkdir("./ModelResult/" + model_name + "/preproc")

        if os.path.exists(train_first_process_data_path):
            train_pair_representation, train_make_pair_label, train_polarity_representation, train_polarity_label = \
                shared_utils.read_pickle(train_first_process_data_path)
        else:
            _, train_pair_representation, train_make_pair_label, train_feature_out, train_bert_feature_out = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, train_loader, generate_second_res_eval,
                    eval_parameters=[data_gene.train_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            train_pair_representation, train_make_pair_label = cpc.generate_train_pair_data(
                train_pair_representation, train_make_pair_label
            )

            train_polarity_representation, train_polarity_label = cpc.create_polarity_train_data(
                config, data_gene.train_data_dict['tuple_pair_col'], train_feature_out,
                train_bert_feature_out, feature_type=feature_type
            )

            shared_utils.write_pickle(
                [train_pair_representation, train_make_pair_label,
                 train_polarity_representation, train_polarity_label],
                train_first_process_data_path
            )

        if os.path.exists(dev_first_process_data_path):
            dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label, dev_polarity_representation, dev_polarity_label = \
                shared_utils.read_pickle(dev_first_process_data_path)

        else:
            dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label, dev_feature_out, dev_bert_feature_out = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, dev_loader, generate_second_res_eval,
                    eval_parameters=[data_gene.dev_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )
            
            dev_polarity_representation, dev_polarity_label = cpc.create_polarity_train_data(
                config, data_gene.dev_data_dict['tuple_pair_col'], dev_feature_out,
                dev_bert_feature_out, feature_type=feature_type
            )

            shared_utils.write_pickle(
                [dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label, dev_polarity_representation, dev_polarity_label],
                dev_first_process_data_path
            )

        if os.path.exists(test_first_process_data_path):
            test_candidate_pair_col, test_pair_representation, test_make_pair_label = \
                shared_utils.read_pickle(test_first_process_data_path)

        else:
            test_candidate_pair_col, test_pair_representation, test_make_pair_label, _, _ = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, test_loader, generate_second_res_eval,
                    eval_parameters=[data_gene.test_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            shared_utils.write_pickle(
                [test_candidate_pair_col, test_pair_representation, test_make_pair_label],
                test_first_process_data_path
            )

        pair_representation = [train_pair_representation, dev_pair_representation, test_pair_representation]
        make_pair_label = [train_make_pair_label, dev_make_pair_label, test_make_pair_label]

        dev_pair_eval = PairEvaluation(
            config,
            gold_pair_col=data_gene.dev_data_dict['tuple_pair_col'],
            candidate_pair_col=dev_candidate_pair_col,
            elem_col=config.val.elem_col,
            ids_to_tags=config.val.norm_id_map,
            save_model=True
        )

        test_pair_eval = PairEvaluation(
            config,
            gold_pair_col=data_gene.test_data_dict['tuple_pair_col'],
            candidate_pair_col=test_candidate_pair_col,
            elem_col=config.val.elem_col,
            ids_to_tags=config.val.norm_id_map,
            save_model=False
        )

        train_test_utils.pair_stage_model_main(
            config, pair_representation, make_pair_label,
            [dev_pair_eval, test_pair_eval, global_pair_eval],
            [train_polarity_representation, train_polarity_label, dev_polarity_representation, dev_polarity_label],
            model_parameters, optimizer_parameters, model_name, feature_type, grid_search_params
        )

    if config.stage_model == "first":
        global_eval.avg_model("./ModelResult/" + model_name + "/test_extraction_result.txt")
        global_eval.store_result_to_csv([model_name], "result.csv")
    else:
        global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
        global_pair_eval.store_result_to_csv([model_name], "result.csv")


if __name__ == "__main__":
    main()
