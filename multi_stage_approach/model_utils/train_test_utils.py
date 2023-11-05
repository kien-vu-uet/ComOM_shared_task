import torch
import copy
import torch.nn as nn
import numpy as np
from data_utils import shared_utils, data_loader_utils
from model_utils import pipeline_model_utils, optimizer_utils, refer_neighbooring_decoder
from data_utils import current_program_code as cpc
from tqdm import tqdm


########################################################################################################################
# Train and Test Program
########################################################################################################################
def first_stage_model_train(model, optimizer, train_loader, config, epoch):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param config:
    :param epoch:
    :return:
    """
    model.train()

    epoch_loss = 0
    for index, data in tqdm(enumerate(train_loader)):
        input_ids, attn_mask, comparative_label, multi_label, result_label = data

        input_ids = torch.tensor(input_ids).long().to(config.device)
        attn_mask = torch.tensor(attn_mask).long().to(config.device)

        comparative_label = torch.tensor(comparative_label).long().to(config.device)
        multi_label = torch.tensor(np.array(multi_label)).long().to(config.device)
        result_label = torch.tensor(np.array(result_label)).long().to(config.device)

        loss = model(input_ids, attn_mask, comparative_label, multi_label, result_label)

        loss = torch.sum(loss)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {} and Loss: {:.2f}".format(epoch, epoch_loss))


def first_stage_model_test(model, config, test_loader, res_eval, eval_parameters=None, test_type="eval", feature_type=1):
    """
    :param model:
    :param config:
    :param test_loader:
    :param res_eval:
    :param eval_parameters:
    :param test_type:
    :param feature_type:
    :return:
    """
    elem_feature_embed, result_feature_embed, bert_feature_embed = [], [], []
    
    assert test_type in {"eval", "gene"}, "[ERROR] test type error!"

    model.eval()
    if test_type == "eval":
        measure_file, model_path = eval_parameters
    else:
        gold_pair_label = eval_parameters[0]

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            input_ids, attn_mask, comparative_label, multi_label, result_label = data

            input_ids = torch.tensor(input_ids).long().to(config.device)
            attn_mask = torch.tensor(attn_mask).long().to(config.device)

            bert_feature, elem_feature, elem_output, result_output, sent_output = model(input_ids, attn_mask)
            """
            bert_feature: batch_size, seq_length, hidden_dim (768)
            elem_feature: batch_size, seq_length, len(val.norm_id_map) (5)
            elem_output: batch_size, 3, seq_length
            result_output: batch_size, seq_length
            sent_output: batch_size
            """
            if test_type == "eval":
                res_eval.add_data(elem_output, result_output, attn_mask)
                res_eval.add_sent_label(sent_output)
            else:
                res_eval.add_data(elem_output, result_output, attn_mask)
                elem_feature_embed.append(elem_feature)
                bert_feature_embed.append(bert_feature)
                res_eval.add_sent_label(sent_output)

    if test_type == "eval":
        res_eval.eval_model(measure_file, model, model_path, multi_elem_score=True)
    else:
        return res_eval.generate_elem_representation(
            gold_pair_label, torch.cat(elem_feature_embed, dim=0),
            torch.cat(bert_feature_embed, dim=0), feature_type=feature_type
        )

def first_stage_model_test_new(model, config, test_loader, res_eval, eval_parameters=None, test_type="eval", feature_type=1, max_len_list=None):
    """
    used for transformers classifier
    :param model:
    :param config:
    :param test_loader:
    :param res_eval:
    :param eval_parameters:
    :param test_type:
    :param feature_type:
    :return: 
    candidate_pair_col: list of (list of 4-element lists)
    comp_input: list of (list of (1 + subject_length + object_length + aspect_length + predicate_length) x feature_dim 2D lists)
    token_type_input: list of (list of (0 + 1*subject_length + 2*object_length + 3*aspect_length + 4*predicate_length) lists)
    make_pair_label: list of 0-1 lists
    elem_feature_embed: [N, 4, sequence_lengh, feature_dim] feature_dim = 5
    bert_feature_embed: [N, sequence_length, hidden_dim] hidden_dim=768
    """
    elem_feature_embed, result_feature_embed, bert_feature_embed = [], [], []
    elem_output_embed, result_output_embed = [], []
    attn_mask_list = []
    assert test_type in {"eval", "gene"}, "[ERROR] test type error!"

    model.eval()
    if test_type == "eval":
        measure_file, model_path = eval_parameters
    else:
        gold_pair_label = eval_parameters[0]

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            input_ids, attn_mask, comparative_label, multi_label, result_label = data

            input_ids = torch.tensor(input_ids).long().to(config.device)
            attn_mask = torch.tensor(attn_mask).long().to(config.device)

            bert_feature, elem_feature, elem_output, result_output, sent_output = model(input_ids, attn_mask)
            """
            bert_feature: batch_size, seq_length, hidden_dim (768)
            elem_feature: batch_size, 4, seq_length, len(val.norm_id_map) (5)
            elem_output: batch_size, 3, seq_length
            result_output: batch_size, seq_length
            sent_output: batch_size
            """
            if test_type == "eval":
                res_eval.add_data(elem_output, result_output, attn_mask)
                res_eval.add_sent_label(sent_output)
            else:
                attn_mask_list.append(attn_mask)
                res_eval.add_data(elem_output, result_output, attn_mask)
                elem_output_embed.append(elem_output)
                result_output_embed.append(result_output)
                elem_feature_embed.append(elem_feature)
                bert_feature_embed.append(bert_feature)
                res_eval.add_sent_label(sent_output)

    if test_type == "eval":
        res_eval.eval_model(measure_file, model, model_path, multi_elem_score=True)
    else:
        return res_eval.generate_elem_representation_new(
            gold_pair_label,
            torch.cat(bert_feature_embed, dim=0),
            torch.cat(elem_feature_embed, dim=0),
            feature_type=feature_type,
            max_len_list=max_len_list
        )


def pair_stage_model_train(model, optimizer, train_loader, config, epoch):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param config:
    :param epoch:
    :return:
    """
    model.train()
    epoch_loss, t = 0, 0
    for index, data in tqdm(enumerate(train_loader)):
        pair_representation, pair_label = data

        pair_representation = torch.tensor(pair_representation).float().to(config.device)
        pair_label = torch.tensor(pair_label).long().to(config.device)

        if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
            continue

        loss = model(pair_representation, pair_label)

        loss = torch.sum(loss)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {} and Loss: {:.2f}".format(epoch, epoch_loss))


def pair_stage_model_test(
        model, refer_params, config, test_loader, res_eval, eval_parameters=None, mode="pair", polarity=False, initialize=(False, False)):
    """
    :param model: the model
    :param test_loader: test data loader: [input_ids, attn_mask, pos_ids, predicate_label]
    :param config:
    :param res_eval: a Evaluation object
    :param eval_parameters:
    :param mode:
    :param polarity:
    :param initialize:
    :return:
    """
    model.eval()
    measure_file, model_path = eval_parameters[0], eval_parameters[1]

    refer_neighbor_decoder = refer_neighbooring_decoder.ReferNeighbooringStrategy(
        refer_features=refer_params[0],
        refer_labels=refer_params[1],
        num_class=refer_params[2],
        best_result=refer_params[3], 
        device=config.device
    )

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            pair_representation = data

            pair_representation = torch.tensor(pair_representation).float().to(config.device)

            # pair_out = model(pair_representation).view(-1)
            pair_out = refer_neighbor_decoder.predict(model, pair_representation, softmax=False).view(-1)

            if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
                pair_out = torch.zeros(pair_out.size())

            if mode == "pair":
                res_eval.add_pair_data(pair_out)
            else:
                res_eval.add_polarity_data(pair_out)

    res_eval.eval_model(measure_file, model, model_path, polarity=polarity, initialize=initialize)

def pair_stage_model_train_new(model, optimizer, train_loader, config, epoch):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param config:
    :param epoch:
    :return:
    """
    model.train()
    epoch_loss, t = 0, 0
    for index, data in tqdm(enumerate(train_loader)):
        comp_input, token_type_input, padding_mask_input, pair_label = data

        batch_size = len(padding_mask_input)
        seq_length = len(padding_mask_input[0])
        
        # if torch.equal(padding_mask_input, torch.zeros_like(token_type_input)):
        #     continue

        if padding_mask_input == [[1] + [0] * (seq_length - 1)] * batch_size:
            continue

        comp_input = torch.tensor(comp_input).float().to(config.device)
        token_type_input = torch.tensor(token_type_input).long().to(config.device)
        padding_mask_input = torch.tensor(padding_mask_input).bool().to(config.device)
        pair_label = torch.tensor(pair_label).long().to(config.device)

        loss = model(comp_input, token_type_input, padding_mask=padding_mask_input, label=pair_label)

        loss = torch.sum(loss)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {} and Loss: {:.2f}".format(epoch, epoch_loss))


def pair_stage_model_test_new(
        model, config, test_loader, res_eval, eval_parameters=None, mode="pair", polarity=False, initialize=(False, False)):
    """
    :param model: the model
    :param test_loader: test data loader: [input_ids, attn_mask, pos_ids, predicate_label]
    :param config:
    :param res_eval: a Evaluation object
    :param eval_parameters:
    :param mode:
    :param polarity:
    :param initialize:
    :return:
    """
    model.eval()
    measure_file, model_path = eval_parameters

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            print(f"test sentence: {index}")
            comp_input, token_type_input, padding_mask_input = data
            batch_size = len(padding_mask_input)
            seq_length = len(padding_mask_input[0])

            comp_input = torch.tensor(comp_input).float().to(config.device)
            token_type_input = torch.tensor(token_type_input).long().to(config.device)
            padding_mask = torch.tensor(padding_mask_input).bool().to(config.device)
        
        # if torch.equal(padding_mask_input, torch.zeros_like(token_type_input)):
        #     continue
            # for sub_index in range(len(comp_input)):
            #     print(f"sub_index test sentence: {sub_index}")
            #     each_comp_input = torch.tensor(comp_input[sub_index]).float().to(config.device)
            #     each_token_type_input = torch.tensor(token_type_input[sub_index]).long().to(config.device)
            #     print(f"each_comp_input: {each_comp_input.shape}", f"each_token_type_input: {each_token_type_input.shape}")

            pair_out = model(comp_input, token_type_input, padding_mask).view(-1)

            if padding_mask_input == [[1] + [0] * (seq_length - 1)] * batch_size:
                pair_out = torch.zeros(pair_out.size())
            # if torch.equal(each_token_type_input, torch.zeros_like(each_token_type_input)):
            #     pair_out = torch.zeros(pair_out.size())
            
            # pair_out_list.append(pair_out)

            # final_pair_out = torch.cat(pair_out_list, dim=-1).view(-1)
            if mode == "pair":
                res_eval.add_pair_data(pair_out)
            else:
                res_eval.add_polarity_data(pair_out)

    res_eval.eval_model(measure_file, model, model_path, polarity=polarity, initialize=initialize)

########################################################################################################################
# each stage model
########################################################################################################################
def first_stage_model_main(
        config, data_gene, data_loaders, comp_eval, model_parameters, optimizer_parameters, model_name):
    """
    :param config:
    :param data_gene:
    :param data_loaders:
    :param comp_eval:
    :param model_parameters:
    :param optimizer_parameters:
    :param model_name:
    :return:
    """
    train_loader, dev_loader, test_loader = data_loaders
    dev_comp_eval, test_comp_eval, global_comp_eval = comp_eval

    # define first stage model and optimizer
    MODEL2FN = {"bert": pipeline_model_utils.Baseline, "norm": pipeline_model_utils.LSTMModel}

    if config.model_mode == "bert":
        model = MODEL2FN[config.model_mode](config, model_parameters).to(config.device)
    else:
        # weight = shared_utils.get_pretrain_weight(
        #     config.path.GloVe_path, config.path.Word2Vec_path, data_gene.vocab
        # )
        weight = None
        model = MODEL2FN[config.model_mode](
            config, model_parameters, data_gene.vocab, weight).to(config.device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        optimizer_need_model = model.module
    else:
        optimizer_need_model = model

    OPTIM2FN = {"bert": optimizer_utils.Baseline_Optim, "norm": optimizer_utils.LSTMModel_Optim}
    optimizer = OPTIM2FN[config.model_mode](optimizer_need_model, optimizer_parameters)

    dev_parameters = ["./ModelResult/" + model_name + "/dev_elem_result.txt",
                      "./PreTrainModel/" + model_name + "/dev_model"]

    # train and test model.
    for epoch in range(config.epochs):
        first_stage_model_train(model, optimizer, train_loader, config, epoch)
        first_stage_model_test(model, config, dev_loader, dev_comp_eval, dev_parameters)

    print("==================test================")
    predicate_model = torch.load(dev_parameters[1])

    test_parameters = ["./ModelResult/" + model_name + "/test_elem_result.txt", None]

    first_stage_model_test(predicate_model, config, test_loader, test_comp_eval, test_parameters)

    test_comp_eval.print_elem_result(
        data_gene.test_data_dict['input_ids'], data_gene.test_data_dict['attn_mask'],
        "./ModelResult/" + model_name + "/test_result_file" + ".txt", drop_span=False
    )

    # add average measure.
    shared_utils.calculate_average_measure(test_comp_eval, global_comp_eval)


def pair_stage_model_main(config, pair_representation, make_pair_label, pair_eval, polarity_col,
                          model_parameters, optimizer_parameters, model_name, feature_type, grid_search_params):
    """

    :param config:
    :param pair_representation:
    :param make_pair_label:
    :param pair_eval:
    :param polarity_col:
    :param model_parameters:
    :param optimizer_parameters:
    :param model_name:
    :param feature_type:
    :return:
    """
    train_pair_representation, dev_pair_representation, test_pair_representation = pair_representation
    train_make_pair_label, dev_make_pair_label, test_make_pair_label = make_pair_label
    dev_pair_eval, test_pair_eval, global_pair_eval = pair_eval
    train_polarity_representation, train_polarity_col, dev_polarity_gold_representation, dev_polarity_gold_label = polarity_col
    alpha, beta, topk, main_metric = grid_search_params
    print("finish second model data generate")

    # get pair loader
    train_pair_loader = data_loader_utils.get_loader([train_pair_representation, train_make_pair_label], config.batch_size)
    dev_pair_loader = data_loader_utils.get_loader([dev_pair_representation], 1)
    test_pair_loader = data_loader_utils.get_loader([test_pair_representation], 1)

    # get polarity data loader.
    train_polarity_loader = data_loader_utils.get_loader([train_polarity_representation, train_polarity_col], config.batch_size)
    shared_utils.write_pickle([train_polarity_representation, train_polarity_col], 
                              '/workspace/nlplab/kienvt/COQE/tmp/polarity_train_data')

    pair_weight = torch.tensor([model_parameters['factor'], 1.0]).float()
    polarity_weight = torch.tensor([model_parameters['penalty']] * len(config.val.polarity_col)).float() 
    if config.val.penalty_index in range(len(config.val.polarity_col)):
        polarity_weight[config.val.penalty_index] = 1.0

    feature_dim = [4 * (5 + config.hidden_size), 4 * 5, 4 * config.hidden_size]
    pair_feature_dim = feature_dim[feature_type]

    # define pair and polarity model.
    print('stage 2 model', config.stage2_clf)
    print('stage 3 model', config.stage3_clf)
    if config.stage2_clf == 'logistic':
        pair_model = copy.deepcopy(
            pipeline_model_utils.LogisticClassifier(config, pair_feature_dim, 2, weight=pair_weight).to(config.device)
        )
    else:
        pair_model = copy.deepcopy(
            pipeline_model_utils.MLPClasifier(config, pair_feature_dim, 2, weight=pair_weight).to(config.device)
        )

    if config.stage3_clf == 'logistic':
        polarity_model = copy.deepcopy(
            pipeline_model_utils.LogisticClassifier(config, pair_feature_dim, \
                                                    len(config.val.polarity_col), \
                                                    weight=polarity_weight).to(config.device)
        )
    else:
        polarity_model = copy.deepcopy(
            pipeline_model_utils.MLPClasifier(config, pair_feature_dim, \
                                              len(config.val.polarity_col),\
                                                weight=polarity_weight).to(config.device)
        )

    if torch.cuda.device_count() > 1:
        pair_model = nn.DataParallel(pair_model)
        polarity_model = nn.DataParallel(polarity_model)
        
        if config.stage2_clf == 'logistic':
            pair_optimizer = optimizer_utils.Logistic_Optim(pair_model.module, optimizer_parameters)
        else:
            pair_optimizer = optimizer_utils.MLP_Optim(pair_model.module, optimizer_parameters)
        
        if config.stage3_clf == 'logistic':
            polarity_optimizer = optimizer_utils.Logistic_Optim(polarity_model.module, optimizer_parameters)
        else:
            polarity_optimizer = optimizer_utils.MLP_Optim(polarity_model.module, optimizer_parameters)
    else:
        if config.stage2_clf == 'logistic':
            pair_optimizer = optimizer_utils.Logistic_Optim(pair_model, optimizer_parameters)
        else:
            pair_optimizer = optimizer_utils.MLP_Optim(pair_model, optimizer_parameters)

        if config.stage3_clf == 'logistic': 
            polarity_optimizer = optimizer_utils.Logistic_Optim(polarity_model, optimizer_parameters)
        else:
            polarity_optimizer = optimizer_utils.MLP_Optim(polarity_model, optimizer_parameters)

    dev_pair_parameters = ["./ModelResult/" + model_name + "/dev_pair_result.txt",
                           "./PreTrainModel/" + model_name + "/dev_pair_model",
                           "./PreTrainModel/" + model_name + "/pair_refer_params",
                           "./ModelResult/" + model_name + "/pair_refer_result.txt"]

    dev_polarity_parameters = ["./ModelResult/" + model_name + "/dev_polarity_result.txt",
                               "./PreTrainModel/" + model_name + "/dev_polarity_model",
                               "./PreTrainModel/" + model_name + "/polarity_refer_params",
                               "./ModelResult/" + model_name + "/polarity_refer_result.txt"]
    
    pair_refer_neighbor_decoder = refer_neighbooring_decoder.ReferNeighbooringStrategy(
        refer_features=train_pair_representation,
        refer_labels=train_make_pair_label,
        num_class=2,
        alpha=alpha,
        beta=beta,
        topk=topk,
        main_metric=main_metric,
        batch_size=config.batch_size,
        device=config.device
    )

    polarity_refer_neighbor_decoder = refer_neighbooring_decoder.ReferNeighbooringStrategy(
        refer_features=train_polarity_representation,
        refer_labels=train_polarity_col,
        num_class=len(config.val.polarity_col),
        alpha=alpha,
        beta=beta,
        topk=topk,
        main_metric=main_metric,
        batch_size=config.batch_size,
        device=config.device
    )

    for epoch in range(config.epochs):
        pair_stage_model_train(pair_model, pair_optimizer, train_pair_loader, config, epoch)
        pair_stage_model_test(
            pair_model, pair_refer_neighbor_decoder.parameters(), config, dev_pair_loader, dev_pair_eval,
            dev_pair_parameters, mode="pair", polarity=False, initialize=(False, True)
        )
    pair_refer_neighbor_decoder.fit(pair_model, dev_pair_representation, dev_make_pair_label, dev_pair_parameters[3])
    shared_utils.write_pickle(pair_refer_neighbor_decoder.parameters(), dev_pair_parameters[2])
    # get optimize pair model.
    predict_pair_model = torch.load(dev_pair_parameters[1])
    test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
    pair_stage_model_test(
        predict_pair_model, pair_refer_neighbor_decoder.parameters(), config, dev_pair_loader, dev_pair_eval,
        test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
    )

    # get representation by is_pair label filter.
    dev_polarity_representation = cpc.get_after_pair_representation(dev_pair_eval.y_hat, dev_pair_representation)
    dev_polarity_loader = data_loader_utils.get_loader([dev_polarity_representation], 1)
    shared_utils.clear_optimize_measure(dev_pair_eval)

    for epoch in range(config.epochs):
        pair_stage_model_train(polarity_model, polarity_optimizer, train_polarity_loader, config, epoch)
        pair_stage_model_test(
            polarity_model, polarity_refer_neighbor_decoder.parameters(), config, dev_polarity_loader, dev_pair_eval,
            dev_polarity_parameters, mode="polarity", polarity=True, initialize=(True, False)
        )
    polarity_refer_neighbor_decoder.fit(polarity_model, dev_polarity_gold_representation, 
                                            dev_polarity_gold_label, dev_polarity_parameters[3], flatten=False)
    shared_utils.write_pickle(polarity_refer_neighbor_decoder.parameters(), dev_polarity_parameters[2])
    print("==================test================")
    predict_pair_model = torch.load(dev_pair_parameters[1])
    predict_polarity_model = torch.load(dev_polarity_parameters[1])

    test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
    test_polarity_parameters = ["./ModelResult/" + model_name + "/test_polarity_result.txt", None]

    pair_stage_model_test(
        predict_pair_model, pair_refer_neighbor_decoder.parameters(), config, test_pair_loader, test_pair_eval,
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

    pair_stage_model_test(
        predict_polarity_model, polarity_refer_neighbor_decoder.parameters(), config, test_polarity_loader, test_pair_eval,
        test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True)
    )

    # add average measure.
    shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)

def pair_stage_model_main_new(config, pair_representation, make_pair_label, pair_eval, polarity_col,
                          model_parameters, optimizer_parameters, model_name, feature_type):
    """

    :param config:
    :param pair_representation:
    :param make_pair_label:
    :param pair_eval:
    :param polarity_col:
    :param model_parameters:
    :param optimizer_parameters:
    :param model_name:
    :param feature_type:
    :return:
    """
    train_pair_comp_input, train_pair_token_type_input, train_pair_padding_mask_input, \
    dev_pair_comp_input, dev_pair_token_type_input, dev_pair_padding_mask_input, \
    test_pair_comp_input, test_pair_token_type_input, test_pair_padding_mask_input = pair_representation
    
    train_make_pair_label, dev_make_pair_label, test_make_pair_label = make_pair_label
    dev_pair_eval, test_pair_eval, global_pair_eval = pair_eval
    train_polarity_comp_input, train_polarity_token_type_input, train_polarity_padding_mask_input, train_polarity_col = polarity_col
    

    print("finish second model data generate")

    # get pair loader
    train_pair_loader = data_loader_utils.get_loader([train_pair_comp_input, train_pair_token_type_input, train_pair_padding_mask_input, train_make_pair_label], 16)
    dev_pair_loader = data_loader_utils.get_loader([dev_pair_comp_input, dev_pair_token_type_input, dev_pair_padding_mask_input], 1)
    test_pair_loader = data_loader_utils.get_loader([test_pair_comp_input, test_pair_token_type_input, test_pair_padding_mask_input], 1)

    # get polarity data loader.
    train_polarity_loader = data_loader_utils.get_loader([train_polarity_comp_input, train_polarity_token_type_input, train_polarity_padding_mask_input, train_polarity_col], 1)
    shared_utils.write_pickle([train_polarity_comp_input, train_polarity_token_type_input, train_polarity_padding_mask_input, train_polarity_col], 
                              '/workspace/nlplab/kienvt/COQE/tmp/new_polarity_train_data')

    pair_weight = torch.tensor([model_parameters['factor'], 1]).float()

    feature_dim = [(5 + config.hidden_size), 5, config.hidden_size]
    pair_feature_dim = feature_dim[feature_type]

    # define pair and polarity model.
    print('stage 2 model', config.stage2_clf)
    print('stage 3 model', config.stage3_clf)

    pair_model = copy.deepcopy(
        pipeline_model_utils.transformer_classifier(config, pair_feature_dim, 4, 2, 2,dropout=0.1, weight=pair_weight).to(config.device)
    )
    polarity_model = copy.deepcopy(
        pipeline_model_utils.transformer_classifier(config, pair_feature_dim, 4, 2, len(config.val.polarity_col), dropout=0.1, weight=pair_weight).to(config.device)
    )

    if torch.cuda.device_count() > 1:
        pair_model = nn.DataParallel(pair_model)
        polarity_model = nn.DataParallel(polarity_model)
        
        pair_optimizer = optimizer_utils.Transformer_cls_Optim(pair_model, optimizer_parameters)
        polarity_optimizer = optimizer_utils.Transformer_cls_Optim(polarity_model, optimizer_parameters)

    else:
        pair_optimizer = optimizer_utils.Transformer_cls_Optim(pair_model, optimizer_parameters)
        polarity_optimizer = optimizer_utils.Transformer_cls_Optim(polarity_model, optimizer_parameters)


    tmp_path_string = "/workspace/nlplab/kienvt/COQE/tmp/transformers_cls_result"
    # dev_pair_parameters = ["./ModelResult/" + model_name + "/new_dev_pair_result.txt",
    #                        "./PreTrainModel/" + model_name + "/new_dev_pair_model"]

    # dev_polarity_parameters = ["./ModelResult/" + model_name + "/new_dev_polarity_result.txt",
    #                            "./PreTrainModel/" + model_name + "/new_dev_polarity_model"]

    dev_pair_parameters = [tmp_path_string + "/ModelResult/" + model_name + "/new_dev_pair_result.txt",
                           tmp_path_string + "/PreTrainModel/" + model_name + "/new_dev_pair_model"]

    dev_polarity_parameters = [tmp_path_string + "/ModelResult/" + model_name + "/new_dev_polarity_result.txt",
                               tmp_path_string + "/PreTrainModel/" + model_name + "/new_dev_polarity_model"]
    
    pair_model_num = sum(p.numel() for p in pair_model.parameters() if p.requires_grad)
    polarity_model_num = sum(p.numel() for p in polarity_model.parameters() if p.requires_grad)

    print("pair_model_size: ", pair_model_num)
    print("polariy_model_size", polarity_model_num)

    for epoch in range(50):
        pair_stage_model_train_new(pair_model, pair_optimizer, train_pair_loader, config, epoch)
        pair_stage_model_test_new(
            pair_model, config, dev_pair_loader, dev_pair_eval,
            dev_pair_parameters, mode="pair", polarity=False, initialize=(False, True)
        )

    # get optimize pair model.
    predict_pair_model = torch.load(dev_pair_parameters[1])
    test_pair_parameters = ["./ModelResult/" + model_name + "/new_test_pair_result.txt", None]
    pair_stage_model_test_new(
        predict_pair_model, config, dev_pair_loader, dev_pair_eval,
        test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
    )

    # get representation by is_pair label filter.
    # dev_polarity_comp_input, dev_polarity_token_type_input, dev_polarity_padding_mask_input = cpc.get_after_pair_representation_new(dev_pair_eval.y_hat, dev_pair_comp_input, dev_pair_token_type_input, dev_pair_padding_mask_input)
    # dev_polarity_loader = data_loader_utils.get_loader([dev_polarity_comp_input, dev_polarity_token_type_input], 1)
    dev_polarity_comp_input = dev_pair_comp_input
    dev_polarity_token_type_input = dev_pair_token_type_input
    dev_polarity_padding_mask_input = dev_pair_padding_mask_input
    dev_polarity_loader = data_loader_utils.get_loader([dev_polarity_comp_input, dev_polarity_token_type_input, dev_polarity_padding_mask_input], 1)
    
    shared_utils.clear_optimize_measure(dev_pair_eval)

    for epoch in range(40):
        pair_stage_model_train_new(polarity_model, polarity_optimizer, train_polarity_loader, config, epoch)
        pair_stage_model_test_new(
            polarity_model, config, dev_polarity_loader, dev_pair_eval,
            dev_polarity_parameters, mode="polarity", polarity=True, initialize=(True, False)
        )

    print("==================test================")
    predict_pair_model = torch.load(dev_pair_parameters[1])
    predict_polarity_model = torch.load(dev_polarity_parameters[1])

    # test_pair_parameters = ["./ModelResult/" + model_name + "/new_test_pair_result.txt", None]
    # test_polarity_parameters = ["./ModelResult/" + model_name + "/new_test_pair_result.txt", None]

    test_pair_parameters = [tmp_path_string + "/ModelResult/" + model_name + "/new_test_pair_result.txt", None]
    test_polarity_parameters = [tmp_path_string + "/ModelResult/" + model_name + "/new_test_pair_result.txt", None]

    pair_stage_model_test_new(
        predict_pair_model, config, test_pair_loader, test_pair_eval,
        test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
    )

    shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)
    global_pair_eval.avg_model(tmp_path_string + "/ModelResult/" + model_name + "/test_pair_result.txt")
    global_pair_eval.store_result_to_csv([model_name], "result.csv")

    shared_utils.clear_global_measure(global_pair_eval)
    shared_utils.clear_optimize_measure(test_pair_eval)

    # create polarity representation and data loader.
    test_polarity_comp_input = test_pair_comp_input
    test_polarity_token_type_input = test_pair_token_type_input
    test_polarity_padding_mask_input = test_pair_padding_mask_input
    test_polarity_loader = data_loader_utils.get_loader([test_polarity_comp_input, test_polarity_token_type_input, test_polarity_padding_mask_input], 1)
    
    # test_polarity_comp_input, test_polarity_token_type_input = cpc.get_after_pair_representation_new(test_pair_eval.y_hat, test_pair_comp_input, test_pair_token_type_input)
    # test_polarity_loader = data_loader_utils.get_loader([test_polarity_comp_input, test_polarity_token_type_input], 1)

    pair_stage_model_test_new(
        predict_polarity_model, config, test_polarity_loader, test_pair_eval,
        test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True)
    )


    # add average measure.
    shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)

