import os
import numpy as np
from data_utils.label_parse import LabelParser
from data_utils import shared_utils
from data_utils import current_program_code as cpc
from open_source_utils import stanford_utils
from transformers import AutoTokenizer

class DataGenerator(object):
    def __init__(self, config):
        """
        :param config: a program configure
        :return: input_ids, attn_mask, pos_ids, dep_matrix, dep_label_matrix, label_ids
        """
        self.config = config
        self.vocab, self.pos_dict = {"<pad>": 1, "<s>": 0, "</s>": 2}, {"<pad>": 1}
        self.vocab_index, self.pos_index = 5, 5
        self.token_max_len, self.char_max_len = -1, -1

        # store some data using in model
        self.train_data_dict, self.dev_data_dict, self.test_data_dict = {}, {}, {}
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.path.bert_model_path)

        self.elem_col = ["entity_1", "entity_2", "aspect", "result"]
        self.segment_syllable_mapping = []

    def create_data_dict(self, data_path, data_type, label_path=None, rm_noise=False):
        self.segment_syllable_mapping = []

        """
        :param data_path: sentence file path
        :param data_type:
        :param label_path: label file path
        :return: a data dict with many parameters
        """
        data_dict = {}

        sent_col, sent_label_col, label_col = cpc.read_standard_file(data_path)
        for line in sent_col:
            segment_index = 1
            syllable_index = 1
            segment_tokens = line.strip(" ").split(" ")
            segment_syllable_mapping_dict = {}
            for segment_token in segment_tokens:
                syllable_token_list = []
                for syllable in segment_token.split("_"):
                    syllable_token_list.append(f"{syllable_index}&&{syllable}")
                    syllable_index += 1
                segment_syllable_mapping_dict[f"{segment_index}&&{segment_token}"] = syllable_token_list
                segment_index += 1
            self.segment_syllable_mapping.append(segment_syllable_mapping_dict)

        # tmp_file = "/workspace/nlplab/kienvt/COQE/tmp/label_col_1.txt"
        # tmp_string = ""
        # for comp_list in label_col:
        #     for comp in comp_list:
        #         tmp_string = tmp_string + comp + "\n"

        # with open(tmp_file, "w") as tmp_file:
        #     tmp_file.write(tmp_string)
        # tmp_file.close()

        if rm_noise:
            sent_col, sent_label_col, label_col = self.remove_noise(sent_col, sent_label_col, label_col, len_under=50)

        label_col = self.convert_label(label_col, label_tags=self.config.val.polarity_col, dim=2)

        LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])
        label_col, tuple_pair_col = LP.parse_sequence_label("&&", sent_col, file_type="eng")
        data_dict['standard_token'] = self.get_token_col(sent_col, split_symbol=' ', dim=1)
        shared_utils.write_pickle(data_dict, self.config.path.pre_process_data[data_type])

        self.token_max_len = max(self.token_max_len, shared_utils.get_max_token_length(data_dict['standard_token']))

        data_dict['label_col'] = label_col
        data_dict['comparative_label'] = sent_label_col
        if self.config.model_mode == 'bert':
            data_dict['bert_token'], mapping_col = self.token_mapping_bert(data_dict['standard_token'])
            label_col = cpc.convert_eng_label_dict_by_mapping(label_col, mapping_col)
            data_dict['token_mapping_col'] = mapping_col
            tuple_pair_col = cpc.convert_eng_tuple_pair_by_mapping(tuple_pair_col, mapping_col)

            data_dict['input_ids'] = shared_utils.bert_data_transfer(
                self.bert_tokenizer,
                data_dict['bert_token'],
                "tokens"
            )

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2
        
        else:

            self.vocab, self.vocab_index = shared_utils.update_vocab(
                data_dict['standard_token'],
                self.vocab,
                self.vocab_index,
                dim=2
            )

            data_dict['input_ids'] = shared_utils.transfer_data(data_dict['standard_token'], self.vocab, dim=1)

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2
        data_dict['tuple_pair_col'] = tuple_pair_col
        print("convert pair number: ", cpc.get_tuple_pair_num(data_dict['tuple_pair_col']))

        token_col = data_dict['standard_token'] if self.config.model_mode == "norm" else data_dict['bert_token']

        data_dict['attn_mask'] = shared_utils.get_mask(token_col, dim=1)

        special_symbol = False

        data_dict['multi_label'], data_dict['result_label'], data_dict['polarity_label'] = \
            cpc.elem_dict_convert_to_multi_sequence_label(
                token_col, label_col, self.config.val.polarity_dict, special_symbol=special_symbol
            )
        
        ################################################################################################################
        # tags to ids
        ################################################################################################################

        data_dict['multi_label'] = shared_utils.transfer_data(
            data_dict['multi_label'],
            self.config.val.norm_id_map,
            dim=2
        )

        data_dict['result_label'] = shared_utils.transfer_data(
            data_dict['result_label'],
            self.config.val.norm_id_map,
            dim=1
        )

        return data_dict

    def generate_data(self, rm_noise=(True, False, False)):
        self.train_data_dict = self.create_data_dict(
            self.config.path.standard_path['train'],
            "train",
            rm_noise=rm_noise[0]
        )

        self.dev_data_dict = self.create_data_dict(
            self.config.path.standard_path['dev'],
            "dev",
            rm_noise=rm_noise[1]
        )

        self.test_data_dict = self.create_data_dict(
            self.config.path.standard_path['test'],
            "test",
            rm_noise=rm_noise[2]
        )

        self.train_data_dict = self.padding_data_dict(self.train_data_dict)
        self.dev_data_dict = self.padding_data_dict(self.dev_data_dict)
        self.test_data_dict = self.padding_data_dict(self.test_data_dict)

        self.train_data_dict = self.data_dict_to_numpy(self.train_data_dict)
        self.dev_data_dict = self.data_dict_to_numpy(self.dev_data_dict)
        self.test_data_dict = self.data_dict_to_numpy(self.test_data_dict)
    
    def generate_infer_data(self, test_path):
        self.test_data_dict = self.create_data_dict(
            test_path,
            "test",
            rm_noise=False
        )
        self.test_data_dict = self.padding_data_dict(self.test_data_dict)
        self.test_data_dict = self.data_dict_to_numpy(self.test_data_dict)

    def padding_data_dict(self, data_dict):
        """
        :param data_dict:
        :return:
        """
        pad_key_ids = { 0: ["input_ids"],
                        1: ["multi_label"],
                        2: ["attn_mask", "result_label"] }

        cur_max_len = self.char_max_len

        param = [{"max_len": cur_max_len, "dim": 1, "pad_num": self.bert_tokenizer.pad_token_id, "data_type": "norm"},
                 {"max_len": cur_max_len, "dim": 2, "pad_num": 0, "data_type": "norm"},
                 {"max_len": cur_max_len, "dim": 1, "pad_num": 0, "data_type": "norm"}]

        for index, key_col in pad_key_ids.items():
            for key in key_col:
                data_dict[key] = shared_utils.padding_data(
                    data_dict[key],
                    max_len=param[index]['max_len'],
                    dim=param[index]['dim'],
                    padding_num=param[index]['pad_num'],
                    data_type=param[index]['data_type']
                )

        return data_dict
    
    @staticmethod
    def data_dict_to_numpy(data_dict):
        """
        :param data_dict:
        :return:
        """
        key_col = ["input_ids", "attn_mask", "tuple_pair_col", "result_label", "multi_label", "comparative_label"]

        for key in key_col:
            data_dict[key] = np.array(data_dict[key])
            print(key, data_dict[key].shape)

        data_dict['comparative_label'] = np.array(data_dict['comparative_label']).reshape(-1, 1)

        return data_dict

    def get_token_col(self, sent_col, split_symbol=None, dim=1):
        if dim==0:
            if split_symbol is not None:
                return shared_utils.split_string(sent_col, split_symbol)
            else:
                return self.bert_tokenizer.tokenize('<s> ' + sent_col + ' </s>')
        else:
            token_col = []
            for index in range(len(sent_col)):
                token_col.append(self.get_token_col(sent_col[index], split_symbol, dim-1))
            
            return token_col
        
    def token_mapping_bert(self, gold_token_col):
        def flatten_ll(ll):
            return [item for l in ll for item in l]
        
        bert_token_col = []
        mapping_col = []
        for index in range(len(gold_token_col)):
            seq_gold_token = gold_token_col[index]
            seq_bert_token = [self.bert_tokenizer.tokenize(token) for token in seq_gold_token]
            bert_token_col.append(['<s>'] + flatten_ll(seq_bert_token) + ['</s>'])
            seq_map, bert_idx = {}, 1
            for i in range(len(seq_bert_token)):
                seq_map[i] = list(range(bert_idx, len(seq_bert_token[i]) + bert_idx))
                bert_idx += len(seq_bert_token[i])
            mapping_col.append(seq_map)
        
        return bert_token_col, mapping_col
    

    def convert_label(self, label_col, label_tags, dim=2):
        relabel_col = []
        if dim == 0:
            for idx, tag in enumerate(label_tags):
                label_col = label_col.replace(f'[{tag}]', f'[{idx-1}]')
            return label_col
        elif dim == 1:
            new_item = []
            for item in label_col:
                new_item.append(self.convert_label(item, label_tags, dim-1))
            return new_item
        else:
            for item in label_col:
                relabel_col.append(self.convert_label(item, label_tags, dim-1))
        return relabel_col


    def remove_noise(self, sent_col, sent_label_col, label_col, len_under=20):
        print('==========================================')
        assert len(sent_col) == len(sent_label_col) and len(sent_col) == len(label_col)
        print(len(sent_col), len(sent_label_col), len(label_col))
        title_idx = [i for i in range(len(sent_col)) if sent_col[i].lower().startswith('title')]
        altdes_idx = [i for i in range(len(sent_col)) if sent_col[i].lower().startswith('alt') or sent_col[i].lower().startswith('des')]
        len_under_idx = [i for i in range(len(sent_col)) if len(sent_col[i]) <= len_under]

        title_with_label_rate = len([i for i in title_idx if sent_label_col[i]==1]) / len(title_idx)
        altdes_with_label_rate = len([i for i in altdes_idx if sent_label_col[i]==1]) / len(altdes_idx)
        len_under_with_label_rate = len([i for i in len_under_idx if sent_label_col[i]==1]) / len(len_under_idx)

        print('Title with label rate:', title_with_label_rate)
        print('alt/des with label rate:', altdes_with_label_rate)
        print(f'Len <{len_under} with label rate:', len_under_with_label_rate)

        title_idx_to_rm = [i for i in title_idx if sent_label_col[i]==0]
        altdes_idx_to_rm = [i for i in altdes_idx if sent_label_col[i]==0]
        len_under_idx_to_rm = [i for i in len_under_idx if sent_label_col[i]==0]

        idx_to_rm = np.unique(title_idx_to_rm + altdes_idx_to_rm + len_under_idx_to_rm).tolist()
        print(f"Remove title sent, alt/des sent, len_u{len_under} sent without label:", len(idx_to_rm))
        sent_col = [sent_col[i] for i in range(len(sent_col)) if i not in idx_to_rm]
        sent_label_col = [sent_label_col[i] for i in range(len(sent_label_col)) if i not in idx_to_rm]
        label_col = [label_col[i] for i in range(len(label_col)) if i not in idx_to_rm]
        assert len(sent_col) == len(sent_label_col) and len(sent_col) == len(label_col)
        print(len(sent_col), len(sent_label_col), len(label_col))
        print('==========================================')
        return sent_col, sent_label_col, label_col