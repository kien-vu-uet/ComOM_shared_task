import argparse
import os
import py_vncorenlp


def TerminalParser():
    # define parse parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', help='random seed', type=int, default=2021)

    # lstm parameters setting
    parser.add_argument('--input_dir', help='the type of data set', default=None)
    """
    input_dir: "/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM/raw/Version 2/VLSP2023_ComOM_public_test_nolabel"
    """
    parser.add_argument('--output_dir', help='the type of data set', default=None)
    """
    output_dir: "/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM/format"
                "/workspace/nlplab/kienvt/COQE/data/VLSP23-ComOM-Segmented/format"
    """
    args = parser.parse_args()

    return args

def get_necessary_parameters(args):
    """
    :param args:
    :return:
    """
    param_dict = {"input_dir": args.input_dir,
                  "output_dir": args.output_dir}

    return param_dict

def main():
    py_vncorenlp.download_model(save_dir='/workspace/nlplab/kienvt/COQE/pretrain_model/vncorenlp')
    segmenter = py_vncorenlp.VnCoreNLP(save_dir='/workspace/nlplab/kienvt/COQE/pretrain_model/vncorenlp')
    args = TerminalParser()
    config_params = get_necessary_parameters(args)
    print(config_params)
    input_file_list = os.listdir(config_params["input_dir"])

    for input_file in input_file_list:
        input_file_link = config_params["input_dir"] + "/{}".format(input_file)
        print(input_file_link)
        raw_sent_list, normalized_sent_list = [], []
        with open(input_file_link) as f_open:
            lines = f_open.readlines()
            for line in lines:
                if line != "\n":
                    sent_pair = line.split("\t")
                    # print(sent_pair)
                    raw_sent_list.append(sent_pair[0].strip("\n"))
                    if config_params["output_dir"].lower().find("segmented") != -1:
                        normalized_sent = " ".join(segmenter.word_segment(sent_pair[1].strip("\n")))
                        normalized_sent_list.append(normalized_sent)
                    else:
                        normalized_sent_list.append(sent_pair[1].strip("\n"))
        f_open.close()
        formated_sent_list = []
        for normalized_sent in normalized_sent_list:
            normalized_sent = normalized_sent + " \t" + "0"
            formated_sent_list.append(normalized_sent)
            formated_sent_list.append("[[];[];[];[];[]]")
        format_content = "\n".join(formated_sent_list)
        raw_content = "\n".join(raw_sent_list)
        
        if not os.path.exists(config_params["output_dir"] + "/no_label/"):
            os.mkdir(config_params["output_dir"] + "/no_label/")

        if not os.path.exists(config_params["output_dir"] + "/raw/"):
            os.mkdir(config_params["output_dir"] + "/raw/")

        format_output_file = config_params["output_dir"] + "/no_label" + "/{}".format(input_file)
        raw_output_file = config_params["output_dir"] + "/raw" + "/{}".format(input_file)
        with open(format_output_file, "w") as f_a:
            f_a.write(format_content)
        f_a.close()

        with open(raw_output_file, "w") as f_b:
            f_b.write(raw_content)
        f_b.close()


if __name__ == "__main__":
    main()