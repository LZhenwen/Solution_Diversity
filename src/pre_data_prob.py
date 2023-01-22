import random
import json
import copy
import re
import numpy as np
import time
from src.expressions_transfer import from_infix_to_prefix
from fractions import Fraction
import torch
from torch.nn import functional
from transformers import BertTokenizer

PAD_token = 0

def seg_(string):
    op_list = ['+', '-', '*', '/', '(', ')']
    out_list = []
    start_pos = 0
    #end_pos = 0
    for idx, token in enumerate(string):
        if token in op_list:
            out_list.append(token)
            start_pos = idx + 1
        else:
            if (idx == len(string) - 1) or (string[idx + 1] in op_list):
                out_list.append(string[start_pos:(idx + 1)])
    return out_list

class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

def raw_data(filename, id_list):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        data_d = json.loads(js)
        if data_d["id"] in id_list:
        #if "千米/小时" in data_d["equation"]:
        #    data_d["equation"] = data_d["equation"][:-5]
        #data_d["type"] = "ape"
            data.append(data_d)
        js = ""

    return data

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


# remove the superfluous brackets
def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y


def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    out_data = {}
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = remove_brackets(xt)
                    y = temp["sQuestion"]
                    seg = y.strip().split(" ")
                    temp_y = ""
                    for s in seg:
                        if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                            temp_y += s[:-1] + " " + s[-1:] + " "
                        else:
                            temp_y += s + " "
                    temp["sQuestion"] = temp_y[:-1]
                    out_data[temp["iIndex"]] = temp
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = remove_brackets(x[2:])
                y = temp["sQuestion"]
                seg = y.strip().split(" ")
                temp_y = ""
                for s in seg:
                    if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
                        temp_y += s[:-1] + " " + s[-1:] + " "
                    else:
                        temp_y += s + " "
                temp["sQuestion"] = temp_y[:-1]
                out_data[temp["iIndex"]] = temp
                continue
    return out_data

# for testing equation
# def out_equation(test, num_list):
#     test_str = ""
#     for c in test:
#         if c[0] == "N":
#             x = num_list[int(c[1:])]
#             if x[-1] == "%":
#                 test_str += "(" + x[:-1] + "/100.0" + ")"
#             else:
#                 test_str += x
#         elif c == "^":
#             test_str += "**"
#         elif c == "[":
#             test_str += "("
#         elif c == "]":
#             test_str += ")"
#         else:
#             test_str += c
#     return test_str

def transfer_num_boost(data):  # transfer num into "NUM"
    out_voc = ['*', '-', '+', '/', '^', '1', '3.14', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14']
    start = time.time()
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    count_dict = {}
    pre_labels = open('pre_labels.txt','r')
    label_dict = {}
    for lines in pre_labels:
        lines = lines.split(',')
        if lines[1].count('3.14') > 1 or lines[1].count('N') == 0:
            continue
        if lines[1].count('3.14') > 0 and lines[1].count('*') == 0:
            continue
        eqs = seg_(lines[1].strip())
        label_dict[str(int(lines[0]))] = (eqs)
    # count_dict = {}
    total = 0
    success = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        answer = d["ans"]
        fractions = re.findall("\d+\(\d+\/\d+\)", answer)
        if len(fractions):
            answer = answer.replace("(", "+(")
        #equations = d["equation"][2:]
        id2 = d["id"]
        i = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
                i += 1
            else:
                input_seq.append(s)
        nums_fraction = []
        if '=' in input_seq:
            if '+' in input_seq or '-' in input_seq or '*' in input_seq or '/' in input_seq:
                continue
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        #print(label_dict)
        
        num_pos = []
        op = ['*', '-', '+', '/']
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)
        nums = [str(i) for i in nums]
        assert len(nums) == len(num_pos)

        #print(final_candi_list)
        if str(id2) in label_dict:
            #print([input_seq[i] for i in num_pos])
            pairs.append((input_seq, from_infix_to_prefix(label_dict[str(id2)]), nums, num_pos, answer, id2))
        else:
            pairs.append((input_seq, [], nums, num_pos, answer, id2))
        string = len(nums)
        #print(string)
        #string2 = str((len(out_seq)))
        if string not in count_dict.keys():
            count_dict[string] = 0
        else:
            count_dict[string] = count_dict[string] + 1
    return pairs

def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
        answer = d["ans"]
        id2 = d["id"]
        fractions = re.findall("\d+\(\d+\/\d+\)", answer)
        if len(fractions):
            answer = answer.replace("(", "+(")
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        assert max(num_pos) <= len(input_seq)
        #print(d['id'])
        if '=' in input_seq:
            if '+' in input_seq or '-' in input_seq or '*' in input_seq or '/' in input_seq:
                continue
        if int(d['id']) == 12094 or int(d['id']) == 13550:
            continue
        seg_list = process(out_seq)
        #sub_aug_list = []
        replace_list = []
        for idx, segment in enumerate(seg_list):
            if isinstance(segment, list):
                #print(segment)
                s = aug(process(segment[1:-1]))
                for ii in s:
                    temp = ['('] + ii + [')']
                    if temp != segment:
                        #seg_copy = copy.deepcopy(seg_list)
                        #seg_copy[idx] = temp
                        replace_list.append((segment ,temp))
                        if len(replace_list) > 5:
                            break
        #print(len(replace_list))
        #replace_list = replace_list[:4]
        kk = 2 ** len(replace_list)
        all_seg_list = []
        if replace_list:
            for i in range(kk):
                aug_seg = copy.deepcopy(seg_list)
                for idx, ii in enumerate(str(bin(i))[2:]):
                    if ii == '1':
                        aug_seg=[replace_list[idx][1] if iii ==replace_list[idx][0] else iii for iii in aug_seg]
                all_seg_list.append(aug_seg)
        else:
            all_seg_list.append(seg_list)
        #print(all_seg_list)
        #input()
        #continue
        aug_list = []
        for s_list in all_seg_list:
            a_list = aug(s_list)
            for aa in a_list:
                if aa not in aug_list:
                    aug_list.append(aa)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, answer, id2, aug_list))
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def transfer_num_g2t(data, group):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d, graph in zip(data, group):
        
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
        answer = d["ans"]
        fractions = re.findall("\d+\(\d+\/\d+\)", answer)
        if len(fractions):
            answer = answer.replace("(", "+(")
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        if '=' in input_seq:
            if '+' in input_seq or '-' in input_seq or '*' in input_seq or '/' in input_seq:
                continue
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, answer, graph['group_num']))
    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained + pairs_tested:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
        else:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            aug_list = [indexes_from_sentence(output_lang, iii, tree) for iii in pair[7]]
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            train_pairs.append([input_cell, len(input_cell), [output_cell], [len(output_cell)],
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], [1], aug_list])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            aug_list = [indexes_from_sentence(output_lang, iii, tree) for iii in pair[7]]
            train_pairs.append([input_cell, len(input_cell), [output_cell], [len(output_cell)],
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], [1], aug_list])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            test_pairs.append([input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack,pair[4], pair[5], pair[6]])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            if max(pair[3]) >= len(input_cell):
                continue
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            test_pairs.append([input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6]])
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs

def prepare_data_newbert(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained + pairs_tested:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
        else:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt", add_special_tokens=False)
        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        seperated_token_list = [tokenizer.convert_ids_to_tokens(int(i)) for i in inputs['input_ids'].squeeze()]
        if '##' in str(seperated_token_list):
            continue
        if 'ò' in pair[0]:
            continue
        if 'ā' in pair[0]:
            continue
        if '什邡市' in pair[0]:
            continue
        if 'è' in pair[0]:
            continue
        if len(num_pos) != len(pair[2]):
            continue
        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            aug_list = [indexes_from_sentence(output_lang, iii, tree) for iii in pair[7]]
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            train_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), [output_cell], [len(output_cell)],
                                pair[2], num_pos, num_stack, pair[4], pair[5], pair[6], [1], aug_list, inputs])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            aug_list = [indexes_from_sentence(output_lang, iii, tree) for iii in pair[7]]
            train_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), [output_cell], [len(output_cell)],
                                pair[2], num_pos, num_stack, pair[4], pair[5], pair[6], [1], aug_list, inputs])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt", add_special_tokens=False)
        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        seperated_token_list = [tokenizer.convert_ids_to_tokens(int(i)) for i in inputs['input_ids'].squeeze()]
        if '##' in str(seperated_token_list):
            continue
        if 'ò' in pair[0]:
            continue
        if 'ā' in pair[0]:
            continue
        if '什邡市' in pair[0]:
            continue
        if 'è' in pair[0]:
            continue
        if len(num_pos) != len(pair[2]):
            continue

        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            test_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                            pair[2], num_pos, num_stack,pair[4], pair[5], pair[6], inputs])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            if max(pair[3]) >= len(input_cell):
                continue
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            test_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, pair[4], pair[5], pair[6], inputs])
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def prepare_data_bert(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained + pairs_tested:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
        else:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:

        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt", add_special_tokens=False)
        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        seperated_token_list = [tokenizer.convert_ids_to_tokens(int(i)) for i in inputs['input_ids'].squeeze()]
        if '##' in str(seperated_token_list):
            continue
        if 'ò' in pair[0]:
            continue
        if 'ā' in pair[0]:
            continue
        if '什邡市' in pair[0]:
            continue
        if 'è' in pair[0]:
            continue
        if len(num_pos) != len(pair[2]):
            continue

        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            train_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), [output_cell], [len(output_cell)],
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], [1], inputs])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            train_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), [output_cell], [len(output_cell)],
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], [1], inputs])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:

        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt", add_special_tokens=False)
        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        seperated_token_list = [tokenizer.convert_ids_to_tokens(int(i)) for i in inputs['input_ids'].squeeze()]
        if '##' in str(seperated_token_list):
            continue
        if 'ò' in pair[0]:
            continue
        if 'ā' in pair[0]:
            continue
        if '什邡市' in pair[0]:
            continue
        if 'è' in pair[0]:
            continue
        if len(num_pos) != len(pair[2]):
            continue

        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            test_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack,pair[4], pair[5], pair[6], inputs])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            if max(pair[3]) >= len(input_cell):
                continue
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            test_pairs.append([input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], inputs])
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs
# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length, token = PAD_token):
    seq += [token for _ in range(max_length - seq_len)]
    return seq

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num

# num net graph
def get_lower_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) <= float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph

def get_greater_num_graph(max_len, sentence_length, num_list, id_num_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    num_list = change_num(num_list)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    if not contain_zh_flag:
        return graph
    for i in range(len(id_num_list)):
        for j in range(len(id_num_list)):
            if float(num_list[i]) > float(num_list[j]):
                graph[id_num_list[i]][id_num_list[j]] = 1
            else:
                graph[id_num_list[j]][id_num_list[i]] = 1
    return graph

# attribute between graph
def get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in quantity_cell_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len:
                if input_batch[i] == input_batch[j]:
                    graph[i][j] = 1
                    graph[j][i] = 1
    return graph

# quantity between graph
def get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    for i in id_num_list:
        for j in id_num_list:
            graph[i][j] = 1
            graph[j][i] = 1
    return graph

# quantity cell graph
def get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list,contain_zh_flag=True):
    diag_ele = np.zeros(max_len)
    for i in range(sentence_length):
        diag_ele[i] = 1
    graph = np.diag(diag_ele)
    #quantity_cell_list = quantity_cell_list.extend(id_num_list)
    if not contain_zh_flag:
        return graph
    for i in id_num_list:
        for j in quantity_cell_list:
            if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
                graph[i][j] = 1
                graph[j][i] = 1
    return graph

def get_single_batch_graph(input_batch, input_length,group,num_value,num_pos):
    batch_graph = []
    max_len = max(input_length)
    for i in range(len(input_length)):
        input_batch_t = input_batch[i]
        sentence_length = input_length[i]
        quantity_cell_list = group[i]
        num_list = num_value[i]
        id_num_list = num_pos[i]
        graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_lower = get_lower_num_graph(max_len, sentence_length, num_list, id_num_list)
        graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_attbet = get_attribute_between_graph(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
        #graph_newc1 = get_quantity_graph1(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
        graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
        batch_graph.append(graph_total)
    batch_graph = np.array(batch_graph)
    return batch_graph

def get_single_example_graph(input_batch, input_length,group,num_value,num_pos):
    batch_graph = []
    max_len = input_length
    sentence_length = input_length
    quantity_cell_list = group
    num_list = num_value
    id_num_list = num_pos
    graph_newc = get_quantity_cell_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_quanbet = get_quantity_between_graph(max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_attbet = get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_greater = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    graph_lower = get_greater_num_graph(max_len, sentence_length, num_list, id_num_list)
    #graph_newc1 = get_quantity_graph1(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)
    graph_total = [graph_newc.tolist(),graph_greater.tolist(),graph_lower.tolist(),graph_quanbet.tolist(),graph_attbet.tolist()]
    batch_graph.append(graph_total)
    batch_graph = np.array(batch_graph)
    return batch_graph

def prepare_train_batch_together(pairs_to_batch, batch_size, only_labeled = True):
    pairs = copy.deepcopy(pairs_to_batch)
    #pairs = [i[:9] for i in pairs]
    if only_labeled:
        pairs_labeled = [i for i in pairs if [] not in i[2]]
        #pairs_unlabeled = [i for i in pairs if not len(i[-1])]
        pairs = pairs_labeled# + pairs_unlabeled
    print('Number of labeled problem: ', len(pairs))

    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    prob_mask_batches = []
    aug_output_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])


    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #print(batch)
        for _, i, _, j, _, _, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(max(j))
        #input_lengths.append(input_length)
        #output_lengths.append(output_length)
        input_len_max = max(input_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        prob_mask_batch = []
        input_length = []
        output_length = []
        aug_output_batch = []
        for i, li, j, lj, num, num_pos, num_stack, _, _, _, prob, aug_eqs in batch:
            input_seq = pad_seq(i, li, input_len_max)
            for idx in range(len(j)):
                input_length.append(li)
                output_length.append(lj[idx])
                num_batch.append(len(num))
                input_batch.append(input_seq)
                output_batch.append(pad_seq(j[idx], lj[idx], output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                num = [str(ii) for ii in num]
                num_value_batch.append(num)
                prob_mask_batch.append(prob[idx])
                #aug_output_batch.append([pad_seq(copy.deepcopy(iii), len(iii), output_len_max) for iii in aug_eqs])
                if aug_eqs != []:
                    aug_output_batch.append(aug_eqs)
                else:
                    aug_output_batch.append([j[idx]])
            #group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        #group_batches.append(group_batch)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        prob_mask_batches.append(prob_mask_batch)
        graph_batches.append(0)
        aug_output_batches.append(aug_output_batch)
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, prob_mask_batches, aug_output_batches

def prepare_train_batch_together_newbert(pairs_to_batch, batch_size, only_labeled = True):
    pairs = copy.deepcopy(pairs_to_batch)
    #pairs = [i[:9] for i in pairs]
    if only_labeled:
        pairs_labeled = [i for i in pairs if [] not in i[2]]
        #pairs_unlabeled = [i for i in pairs if not len(i[-1])]
        pairs = pairs_labeled# + pairs_unlabeled
    print('Number of labeled problem: ', len(pairs))

    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    prob_mask_batches = []
    aug_output_batches = []
    bert_input_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])


    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #print(batch)
        for _, i, _, j, _, _, _, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(max(j))
        #input_lengths.append(input_length)
        #output_lengths.append(output_length)
        input_len_max = max(input_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        prob_mask_batch = []
        input_length = []
        output_length = []
        aug_output_batch = []
        bert_input_batch = []
        for i, li, j, lj, num, num_pos, num_stack, _, _, _, prob, aug_eqs, bert_inputs in batch:
            input_seq = pad_seq(i, li, input_len_max)
            for idx in range(len(j)):
                input_length.append(li)
                output_length.append(lj[idx])
                num_batch.append(len(num))
                input_batch.append(input_seq)
                output_batch.append(pad_seq(j[idx], lj[idx], output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                num = [str(ii) for ii in num]
                num_value_batch.append(num)
                prob_mask_batch.append(prob[idx])
                #aug_output_batch.append([pad_seq(copy.deepcopy(iii), len(iii), output_len_max) for iii in aug_eqs])
                if aug_eqs != []:
                    aug_output_batch.append(aug_eqs)
                else:
                    aug_output_batch.append([j[idx]])
                bert_input_batch.append(bert_inputs)
            #group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        #group_batches.append(group_batch)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        prob_mask_batches.append(prob_mask_batch)
        graph_batches.append(0)
        aug_output_batches.append(aug_output_batch)
        bert_input_batches.append(bert_input_batch)
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, prob_mask_batches, aug_output_batches, bert_input_batches


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size, only_labeled = True):
    pairs = copy.deepcopy(pairs_to_batch)
    if only_labeled:
        pairs_labeled = [i for i in pairs if i[2]]
        #pairs_unlabeled = [i for i in pairs if not len(i[-1])]
        pairs = pairs_labeled# + pairs_unlabeled
    print('Number of labeled problem: ', len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        for i, li, j, lj, num, num_pos, num_stack, _, group in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            if j:
                output_batch.append(pad_seq(j, lj, output_len_max))
            else:
                output_batch.append([])
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num = [str(ii) for ii in num]
            num_value_batch.append(num)
            group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        group_batches.append(group_batch)

        graph_batches.append(get_single_batch_graph(input_batch, input_length, group_batch, num_value_batch, num_pos_batch))
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches

def prepare_train_batch_together_bert(pairs_to_batch, batch_size, only_labeled = True):
    pairs = copy.deepcopy(pairs_to_batch)
    #pairs = [i[:9] for i in pairs]
    if only_labeled:
        pairs_labeled = [i for i in pairs if [] not in i[2]]
        #pairs_unlabeled = [i for i in pairs if not len(i[-1])]
        pairs = pairs_labeled# + pairs_unlabeled
    print('Number of labeled problem: ', len(pairs))

    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    prob_mask_batches = []
    bert_input_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])


    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #print(batch)
        for _, i, _, j, _, _, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(max(j))
        #input_lengths.append(input_length)
        #output_lengths.append(output_length)
        input_len_max = max(input_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        prob_mask_batch = []
        input_length = []
        output_length = []
        bert_input_batch = []
        for i, li, j, lj, num, num_pos, num_stack, _, _, _, prob, bert in batch:
            input_seq = pad_seq(i, li, input_len_max)
            for idx in range(len(j)):
                input_length.append(li)
                output_length.append(lj[idx])
                num_batch.append(len(num))
                input_batch.append(input_seq)
                output_batch.append(pad_seq(j[idx], lj[idx], output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                num = [str(ii) for ii in num]
                num_value_batch.append(num)
                prob_mask_batch.append(prob[idx])
                bert_input_batch.append(bert)

            #group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        #group_batches.append(group_batch)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        prob_mask_batches.append(prob_mask_batch)
        graph_batches.append(0)
        bert_input_batches.append(bert_input_batch)
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, prob_mask_batches, bert_input_batches

def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches


# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex


def transfer_num_boost_57k(data):  # transfer num into "NUM"
    out_voc = ['*', '-', '+', '/', '^', '1', '3.14', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14']
    voc_voc = ['N0', 'N1', 'N2', 'N3']
    start = time.time()
    print("Transfer numbers...")
    pattern = re.compile("temp_.")
    pairs = []
    count_dict = {}
    pre_labels = open('pre_labels_57k.txt','r')
    label_dict = {}
    for lines in pre_labels:
        lines = lines.split(',')
        if lines[1].count('3.14') > 1 or lines[1].count('N') < 2:
            continue
        if lines[1].count('3.14') > 0 and lines[1].count('*') == 0:
            continue
        eqs = seg_(lines[1].strip())
        if eqs.count('1') > 1:
            continue
        cont_flag = False
        for num in voc_voc:
            if eqs.count(num) > 2:
                cont_flag = True
        if cont_flag:
            continue
        label_dict[str(int(lines[0]))] = (eqs)
    # count_dict = {}
    total = 0
    success = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["text"].strip().split(" ")
        answer = d["ans"]
        if len(seg) >= 120:
            continue
        nums = d["num_list"]
        if len(nums) < 2 or len(nums) > 14:
            continue
        if len(list(set(nums))) != len(nums):
            continue
        fractions = re.findall("\d+\(\d+\/\d+\)", answer)
        if len(fractions):
            answer = answer.replace("(", "+(")
        try:
            answer = str(to_nums(answer))
        except:
            continue
        #equations = d["equation"][2:]
        id2 = d["id"]
        i = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                #nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
                i += 1
            else:
                input_seq.append(s)
        nums_fraction = []
        nums = [str(i) for i in nums]
        if '=' in input_seq:
            if '+' in input_seq or '-' in input_seq or '*' in input_seq or '/' in input_seq:
                continue
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        #print(label_dict)
        
        num_pos = []
        op = ['*', '-', '+', '/']
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)

        assert len(nums) == len(num_pos)

        #print(final_candi_list)
        if str(id2) in label_dict:
            pairs.append((input_seq, from_infix_to_prefix(label_dict[str(id2)]), nums, num_pos, answer, id2))
        else:
            pairs.append((input_seq, [], nums, num_pos, answer, id2))
    #print(len(pairs))
    return pairs

def transfer_num_57k(data):  # transfer num into "NUM"
    start = time.time()
    print("Transfer numbers...")
    pattern = re.compile("temp_.")
    pairs = []
    out = open('pre_labels_57k.txt','w')
    # count_dict = {}
    total = 0
    success = 0

    for d in data:
        nums = []
        input_seq = []
        seg = d["text"].strip().split()
        if len(seg) >= 120:
            continue
        nums = d["num_list"]
        if len(nums) < 2:
            continue
        if len(list(set(nums))) != len(nums):
            continue
        answer = d["ans"]
        fractions = re.findall("\d+\(\d+\/\d+\)", answer)
        if len(fractions):
            answer = answer.replace("(", "+(")
        #equations = d["equation"][2:]
        id2 = d["id"]
        print(id2)
        try:
            answer = str(to_nums(answer))
        except:
            continue
        i = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                #nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM"+str(i))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
                i += 1
            else:
                input_seq.append(s)

        num_pos = []
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        nums = [str(i) for i in nums]

        if '=' in input_seq:
            if '+' in input_seq or '-' in input_seq or '*' in input_seq or '/' in input_seq:
                continue
        try:
            result = pre_label(nums, answer)
        except:
            result = False

        if result:
            success += 1
            out.write(str(id2) + ',' + str(result) + '\n')
            out.flush()
        total += 1
        pairs.append((input_seq, nums, num_pos, answer, id2))
    print(success)
    print(total)
    return pairs

def transfer_num_new(data):  # transfer num into "NUM"
    out_voc = ['*', '-', '+', '/', '^', '1', '3.14', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14']
    start = time.time()
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    count_dict = {}
    out = open('pre_labels.txt','a+')
    # count_dict = {}
    total = 0
    success = 0
    for d in data:

        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        answer = d["ans"]
        id2 = d["id"]
        #if int(id2) < 23100:
        #    continue
        #fractions = re.findall("\d+\(\d+\/\d+\)", answer)
        #if len(fractions):
        #    answer = answer.replace("(", "+(")
        print(id2)
        try:
            answer = to_nums(answer)
        except:
            equations = d["equation"][2:]
            answer = to_nums(equations)

        #equations = d["equation"][2:]

        i = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM"+str(i))
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
                i += 1
            else:
                input_seq.append(s)
        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        #out_seq = seg_and_tag(equations)
        num_pos = []
        #op = ['*', '-', '+', '/']
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)
        nums = [str(i) for i in nums]
        assert len(nums) == len(num_pos)
        if '=' in input_seq:
            if '+' in input_seq or '-' in input_seq or '*' in input_seq or '/' in input_seq:
                continue

        result = pre_label(nums, answer)

        if result:
            success += 1
            out.write(str(id2) + ',' + str(result) + '\n')
            out.flush()
        total += 1

        #if final_candi_list:
        #    print(final_candi_list[0])
        #    pairs.append((input_seq, final_candi_list[0], nums, num_pos, answer))
        #else:
        #    pairs.append((input_seq, [], nums, num_pos, answer))
        #string = len(nums)
        #print(string)
        #string2 = str((len(out_seq)))
        #if string not in count_dict.keys():
        #    count_dict[string] = 0
        #else:
        #    count_dict[string] = count_dict[string] + 1
    print(success)
    print(total)
    return pairs

class Equ:
    def __init__(self, string):
        self.string = string
    def str2res(self):
        try:
            return float(eval(self.string))
        except:
            return float(-1)
    def get_string(self):
        return self.string
    def get_list(self):
        op = ['+', '-', '*', '/', '(', ')']
        string_list = list(self.string)
        tmp = ''
        final_list = []
        for i in string_list:
            if i not in op:
                tmp = tmp + str(i)
            else:
                if tmp:
                    final_list.append(tmp)
                    tmp = ''
                final_list.append(i)
        if tmp:
            final_list.append(tmp)
        return final_list

def to_nums(string):
    string = str(string)
    string = string.replace('%','/100')
    return eval(string)

def pre_label(num_list, value):
    op_list = ['+', '-', '*', '/']
    constant = ['1', '3.14']
    token_list = ['N'+str(i) for i in range(len(num_list))]
    token2num = {}
    for token, num in zip(token_list, num_list):
        token2num[str(token)] = num
    token_list += constant
    #num_list = list(set(num_list))
    #reached_values = copy.deepcopy(num_list)
    reached_values = [to_nums(i) for i in num_list]
    reached = [Equ(str(i)) for i in token_list]
    reached_current = copy.deepcopy(reached)
    #reached_empty = []
    #reached_original = copy.deepcopy(reached)
    #print(reached_values)
    trys = 0
    start = time.time()
    while(trys < 8000 or (time.time()-start) > 500):
        #end = time.time()
        for i in reached_current:
            for j in reached_current:
                for op in op_list:
                    if trys >= 50000:
                        print('failed')
                        return False
                    trys += 1
                    reached_eq = '(' + i.string + ')' + op + j.string
                    try:
                        reached_value = eval(replace_token(reached_eq, token2num).replace('%','/100'))
                    except:
                        reached_value = -1
                    if reached_value > 0 and reached_value not in reached_values:
                        if abs(reached_value - value) < 1e-3:
                            return reached_eq
                        reached.append(Equ(reached_eq))
                        reached_values.append(reached_value)

                    reached_eq = i.string + op + '(' +  j.string + ')'
                    try:
                        reached_value = eval(replace_token(reached_eq, token2num).replace('%','/100'))
                    except:
                        reached_value = -1
                    if reached_value > 0 and reached_value not in reached_values:
                        if abs(reached_value - value) < 1e-3:
                            return reached_eq
                        reached.append(Equ(reached_eq))
                        reached_values.append(reached_value)
                    #print(len(reached_current))
                    #print(trys)
        reached_current = copy.deepcopy(reached)
    print('Failed!!')
    return False

def replace_token(string, token2num):
    for token in token2num:
        string = string.replace(str(token), token2num[token])
    return string

def seg(string):
    op_list = ['+', '-', '*', '/', '(', ')']
    out_list = []
    start_pos = 0
    #end_pos = 0
    for idx, token in enumerate(string):
        if token in op_list:
            out_list.append(token)
            start_pos = idx + 1
        else:
            if (idx == len(string) - 1) or (string[idx + 1] in op_list):
                out_list.append(string[start_pos:(idx + 1)])
    return out_list

def prepare_train_batch_together_em(pairs_to_batch, batch_size, epoch, only_labeled = True):
    pairs = copy.deepcopy(pairs_to_batch)
    pairs = [i[:9] for i in pairs]
    if only_labeled:
        pairs_labeled = [i for i in pairs if i[2]]
        #pairs_unlabeled = [i for i in pairs if not len(i[-1])]
        pairs = pairs_labeled# + pairs_unlabeled
    if epoch <= 25:
        pairs_23k = [i for i in pairs if i[8] == '23k']
        pairs = pairs_23k
    print('Number of labeled problem: ', len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    answer_batches = []
    vani_in_batches = []
    vani_out_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        answer_batch = []
        group_batch = []
        num_value_batch = []
        vani_in_batch = []
        vani_out_batch = []
        for i, li, j, lj, num, num_pos, num_stack, answer, _ in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            if j:
                output_batch.append(pad_seq(j, lj, output_len_max))
            else:
                output_batch.append([])
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            num = [str(ii) for ii in num]
            num_value_batch.append(num)
            answer_batch.append(answer)
            vani_in_batch.append(i)
            vani_out_batch.append(j)
            #group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        answer_batches.append(answer_batch)
        vani_in_batches.append(vani_in_batch)
        vani_out_batches.append(vani_out_batch)
        #group_batches.append(group_batch)

        graph_batches.append(0)
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, answer_batches, vani_in_batches, vani_out_batches


def masked_cross_entropy_with_prob(logits, target, length, prob):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    losses = losses * prob
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def prepare_data_g2t(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained + pairs_tested:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
        else:
            input_lang.add_sen_to_vocab(pair[0])
            if pair[1]:
                output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            train_pairs.append([input_cell, len(input_cell), [output_cell], [len(output_cell)],
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], [1], pair[7]])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            train_pairs.append([input_cell, len(input_cell), [output_cell], [len(output_cell)],
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], [1], pair[7]])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        if pair[1]:
            num_stack = []
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])

            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if max(pair[3]) >= len(input_cell):
                continue
            test_pairs.append([input_cell, len(input_cell), output_cell, len(output_cell),
                            pair[2], pair[3], num_stack,pair[4], pair[5], pair[6], pair[7]])
        else:
            num_stack = []
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            if max(pair[3]) >= len(input_cell):
                continue
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            test_pairs.append([input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pair[3], num_stack, pair[4], pair[5], pair[6], pair[7]])
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs

def prepare_train_batch_together_g2t(pairs_to_batch, batch_size, only_labeled = True):
    pairs = copy.deepcopy(pairs_to_batch)
    #pairs = [i[:9] for i in pairs]
    if only_labeled:
        pairs_labeled = [i for i in pairs if [] not in i[2]]
        #pairs_unlabeled = [i for i in pairs if not len(i[-1])]
        pairs = pairs_labeled# + pairs_unlabeled
    print('Number of labeled problem: ', len(pairs))

    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    group_batches = []
    graph_batches = []
    num_value_batches = []
    prob_mask_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])


    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #print(batch)
        for _, i, _, j, _, _, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(max(j))
        #input_lengths.append(input_length)
        #output_lengths.append(output_length)
        input_len_max = max(input_length)
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        group_batch = []
        num_value_batch = []
        prob_mask_batch = []
        input_length = []
        output_length = []
        for i, li, j, lj, num, num_pos, num_stack, _, _, _, prob, group in batch:
            input_seq = pad_seq(i, li, input_len_max)
            for idx in range(len(j)):
                input_length.append(li)
                output_length.append(lj[idx])
                num_batch.append(len(num))
                input_batch.append(input_seq)
                output_batch.append(pad_seq(j[idx], lj[idx], output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                num = [str(ii) for ii in num]
                num_value_batch.append(num)
                prob_mask_batch.append(prob[idx])
                group_batch.append(group)
            
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        num_value_batches.append(num_value_batch)
        #group_batches.append(group_batch)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        prob_mask_batches.append(prob_mask_batch)
        group_batches.append(group_batch)
        graph_batches.append(get_single_batch_graph(input_batch, input_length,group_batch,num_value_batch,num_pos_batch))
        
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, prob_mask_batches


def process(seq):
    
    seq=['(' if i =='[' else i for i in seq]
    seq=[')' if i ==']' else i for i in seq]

    op = ['+', '-', '*', '/', '^']
    temp = []
    final_list = []
    b_count = 0
    for ii in seq:
        if ii == '(':
            temp.append(ii)
            b_count += 1
            continue
        if ii == ')':
            temp.append(ii)
            b_count -= 1
            if b_count == 0:
                final_list.append(temp)
                temp = []
            continue
        if b_count != 0:
            temp.append(ii)
        else:
            #final_list.append(temp)
            final_list.append(ii)
    return final_list



import itertools
def aug(all_list):
    final_list = []
    list_add = []
    list_sub = []
    list_multi = []
    list_div = []
    for idx, one_op in enumerate(all_list):
        if one_op == '+':
            list_add.append(idx)
        elif one_op == '-':
            list_sub.append(idx)
        elif one_op == '*':
            list_multi.append(idx)
        elif one_op == '/':
            list_div.append(idx)
    final_list.append(unfold(all_list))
    add_block = []
    mul_block = []
    if list_add or len(list_sub) > 1:
        add_sub_list = list_add + list_sub
        add_sub_list.sort()
        for iii, idxidx in enumerate(add_sub_list):
            if iii == 0:
                add_block.append([0,idxidx])
            if idxidx == add_sub_list[-1]:
                add_block.append([idxidx + 1])
            else:
                add_block.append([idxidx + 1, add_sub_list[iii + 1]])
    final_mul_list = [all_list]
    #print(list_multi)
    if list_multi or len(list_div) > 1:
        mul_div_list = list_multi + list_div
        mul_div_list.sort()
        temp = []
        for jjj, idxidx2 in enumerate(mul_div_list):
            if temp == []:
                temp.append(idxidx2 - 1)
                temp.append(idxidx2 + 1)
            else:
                if idxidx2 == temp[-1] + 1:
                    temp.append(idxidx2 + 1)
                else:
                    mul_block.append(temp)
                    temp = []
        if temp != []:
            mul_block.append(temp)
            temp = []
        #print(mul_block)
        

        for idx_m, mul in enumerate(mul_block):
            p_mul_sub_block = list(itertools.permutations(mul))
            #print(p_mul_sub_block)
            for sub_mul in p_mul_sub_block:
                mul_aug = copy.deepcopy(all_list)
                for mul_idx, p_sub_block in enumerate(sub_mul):
                    #print(sub_mul)
                    #print(all_list[p_sub_block - 1])
                    if mul_idx == 0 and all_list[p_sub_block - 1] == '/':
                        #print('break')
                        break
                    mul_aug[mul[mul_idx]] = all_list[p_sub_block]
                    if mul_idx != 0:
                        if p_sub_block != 0:
                            if all_list[p_sub_block - 1] in ['*', '/']:
                                mul_aug[mul[mul_idx] - 1] = all_list[p_sub_block - 1]
                            else:
                                mul_aug[mul[mul_idx] - 1] = '*' 
                        else:
                            mul_aug[mul[mul_idx] - 1] = '*' 
                final_mul_temp = mul_aug
                if final_mul_temp not in final_mul_list and final_mul_temp != []:
                    final_mul_list.append(final_mul_temp)
    #print(final_mul_list)

    #print(final_mul_list)
    #print(add_block)
    for one_list in final_mul_list:
        if add_block:
            p_add_block = list(itertools.permutations(add_block))
            #print(p_add_block)
            for k in p_add_block:
                unfold_list = []
                for k_idx, sub_block in enumerate(k):
                    if k_idx == 0 and one_list[sub_block[0] - 1] == '-':
                        break
                    if k_idx == 0:
                        if len(sub_block) == 1:
                            unfold_list.append(one_list[sub_block[0]:])
                        else:
                            unfold_list.append(one_list[sub_block[0]:sub_block[1]])
                    else:
                        if sub_block[0] == 0:
                            unfold_list.append('+')
                        else:
                            unfold_list.append(one_list[sub_block[0] - 1])
                        if len(sub_block) == 1:
                            unfold_list.append(one_list[sub_block[0]:])
                        else:
                            unfold_list.append(one_list[sub_block[0]:sub_block[1]])
                #print(unfold_list)
                final_temp = unfold(unfold(unfold_list))
                if final_temp not in final_list and final_temp != []:
                    final_list.append(final_temp)
                    if len(final_list) > 50:
                        return final_list
        else:
            final_temp = unfold(unfold(one_list))
            if final_temp not in final_list and final_temp != []:
                    final_list.append(final_temp)
                    if len(final_list) > 50:
                        return final_list
    #print(final_list)
    return final_list

            
               

def unfold(seq):
    output = []
    for i in seq:
        if isinstance(i, list):
            for ii in i:
                output.append(ii)
        else:
            output.append(i)
    return output