# coding: utf-8
from src.train_and_evaluate_prob import *
from src.models_prob import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import torch
from itertools import chain

def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        file = json.load(f)
    return file

torch.cuda.set_device(1)
batch_size = 10
embedding_size = 128
hidden_size = 768
n_epochs = 200
learning_rate = 2e-5
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'


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

id_file = open('57k_id.txt', 'r')
id_list = []
for line in id_file:
    id_list.append(str(int(line.strip())))
data_23k = load_raw_data("data/Math_23K.json")

#group_data = read_json("data/Math_23K_processed.json")


pairs_23k, _, _ = transfer_num(data_23k)
print('number of 23k:', len(pairs_23k))


generate_nums = ['1', '3.14']
copy_nums = 15
#exit()
temp_pairs = []
for p in pairs_23k:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], p[4],'23k',p[5], p[6]))
pairs_23k = temp_pairs


fold_size = int(len(pairs_23k) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs_23k[fold_start:fold_end])
fold_pairs.append(pairs_23k[(fold_size * 4):])

fold = 1 #we can also iterate all the folds like GTS
pairs_tested_23k = []
pairs_trained_23k = []
for fold_t in range(5):
    if fold_t == fold:
        pairs_tested_23k += fold_pairs[fold_t]
    else:
        pairs_trained_23k += fold_pairs[fold_t]




input_lang, output_lang, train_pairs, test_pairs = prepare_data_newbert(pairs_trained_23k, pairs_tested_23k, 5, generate_nums,
                                                                copy_nums, tree=True)
# Initialize models

seq_encoder = Equation_encoder(input_size = output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size, n_layers=1, dropout=0)
encoder = Encoder_BERT(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
mlp = MLP(hidden_size=hidden_size)

#total_num = sum(p.numel() for p in predict.parameters()) + sum(p.numel() for p in generate.parameters()) + sum(p.numel() for p in merge.parameters())
#print(total_num)
#exit()
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
seq_optimizer = torch.optim.Adam(chain(seq_encoder.parameters(), mlp.parameters()), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=50, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=50, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=50, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=50, gamma=0.5)
seq_scheduler = torch.optim.lr_scheduler.StepLR(seq_optimizer, step_size=50, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    seq_encoder.cuda()
    mlp.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
    num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, prob_mask_batches, aug_batches, bert_batches = prepare_train_batch_together_newbert(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    #print(output_batches)
    start = time.time()
    for idx in range(len(input_lengths)):
    #for idx in range(5):
        loss = train_tree_bertnew(
                    input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                    num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge, seq_encoder, mlp, seq_optimizer,
                    encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], 0, prob_mask_batches[idx], aug_batches[idx], bert_batches[idx], epoch)
        loss_total += loss
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    seq_scheduler.step()
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    #if epoch % 10 == 0 or epoch > n_epochs - 10:
    if epoch % 5 == 0 and epoch > 0:
    #if epoch % 5 == 0:
            for idx, train_batch in enumerate(train_pairs):
                loss_dis = 0
                loss_count = 0
                #if train_batch[8] == '57k':
                if True:
                    seq_optimizer.zero_grad()
                    #batch_graph = get_single_example_graph(train_batch[0], train_batch[1], train_batch[8], [str(ii) for ii in train_batch[4]], train_batch[5])
                    test_res, mwp_feature = evaluate_tree_newbert(train_batch[0], train_batch[1], generate_num_ids, encoder, predict, generate,
                                                    merge, output_lang, train_batch[5], 0, train_batch[12], beam_size=5)
                    pos_logit_list = [0.5]
                    for pos_eq in train_batch[11]:
                        pos_eq_tensor = torch.LongTensor([pos_eq]).transpose(0, 1).cuda()
                        logits_pos = mlp(mwp_feature.squeeze(), seq_encoder(pos_eq_tensor).squeeze())
                        #logits_pos = torch.sigmoid(torch.matmul(mwp_feature.squeeze().unsqueeze(0), seq_encoder(pos_eq_tensor).squeeze().unsqueeze(1))).squeeze()
                        pos_logit_list.append(float(logits_pos))
                        if float(logits_pos) < 0.5 and train_batch[8] == '23k':
                            loss_count += 1
                            loss_dis += f.binary_cross_entropy(logits_pos, torch.ones(logits_pos.shape).cuda().float())
                    for i in test_res:
                        eq_tensor = torch.LongTensor([i]).transpose(0, 1).cuda()
                        #logits = torch.sigmoid(torch.matmul(mwp_feature.squeeze().unsqueeze(0), seq_encoder(eq_tensor).squeeze().unsqueeze(1))).squeeze()
                        logits = mlp(mwp_feature.squeeze(), seq_encoder(eq_tensor).squeeze())
                        val_ac = compute_result_weak(i, train_batch[7], output_lang, train_batch[4], train_batch[6])
                        if not val_ac:
                            if float(logits) > 0.5:
                                loss_count += 1
                                loss_dis += f.binary_cross_entropy(logits, torch.zeros(logits.shape).cuda().float())
                        if val_ac and i not in train_pairs[idx][2]:
                        #if float(logits) > 0.5 and val_ac and i not in train_pairs[idx][2]:
                            if [] in train_pairs[idx][2]:
                                train_pairs[idx][2] = [i]
                                train_pairs[idx][3] = [len(i)]
                                train_pairs[idx][10] = [1]
                            else:                                
                                train_pairs[idx][2].append(i)
                                train_pairs[idx][3].append(len(i))
                    try:
                        if loss_count > 0:
                            batch_count += 1
                            loss_dis = loss_dis / loss_count
                            loss_dis.backward()
                            seq_optimizer.step()
                            if idx % 5000 == 0:
                                print('loss_dis_2', loss_dis.item())
                                print('logit', logits)
                    except:
                        pass
                    if len(train_pairs[idx][2]) > 1:
                    #if True:
                        p_list = []
                        logits_list = []
                        for sub_target, sub_len in zip(train_pairs[idx][2], train_pairs[idx][3]):
                            p, logits = train_tree_prob_newbert(
                            [train_pairs[idx][0]], [train_pairs[idx][1]], [sub_target], [sub_len],
                            [copy.deepcopy(train_pairs[idx][6])], [len(train_pairs[idx][5])], generate_num_ids, encoder, predict, generate, merge, seq_encoder,
                            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, [train_pairs[idx][5]], 0, train_pairs[idx][12])
                            p_list.append(p) 
                            logits_list.append(float(logits))
                        #print(logits_list)  
                        p_list = [pp / sum(p_list) for pp in p_list]

                        #logits_list = [0 if ll < 0.5 else 1 for ll in logits_list]
                        #final_list = [p * l for p,l in zip(p_list, logits_list)]
                        if epoch > 100:
                            final_list = [(p + l)/2 for p,l in zip(p_list, logits_list)]
                        else:  
                            final_list = p_list
                        train_pairs[idx][10] = final_list
                    if False:
                        print(train_pairs[idx][8])
                        print('id:', train_pairs[idx][9])
                        print('solutions:', train_pairs[idx][2])
                        print('weights:', train_pairs[idx][10])
                        for solution in train_pairs[idx][2]:
                            s = [output_lang.index2word[iii] for iii in solution]
                            print(s)
                        print('end')
    print('replace finish')
    if epoch % 5 == 0 or epoch > n_epochs - 10:
        value_ac_23k = 0
        value_ac_57k = 0
        eval_total_23k = 0
        eval_total_57k = 0
        start = time.time()
        for test_batch in test_pairs:
            #print(test_batch)
            #batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[8], [str(ii) for ii in test_batch[4]], test_batch[5])
            test_res, _ = evaluate_tree_newbert(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], 0, test_batch[10], beam_size=beam_size)
            val_ac = compute_result_weak(test_res[0], test_batch[7], output_lang, test_batch[4], test_batch[6])
            if test_batch[8] == '23k':
                if val_ac:
                    value_ac_23k += 1
                eval_total_23k += 1
            if test_batch[8] == '57k':
                if val_ac:
                    value_ac_57k += 1
                eval_total_57k += 1           
        #print(value_ac, eval_total)
        print("test_answer_acc_23k", float(value_ac_23k) / eval_total_23k, eval_total_23k)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        #torch.save(encoder.state_dict(), "models/encoder_weak_" + str(epoch))
        #torch.save(predict.state_dict(), "models/predict_weak_" + str(epoch))
        #torch.save(generate.state_dict(), "models/generate_weak_" + str(epoch))
        #torch.save(merge.state_dict(), "models/merge_weak_" + str(epoch))
