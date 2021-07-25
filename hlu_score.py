import argparse
import numpy as np
import csv
import re

parser = argparse.ArgumentParser(description='Calculate recall')
parser.add_argument('--result_file', type=str, help='file contains predicted result', required=True)
parser.add_argument('--top_k', type=int, help='top k highest rank items', required=True)
parser.add_argument('--score_file', type=str, help='file contains score result', required=True)
# parser.add_argument('--log_result_folder', type=str, help='folder result', required=True)
parser.add_argument('--model_name', type=str, help='file contains predicted result', required=True)
args = parser.parse_args()

result_file = args.result_file
top_k = args.top_k
score_file = args.score_file
model_name = args.model_name

list_HLU_score = []
C = 100
beta = 5
with open(result_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        elements = line.split('|')
        ground_truth = elements[0]
        list_gt_item = re.split('\\s+',ground_truth.split(':')[1])
        # predicted_items = elements[1:top_k + 1]
        predicted_items = elements[1:]
        list_top_k_item = [item.strip().split(':')[0] for item in predicted_items]

        #tinh tu so HLU
        sum_of_rank_score = 0
        for item in list_gt_item:
            if item not in list_top_k_item:
                rank_score = 0
            else:
                item_rank = list_top_k_item.index(item) + 1
                rank_score = 2 ** ((1 - item_rank) / (beta - 1))
            sum_of_rank_score += rank_score

        # tinh mau so cua HLU
        sum_of_rank_target_basket = 0
        target_basket_size = len(list_gt_item)
        for r in range(1, target_basket_size + 1):
            target_rank_score = 2 ** ((1 - r) / (beta - 1))
            sum_of_rank_target_basket += target_rank_score

        HLU_score = C * sum_of_rank_score / sum_of_rank_target_basket
        list_HLU_score.append(HLU_score)
        # list_seq_topk_predicted.append(list_top_k_item.copy())
        # list_top_k_item.clear()

avg_hlu_score = np.array(list_HLU_score).mean()
print("Recall@%d : %.6f" % (top_k, avg_hlu_score))
with open(score_file,'a',newline='') as f:
    writer=csv.writer(f)
    writer.writerow([model_name, top_k, avg_hlu_score])