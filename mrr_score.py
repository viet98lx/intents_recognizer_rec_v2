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

mrr_score = 0
list_rank = []
list_mrr_score = []
with open(result_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        elements = line.split('|')
        ground_truth = elements[0]
        list_gt_item = re.split('\\s+',ground_truth.split(':')[1])
        # predicted_items = elements[1:top_k + 1]
        predicted_items = elements[1:]
        list_top_k_item = [item.strip().split(':')[0] for item in predicted_items]
        for item in list_gt_item:
            if item not in list_top_k_item:
                list_rank.append(0)
            else:
                item_rank = list_top_k_item.index(item)+1
                reciprocal_rank = 1/(item_rank)
                list_rank.append(reciprocal_rank)
        list_mrr_score.append(np.array(list_rank).mean())
        list_gt_item.clear()
        list_top_k_item.clear()
        list_rank.clear()
        # list_seq_topk_predicted.append(list_top_k_item.copy())
        # list_top_k_item.clear()

mrr_score = np.array(list_mrr_score).mean()
print("MRR@%d : %.6f" % (top_k, mrr_score))
with open(score_file,'a',newline='') as f:
    writer=csv.writer(f)
    # writer.writerow([model_name, top_k, mrr_score])
    writer.writerow([model_name, top_k, mrr_score])