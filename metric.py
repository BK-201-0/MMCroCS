
def cal_mrr_k(answer_list, ground_truth_urls, k):
    mrr_scores = []
    for urls, gt_url in zip(answer_list, ground_truth_urls):
        mrr = 0
        if gt_url in urls:
            rank = urls.index(gt_url) + 1
            if rank <= k:
                mrr += 1 / rank
        mrr_scores.append(mrr)
    
    mean_mrr = sum(mrr_scores) / len(mrr_scores)
    return mean_mrr
    
def cal_mrr(answers, ground_truth_urls):
    num_list = [1,5,10,100,500,1000]
    mrr_k = []
    for k in num_list:
        mrr_k.append(cal_mrr_k(answers, ground_truth_urls, k))
    answers = {f"mrr-{k}":mrr for k, mrr in zip(num_list, mrr_k)}
    return answers

    
def cal_recall_k(answer_list, ground_truth_urls, k) :
    recall = 0
    for urls, gt_url in zip(answer_list, ground_truth_urls):
        if gt_url in urls:
            rank = urls.index(gt_url) + 1
            if rank <= k:
                recall += 1
    return recall / len(answer_list)

def cal_recall(answers, ground_truth_urls):
    """
    :param: answers: [[url1,url2,...],]
    :param: ground_truths: [url1,url1..]
    """
    # return top-1/5/10/100/500/1000
    num_list = [1,5,10,100,500,1000]
    recall_k = []
    for k in num_list:
        recall_k.append(cal_recall_k(answers, ground_truth_urls, k))
    answers = {f"recall-{k}":recall for k, recall in zip(num_list, recall_k)}
    return  answers
