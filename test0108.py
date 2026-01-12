import os
import json
import torch
import torch.nn.functional as F  # 用于计算 softmax
import random
import numpy as np
import logging
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from sklearn.metrics import roc_auc_score  # 导入 AUC 计算库

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def set_seed(seed=52):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_answers(filename):
    answers = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            js = json.loads(line)
            for i in range(len(js)):
                example = InputExample(label=js[i]['target'], guid=js[i]['id'], text_a=js[i]['code1'], text_b=js[i]['func'])
                answers.append(example)
    return answers

def test(model, test_data_loader, test_dataset, output_file_true, output_file_pred):
    sum_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    all_labels = []        # 存放真实标签
    all_probabilities = [] # 存放模型预测为正类的概率 (用于计算 AUC)

    fp_records = []
    fn_records = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    current_idx = 0
    with open(output_file_true, 'w') as true_f, open(output_file_pred, 'w') as pred_f:
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc="Testing"):
                batch = batch.to(device)
                logits = model(batch)
                
                # ✅ 获取概率值 (Softmax)
                probs = F.softmax(logits, dim=-1)
                # 提取正类 (positive, 索引为1) 的概率
                pos_probs = probs[:, 1].cpu().numpy()
                
                preds = torch.argmax(logits, dim=-1)
                trues_cpu = batch['label'].cpu().numpy()
                preds_cpu = preds.cpu().numpy()

                all_labels.extend(trues_cpu)
                all_probabilities.extend(pos_probs)

                for t, p in zip(trues_cpu, preds_cpu):
                    true_f.write(f"{t}\n")
                    pred_f.write(f"{p}\n")

                    example = test_dataset[current_idx]
                    if t == p:
                        sum_correct += 1
                        if t == 1: true_positive += 1
                    else:
                        error_data = {
                            "id": example.guid,
                            "true_label": int(t),
                            "pred_label": int(p),
                            "code1": example.text_a,
                            "func": example.text_b
                        }
                        if t == 0 and p == 1:
                            fp_records.append(error_data)
                            false_positive += 1
                        elif t == 1 and p == 0:
                            fn_records.append(error_data)
                            false_negative += 1
                    current_idx += 1

    # 保存分析文件
    with open("fp_analysis.json", "w", encoding="utf-8") as f:
        json.dump(fp_records, f, indent=4, ensure_ascii=False)
    with open("fn_analysis.json", "w", encoding="utf-8") as f:
        json.dump(fn_records, f, indent=4, ensure_ascii=False)

    # --- 计算指标 ---
    accuracy = sum_correct / len(test_dataset)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # ✅ 计算 AUC
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = 0.0  # 如果测试集只有一类标签，AUC会报错

    print("\n" + "="*45)
    print(f"Final Evaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")  # ✅ 输出 AUC
    print("-" * 45)
    print(f"Saved: fp_analysis.json ({len(fp_records)} samples)")
    print(f"Saved: fn_analysis.json ({len(fn_records)} samples)")
    print("="*45)

if __name__ == "__main__":
    set_seed(52)
    
    # 加载测试集
    test_dataset = read_answers('../../../dataset/0107final/test.json')
    
    # 模型架构初始化
    classes = ['negative', 'positive']
    local_path = "/root/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", local_path)

    promptTemplate = MixedTemplate(
        model=plm,
        text='The code changed from {"placeholder":"text_a"} to {"placeholder":"text_b"}. {"soft"} The commit is {"mask"}.',
        tokenizer=tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={"negative": ["indefective", "robust"], "positive": ["defective", "faulty"]},
        tokenizer=tokenizer,
    )
    promptModel = PromptForClassification(template=promptTemplate, plm=plm, verbalizer=promptVerbalizer)

    # DataLoader
    test_data_loader = PromptDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=32,
        shuffle=False
    )

    # 加载最佳模型权重
    best_model_path = "best_model_epoch_6.pt" 
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        promptModel.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))
        test(promptModel, test_data_loader, test_dataset, 'true_labels.txt', 'pred_labels.txt')
    else:
        print(f"Error: {best_model_path} not found!")