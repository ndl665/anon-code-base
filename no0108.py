import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import logging
from tqdm import tqdm
from openprompt.data_utils import InputExample
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from sklearn.metrics import f1_score, roc_auc_score

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 获取当前脚本的文件名（不含后缀）用于命名输出文件
script_name = os.path.splitext(os.path.basename(__file__))[0]

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

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

def set_seed(seed=52):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 1. 加载数据集
train_dataset = read_answers('../../../dataset/0107final/train.json')
valid_dataset = read_answers('../../../dataset/0107final/valid.json')
test_dataset = read_answers('../../../dataset/0107final/test.json')

# 2. 加载模型
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

# 3. DataLoaders
train_data_loader = PromptDataLoader(dataset=train_dataset, tokenizer=tokenizer, template=promptTemplate, tokenizer_wrapper_class=WrapperClass, batch_size=16, shuffle=True, drop_last=True)
valid_data_loader = PromptDataLoader(dataset=valid_dataset, tokenizer=tokenizer, template=promptTemplate, tokenizer_wrapper_class=WrapperClass, batch_size=16, shuffle=False)
test_data_loader = PromptDataLoader(dataset=test_dataset, tokenizer=tokenizer, template=promptTemplate, tokenizer_wrapper_class=WrapperClass, batch_size=32, shuffle=False)

# 4. 验证函数
def evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    return f1_score(all_labels, all_preds)

# 5. 测试函数（含 AUC 和 动态文件名）
def test(model, data_loader, raw_dataset, output_file_true, output_file_pred):
    sum_correct = 0
    true_positive, false_positive, false_negative = 0, 0, 0
    fp_records, fn_records = [], []
    all_labels, all_probs = [], [] # 用于 AUC 计算

    model.eval()
    device = next(model.parameters()).device
    current_idx = 0

    with open(output_file_true, 'w') as true_file, open(output_file_pred, 'w') as pred_file:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Testing"):
                batch = batch.to(device)
                logits = model(batch)
                
                # 计算概率值用于 AUC
                probs = F.softmax(logits, dim=-1)
                pos_probs = probs[:, 1].cpu().numpy()
                
                preds = torch.argmax(logits, dim=-1)
                trues_cpu = batch['label'].cpu().numpy()
                preds_cpu = preds.cpu().numpy()

                all_labels.extend(trues_cpu)
                all_probs.extend(pos_probs)
                sum_correct += (preds.cpu() == batch['label'].cpu()).sum().item()

                for t, p in zip(trues_cpu, preds_cpu):
                    true_file.write(f"{t}\n")
                    pred_file.write(f"{p}\n")
                    example = raw_dataset[current_idx]
                    
                    data_info = {"id": example.guid, "target": int(t), "prediction": int(p), "code1": example.text_a, "func": example.text_b}
                    if t == 0 and p == 1:
                        fp_records.append(data_info)
                        false_positive += 1
                    elif t == 1 and p == 0:
                        fn_records.append(data_info)
                        false_negative += 1
                    elif t == 1 and p == 1:
                        true_positive += 1
                    current_idx += 1

    # 使用脚本名作为前缀保存 JSON
    with open(f"{script_name}_fp_analysis.json", "w", encoding="utf-8") as f:
        json.dump(fp_records, f, indent=4, ensure_ascii=False)
    with open(f"{script_name}_fn_analysis.json", "w", encoding="utf-8") as f:
        json.dump(fn_records, f, indent=4, ensure_ascii=False)

    # 计算指标
    accuracy = sum_correct / len(raw_dataset)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*45)
    print(f"Test Results for {script_name}:")
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
    print(f"Recall:   {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Files saved: {script_name}_fp_analysis.json, {script_name}_fn_analysis.json")
    print("="*45)

# 6. 训练主函数
def train(model, train_loader, valid_loader, test_loader, test_set):
    model.plm.config.hidden_dropout_prob = 0.2
    model.plm.config.attention_probs_dropout_prob = 0.2
    model = model.cuda()
    set_seed()

    max_epochs = 12
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader)*max_epochs)
    
    best_val_f1 = 0
    patience = 4
    no_improve_epochs = 0
    best_model_path = f"best_{script_name}.pt"

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to("cuda")
            logits = model(batch)
            loss = nn.CrossEntropyLoss()(logits, batch['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        val_f1 = evaluate(model, valid_loader)
        print(f"Epoch {epoch} - Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping.")
                break

    # 加载最佳模型进行最终测试
    print(f"\nTraining complete. Loading best model: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    test(model, test_loader, test_set, 'true_labels.txt', 'pred_labels.txt')

if __name__ == "__main__":
    train(promptModel, train_data_loader, valid_data_loader, test_data_loader, test_dataset)