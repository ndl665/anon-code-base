# Fine-tuning版本代码
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import torch.nn as nn
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# 自定义数据集类
class DefectDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 拼接clean code和buggy code
        text = f"Clean code: {item['text_a']} Buggy code: {item['text_b']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def read_answers(filename):
    answers = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            for i in range(len(js)):
                example = {
                    'text_a': js[i]['code1'],
                    'text_b': js[i]['func'],
                    'label': js[i]['target']
                }
                answers.append(example)
    return answers


def set_seed(seed=52):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 读取数据
train_data = read_answers('../../dataset/fewshot/train16_1.json')
valid_data = read_answers('../../dataset/val12.json')
test_data = read_answers('../../dataset/test12.json')

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(valid_data)}")
print(f"Test size: {len(test_data)}")

# 检查标签分布
def check_label_distribution(dataset, name):
    labels = [example['label'] for example in dataset]
    print(f"{name} set label distribution:")
    print(f"Positive (1): {sum(labels)} samples")
    print(f"Negative (0): {len(labels) - sum(labels)} samples")

check_label_distribution(train_data, "Train")
check_label_distribution(valid_data, "Valid")
check_label_distribution(test_data, "Test")

# 加载本地模型和tokenizer
local_path = "/root/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"
tokenizer = RobertaTokenizer.from_pretrained(local_path)

# 创建分类模型（2分类）
model = RobertaForSequenceClassification.from_pretrained(
    local_path,
    num_labels=2
)

# 创建数据集和数据加载器
train_dataset = DefectDataset(train_data, tokenizer, max_length=512)
valid_dataset = DefectDataset(valid_data, tokenizer, max_length=512)
test_dataset = DefectDataset(test_data, tokenizer, max_length=512)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 检查样本
print("\nSample data processing:")
sample = train_data[0]
print("Clean code:", sample['text_a'][:100])
print("Buggy code:", sample['text_b'][:100])
print("Label:", sample['label'])


def test(model, test_data_loader, output_file_true, output_file_pred):
    sum_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(output_file_true, 'w') as true_file, open(output_file_pred, 'w') as pred_file:
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                sum_correct += torch.eq(labels, preds).sum().item()

                for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    true_file.write(f"{true_label}\n")
                    pred_file.write(f"{pred_label}\n")

                true_positive += ((labels == 1) & (preds == 1)).sum().item()
                false_positive += ((labels == 0) & (preds == 1)).sum().item()
                false_negative += ((labels == 1) & (preds == 0)).sum().item()

    accuracy = sum_correct / len(test_dataset)
    print(f"Accuracy: {accuracy:.4f}")

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    if precision + recall > 0:
        f_measure = (2 * precision * recall) / (precision + recall)
    else:
        f_measure = 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f_measure:.4f}")

    return accuracy, precision, recall, f_measure


def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds)
    print(f"验证的F1分数是：{f1:.2f}")
    return f1


def train(model, train_data_loader, valid_data_loader, test_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    set_seed()
    
    # 训练参数
    max_epochs = 5
    lr = 1e-5
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    gradient_accumulation_steps = 1
    output_dir = './saved_models'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 优化器设置
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    
    max_steps = max_epochs * len(train_data_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=max_steps * 0.1,
        num_training_steps=max_steps
    )
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", max_epochs)
    logger.info("  Total optimization steps = %d", max_steps)
    
    global_step = 0
    best_val_f1 = 0
    patience = 2
    no_improve_epochs = 0
    best_model_path = ""
    
    for epoch in range(max_epochs):
        total_loss = 0.0
        model.train()
        
        logger.info(f"******* Epoch {epoch} *****")
        
        for batch_idx, batch in enumerate(tqdm(train_data_loader, desc=f"Training Epoch {epoch}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 50 == 0:
                    print(f"Step {global_step}, Loss: {loss.item():.4f}")
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data_loader)
        logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # 验证
        val_f1 = evaluate(model, valid_data_loader)
        print(f"Epoch {epoch} - Validation F1: {val_f1:.4f}")
        
        # 早停判断
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            best_model_path = os.path.join(output_dir, f"finetune_best_model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 加载最佳模型进行测试
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation.")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("\nBest model not found. Using the model from the last epoch.")
    
    # 测试
    test(model, test_data_loader, 'true_labels.txt', 'pred_labels.txt')


# 开始训练
train(model, train_data_loader, valid_data_loader, test_data_loader)