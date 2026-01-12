# 删掉测试集重复测试
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import torch.nn as nn
import random

import numpy as np
from openprompt.data_utils import InputExample
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)



def read_answers(filename):
    answers = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            # print(js)
            # code = js['func']
            # target = js['target']
            for i in range(len(js)):
                # print(js[i])
                example = InputExample(guid=js[i]['target'], text_a=js[i]['diff_added'], text_b=js[i]['diff_deleted'])
                answers.append(example)
    return answers


def set_seed(seed=52):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_dataset = read_answers('../../../dataset/1204_add_del/train_set.json')
valid_dataset = read_answers('../../../dataset/1204_add_del/val_set.json')
test_dataset = read_answers('../../../dataset/1204_add_del/test_set.json')

# ✅ 添加在加载数据集后！！
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")
# ✅ 检查标签分布！！
def check_label_distribution(dataset, name):
    labels = [example.guid for example in dataset]
    print(f"{name} set label distribution:")
    print(f"Positive (1): {sum(labels)} samples")
    print(f"Negative (0): {len(labels) - sum(labels)} samples")

check_label_distribution(train_dataset, "Train")
check_label_distribution(test_dataset, "Test")

# print(len(dataset), dataset[:5])
classes = ['negative', 'positive']
from openprompt.plms import load_plm
local_path = "/root/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", local_path)
# plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "microsoft/codebert-base")
from openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate
print('!!!!!Added lines: [ADD] {"placeholder":"text_a"}.Deleted lines: [DEL]  {"placeholder":"text_b"}. The commit is {"mask"}.')
promptTemplate = MixedTemplate(
    model=plm,
    text='Added lines: [ADD] {"placeholder":"text_a"}.Deleted lines: [DEL]  {"placeholder":"text_b"}. The commit is {"mask"}.',
    tokenizer=tokenizer,
)
from openprompt.prompts import ManualVerbalizer

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "negative": ["indefective", "robust"],
        "positive": ["defective", "faulty"],

    },
    tokenizer=tokenizer,
)
# ✅ 添加在定义verbalizer后：用于调试和验证 verbalizer 的 label words 和 token ID 映射
print("Verbalizer mapping:")
for idx, cls in enumerate(classes):
    words = promptVerbalizer.label_words[idx]  # 使用索引，而不是字符串键
    word_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in words]
    print(f"{cls}: {words} → Token IDs: {word_ids}")
from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

from openprompt import PromptDataLoader

train_data_loader = PromptDataLoader(
    dataset=train_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=16,
    shuffle=True,
    drop_last=True
)
valid_data_loader = PromptDataLoader(
    dataset=valid_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=16,
    shuffle=False
)
test_data_loader = PromptDataLoader(
    dataset=test_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=32,
    shuffle=False
)

promptModel = promptModel.cuda()

# ✅ 添加在创建PromptDataLoader后！！
sample = train_dataset[0]
wrapped_sample = promptTemplate.wrap_one_example(sample)
print("\nSample template processing:")
print("Raw text:", sample.text_a)
wrapped_text = "".join([d['text'] for d in wrapped_sample[0]])
print("Wrapped text:", wrapped_text)
print("Label:", sample.guid)

# 检查tokenization
tokenized = tokenizer(wrapped_text)
print("Tokenized:", tokenized)
print("Decoded:", tokenizer.decode(tokenized['input_ids']))
# Tokenize
print("Tokenized input_ids:", tokenized['input_ids'])
print("Tokenized attention_mask:", tokenized['attention_mask'])

# 解码回来，检查是否一致
decoded = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=False)
print("Decoded:", decoded)
tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
print("Tokens:", tokens)
def test(model, test_data_loader, output_file_true, output_file_pred):
    sum_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    model.eval()
    device = torch.device("cuda")

    # 创建文件以写入真实标签和预测标签
    with open(output_file_true, 'w') as true_file, open(output_file_pred, 'w') as pred_file:
        with torch.no_grad():
            for batch in test_data_loader:
                batch = batch.to(device)
                logits = model(batch)
                preds = torch.argmax(logits, dim=-1)

                # 计算正确预测数量
                sum_correct += torch.eq(batch['guid'], preds.cpu()).sum()

                # 写入真实标签和预测标签
                for true_label, pred_label in zip(batch['guid'].cpu().numpy(), preds.cpu().numpy()):
                    true_file.write(f"{true_label}\n")
                    pred_file.write(f"{pred_label}\n")

                # 计算召回率和精确率
                true_positive += ((batch['guid'] == 1) & (preds.cpu() == 1)).sum().item()
                false_positive += ((batch['guid'] == 0) & (preds.cpu() == 1)).sum().item()
                false_negative += ((batch['guid'] == 1) & (preds.cpu() == 0)).sum().item()

    # 输出准确率
    accuracy = sum_correct / len(test_dataset)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算精确率和召回率
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # 计算F-Measure
    if precision + recall > 0:
        f_measure = (2 * precision * recall) / (precision + recall)
    else:
        f_measure = 0

    # 输出精确率、召回率和F-Measure
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f_measure:.4f}")

        
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # 打印前五个 guid 和 label 的值
            if batch_idx == 0:  # 只在第一个 batch 打印
                print("First batch's guids and labels in evaluation:")
                print('guids:', batch['guid'][:5])

                
            input_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            labels = batch['guid'].to(device)
            logits = model(input_batch)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import f1_score
    return f1_score(all_labels, all_preds)

def train(model, train_data_loader):
    # ✅ 在这里添加 dropout 设置（在 model.cuda() 之后也可以，不影响）
    model.plm.config.hidden_dropout_prob = 0.2
    model.plm.config.attention_probs_dropout_prob = 0.2

    model = model.cuda()
    set_seed()
    # ---------------
    max_epochs = 5
    max_steps = max_epochs * len(train_data_loader)
    warm_up_steps = len(train_data_loader)
    output_dir = './saved_models'
    gradient_accumulation_steps = 1
    # lr = 2e-5
    lr = 1e-5
    adam_epsilon = 1e-8
    device = torch.device("cuda")
    max_grad_norm = 1.0
    # ----------------
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                 num_training_steps=max_steps)
    checkpoint_last = os.path.join(output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))

    logger.info("  Total optimization steps = %d", max_steps)
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    
    best_val_f1 = 0
    patience = 2
    no_improve_epochs = 0

    total_loss = 0.0
    sum_loss = 0.0
    best_model_path = ""
    for idx in range(0, max_epochs):
        total_loss = 0.0
        sum_loss = 0.0
        logger.info("******* Epoch %d *****", idx)
        for batch_idx, batch in enumerate(train_data_loader):
            if batch_idx == 0:  # 只在第一个 batch 打印
                print("First batch's guids and labels:")
                print('guids:', batch['guid'][:5])  # 假设 'guid' 是 tensor 类型

            batch.to(device)
            labels = batch['guid'].to(device)
            model.train()
            logits = model(batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

            sum_loss += loss.item()
            loss.backward()
            # ✅ 在loss.backward()后添加！！
            grad_norms = [
                torch.norm(p.grad).item()
                for p in model.parameters()
                if p.grad is not None
            ]
            if global_step % 10 == 0:
                print(f"Gradient norms - Min: {min(grad_norms):.6f} Max: {max(grad_norms):.6f}")
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if global_step % 50 == 0:
                    print('train/loss', sum_loss, global_step)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += sum_loss
                sum_loss = 0.
                global_step += 1
                
        # ✅ ✅ ✅【关键位置】每个 epoch 训练结束后，进行验证和早停判断
        # 注意：这个循环在 train loop 之外，epoch loop 之内
        val_f1 = evaluate(model, valid_data_loader)  # 你需要实现这个函数
        print(f"Epoch {idx} - Validation F1: {val_f1:.4f}")      
        
        # 早停判断
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            # 保存最佳模型
            best_model_path = f"best_model_epoch_{idx}.pt"
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {idx}")
                break
        logger.info(f"Training epoch {idx}, num_steps {global_step},  total_loss: {total_loss:.4f}")
    
    # ✅ ✅ ✅【关键位置】在训练和早停逻辑之后，加载最佳模型并进行最终测试
    if os.path.exists(best_model_path):
        print(f"\nTraining finished. Loading the best model from {best_model_path} for final evaluation.")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("\nBest model not found. Using the model from the last completed epoch for final evaluation.")
    
    # 最后，使用最佳模型在测试集上进行一次评估
    test(model, test_data_loader, 'true_labels.txt', 'pred_labels.txt')


train(promptModel, train_data_loader)