import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# Function to load the model and tokenizer from Hugging Face
def load_huggingface_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Functions to evaluate metrics
def evaluate_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def evaluate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]
    rouge1 = sum(score['rouge1'].fmeasure for score in scores) / len(scores)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores) / len(scores)
    rougeL = sum(score['rougeL'].fmeasure for score in scores) / len(scores)
    return rouge1, rouge2, rougeL

def evaluate_bleu(predictions, references):
    smoothing = SmoothingFunction().method4
    bleu1 = sum(sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing) for pred, ref in zip(predictions, references)) / len(predictions)
    bleu2 = sum(sentence_bleu([ref.split()], pred.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing) for pred, ref in zip(predictions, references)) / len(predictions)
    bleu3 = sum(sentence_bleu([ref.split()], pred.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing) for pred, ref in zip(predictions, references)) / len(predictions)
    bleu4 = sum(sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing) for pred, ref in zip(predictions, references)) / len(predictions)
    return bleu1, bleu2, bleu3, bleu4

# Load dataset
dataset = load_dataset("HooshvareLab/pn_summary")
validation_data = dataset['validation']
eval_articles = [validation_data[i]['article'] for i in range(100)]
eval_titles = [validation_data[i]['title'] for i in range(100)]
eval_df = pd.DataFrame({'article': eval_articles, 'title': eval_titles})

# Load model and tokenizer from Hugging Face
model_name = "miirzamiir/title-generation" 
tokenizer, model = load_huggingface_model(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Tokenize and create dataloader
class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data['article']
        self.target_text = self.data['title']

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

eval_set = SummarizationDataset(eval_df, tokenizer, 512, 150)
eval_params = {
    'batch_size': 2,
    'shuffle': False,
    'num_workers': 0
}
eval_loader = torch.utils.data.DataLoader(eval_set, **eval_params)

# Evaluate the model
def evaluate_model(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            if _ % 10 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

print('Evaluating validation dataset')
preds, actuals = evaluate_model(tokenizer, model, device, eval_loader)

# Calculate BERTScore
P, R, F1 = evaluate_bertscore(preds, actuals)
print(f'BERTScore -> Precision: {P}, Recall: {R}, F1: {F1}')

# Calculate ROUGE
rouge1, rouge2, rougeL = evaluate_rouge(preds, actuals)
print(f'ROUGE -> ROUGE-1: {rouge1}, ROUGE-2: {rouge2}, ROUGE-L: {rougeL}')

# Calculate BLEU
bleu1, bleu2, bleu3, bleu4 = evaluate_bleu(preds, actuals)
print(f'BLEU -> BLEU-1: {bleu1}, BLEU-2: {bleu2}, BLEU-3: {bleu3}, BLEU-4: {bleu4}')

# Save results to CSV
final_df = pd.DataFrame({'Generated Text': preds, 'Actual Text': actuals})
final_df.to_csv('predictions.csv')

# Save evaluation metrics to a text file
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f'BERTScore -> Precision: {P}, Recall: {R}, F1: {F1}\n')
    f.write(f'ROUGE -> ROUGE-1: {rouge1}, ROUGE-2: {rouge2}, ROUGE-L: {rougeL}\n')
    f.write(f'BLEU -> BLEU-1: {bleu1}, BLEU-2: {bleu2}, BLEU-3: {bleu3}, BLEU-4: {bleu4}\n')
