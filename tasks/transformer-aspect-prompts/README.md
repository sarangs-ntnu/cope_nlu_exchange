# Aspect-prompted transformer fine-tuning

## Objective
Fine-tune a lightweight transformer with explicit aspect prompts to jointly model aspect context and sentiment.

## Dataset assumptions
- Input columns: `comments`, `sentiment`, `teacher/course` (aspect flag).
- Convert aspect flag into prompt text (e.g., `Aspect: teacher | Text: ...`).

## Experiment steps
1. **Model choice**: Start with `distilbert-base-uncased` or similar compact encoder for efficiency.
2. **Input formatting**: Build sequences like `Aspect: <aspect> [SEP] <comment>` or add special tokens for aspects; ensure max length fits model.
3. **Training setup**: 3–5 epochs with early stopping on validation macro F1. Use class weights if imbalance exists; optimizer AdamW with linear warmup.
4. **Hyperparameters to sweep**: learning rate (2e-5 to 5e-5), batch size, dropout, and aspect token strategy.
5. **Evaluation**: Macro F1 overall and per aspect; track training curves and validation loss. Save the best checkpoint and tokenizer.
6. **Error analysis**: Inspect misclassifications to see whether aspect prompts reduce confusion between teacher and course comments.

## Reporting
Provide training command(s), final metrics, and observations about the impact of aspect prompts versus unconditioned fine-tuning.

## How to run
1. Install dependencies with Hugging Face tooling:
   ```bash
   pip install -r requirements.txt
   pip install transformers datasets accelerate
   ```
2. Create a prompt-formatted CSV if your data does not already embed the aspect:
   ```bash
   python - <<'PY'
   import pandas as pd
   df = pd.read_csv('data/teacher_course.csv')
   df['prompted'] = 'Aspect: ' + df['aspect'].str.lower() + ' | Text: ' + df['comments']
   df[['prompted', 'sentiment']].to_csv('data/teacher_course_prompted.csv', index=False)
   PY
   ```
3. Fine-tune a compact encoder (DistilBERT example) using the Hugging Face Trainer:
   ```bash
   python - <<'PY'
   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
   import numpy as np
   from sklearn.metrics import f1_score

   model_name = 'distilbert-base-uncased'
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   dataset = load_dataset('csv', data_files='data/teacher_course_prompted.csv').class_encode_column('sentiment')

   def tokenize(batch):
       return tokenizer(batch['prompted'], truncation=True, padding='max_length', max_length=256)

   tokenized = dataset.map(tokenize, batched=True)
   tokenized = tokenized.rename_column('sentiment', 'labels').remove_columns(['prompted'])

   model = AutoModelForSequenceClassification.from_pretrained(
       model_name,
       num_labels=tokenized['train'].features['labels'].num_classes,
   )
   args = TrainingArguments(
       output_dir='outputs/transformer-aspect-prompts',
       evaluation_strategy='epoch',
       num_train_epochs=3,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=32,
   )

   def compute_metrics(eval_pred):
       logits, labels = eval_pred
       preds = np.argmax(logits, axis=-1)
       return {'macro_f1': f1_score(labels, preds, average='macro')}

   trainer = Trainer(
       model=model,
       args=args,
       train_dataset=tokenized['train'],
       eval_dataset=tokenized['train'],
       compute_metrics=compute_metrics,
   )
   trainer.train()
   trainer.save_model('outputs/transformer-aspect-prompts/best')
   PY
   ```
