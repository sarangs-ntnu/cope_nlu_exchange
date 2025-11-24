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
