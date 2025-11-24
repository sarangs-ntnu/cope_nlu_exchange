# Prompt-based zero/low-shot sentiment by aspect

## Objective
Evaluate instruction-tuned language models for aspect-aware sentiment without full supervised training, using zero-shot and few-shot prompting.

## Dataset assumptions
- Small labeled set available for few-shot examples; same columns as prior tasks.
- Sentiment label space known (e.g., positive/negative/neutral).

## Experiment steps
1. **Model selection**: Choose instruction models accessible in the environment (e.g., local Mistral/LLama variants or API-based analogs).
2. **Prompt design**:
   - Include aspect explicitly: "You are grading sentiment about the **teacher**."
   - Constrain the output label set and format (e.g., JSON with `sentiment`).
   - Prepare k-shot prompts with balanced examples per aspect.
3. **Decoding settings**: Use deterministic decoding (temperature=0–0.2) for consistency; limit max tokens.
4. **Evaluation**: Run zero-shot and few-shot settings; compute accuracy/macro F1 overall and per aspect. Record qualitative failures (hallucinated labels, ignoring aspect).
5. **Cost/runtime tracking**: Log latency and, if applicable, token costs.

## Reporting
Compare zero/low-shot performance to supervised baselines, noting cases where prompts succeed or fail (e.g., implicit sentiment, aspect confusion).

## How to run
1. Install dependencies for prompt calls (example uses a local transformer):
   ```bash
   pip install -r requirements.txt
   pip install transformers
   ```
2. Run a zero-shot prompt with a small instruction model (swap to a model available in your environment):
   ```bash
   python - <<'PY'
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch, pandas as pd

   model_name = 'mistralai/Mistral-7B-Instruct-v0.2'  # replace with a local/quantized model you can load
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

   prompt_tmpl = (
       "You are grading sentiment about the {aspect}. "
       "Answer with one label: positive, negative, or neutral.\n"
       "Comment: {text}\nSentiment:"
   )

   df = pd.read_csv('data/teacher_course.csv')
   sample = df.iloc[0]
   prompt = prompt_tmpl.format(aspect=sample['aspect'], text=sample['comments'])
   inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
   output = model.generate(**inputs, max_new_tokens=5, temperature=0.0)
   print(tokenizer.decode(output[0], skip_special_tokens=True))
   PY
   ```
3. For few-shot experiments, append k labeled examples per aspect to `prompt_tmpl` before the target comment, then iterate over the dataset to compute accuracy or macro F1.
4. If using an API-based model, set the required credentials (e.g., `OPENAI_API_KEY`) and swap the generation call for the provider's SDK while keeping the same prompt structure and deterministic decoding.
