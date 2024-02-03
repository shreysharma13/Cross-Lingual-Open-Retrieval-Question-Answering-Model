import sys
print(sys.prefix != sys.base_prefix)
from transformers import MarianTokenizer, MarianMTModel, MarianConfig
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering
import torch
import os
import sys
import torch
import nltk
import langid
from rank_bm25 import BM25Okapi
from typing import List
import langid

def answer_question(document, question, model, tokenizer):
    inputs = tokenizer(question, document, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    answer = tokenizer.decode(inputs["input_ids"][0, start_index:end_index+1])
    return answer

def identify_language(query):
    langid.set_languages(['en', 'fr', 'es', 'de', 'ja', 'zh', 'ko', 'vi'])
    lang, confidence = langid.classify(query)
    return lang

def download_model(src, trg):
    config = MarianConfig.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')
    model[(src, trg)] = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')
    tokenizer[(src, trg)] = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')

# for id in names.keys():
#     for language in names.keys():
#         if id != language and not (
#             (id == "ar" and language == "vi") or (id == "ar" and language == "zh") or (
#             id == "vi" and language == "ar") or (id == "vi" and language == "zh") or (
#             id == "zh" and language == "ar") or (id == "zh" and language == "fr")):
#             if (id, language) not in model or (id, language) not in tokenizer:
#                 download_model(id, language)


names = {"ar": {},
         "de": {},
         "en": {},
         "vi": {},
         "zh": {}}

target_languages = ["en", "ar", "vi", "de", "zh"]

model_names = {}
model = {}
tokenizer = {}
for src in target_languages:
    for trg in target_languages:
        if (src == "ar" and trg == "vi") or (src == "ar" and trg == "zh") or (src == "vi" and trg == "ar") or (src == "vi" and trg == "zh") or (src == "zh" and trg == "ar") or (src == "zh" and trg == "fr"):
            continue
        if trg != src:
            model_name = f'Helsinki-NLP/opus-mt-{src}-{trg}'
            model_names[(src, trg)] = model_name
            model[(src, trg)] = f"model_name_{src}_{trg}"
            
for (src, trg), model_name in model_names.items():
    config = MarianConfig.from_pretrained(model_names[(src, trg)])
    model[(src, trg)] = MarianMTModel.from_pretrained(f"/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Models/{src}_{trg}",config=config)
    tokenizer[(src, trg)] = MarianTokenizer.from_pretrained(f'/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Tokenizers/{src}_{trg}')
    
print("Model Saved")