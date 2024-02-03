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


# sys.stdout.reconfigure(encoding='utf-8')
# nltk.download('punkt')
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
    model[(src, trg)] = MarianMTModel.from_pretrained(f'/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Models/{src}_{trg}',config=config)
    tokenizer[(src, trg)] = MarianTokenizer.from_pretrained(f'/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Tokenizers/{src}_{trg}')
    
def answer_question(document, question, model, tokenizer):
    inputs = tokenizer(question, document, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits, dim=1).item()
    end_index = torch.argmax(end_logits, dim=1).item()
    answer = tokenizer.decode(inputs["input_ids"][0, start_index:end_index+1])
    return answer

def download_model(src, trg):
    config = MarianConfig.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')
    model[(src, trg)] = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')
    tokenizer[(src, trg)] = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src}-{trg}')

for id in names.keys():
    for language in names.keys():
        if id != language and not (
            (id == "ar" and language == "vi") or (id == "ar" and language == "zh") or (
            id == "vi" and language == "ar") or (id == "vi" and language == "zh") or (
            id == "zh" and language == "ar") or (id == "zh" and language == "fr")):
            if (id, language) not in model or (id, language) not in tokenizer:
                download_model(id, language)

def identify_language(query):
    langid.set_languages(['en', 'fr', 'es', 'de', 'ja', 'zh', 'ko', 'vi'])
    lang, confidence = langid.classify(query)
    return lang

# print(model)
print("Enter the query")
query = input()
id = identify_language(query)


input_folder = os.path.join(os.getcwd(), "books")
output_folder = os.path.join(os.getcwd(), "Documents")

for language in names.keys():
    names[language]["text"] = []
    names[language]["title"] = []
    path = output_folder + '/' + language
    print("path is :",path)
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(path + "/" + filename, 'r', encoding='utf8') as f:
                names[language]["text"].append(" ".join(f.readlines()))
                names[language]["title"].append(filename)

q = {id: query}
for language in names.keys():
    if language != id and not (
            (id == "ar" and language == "vi") or (id == "ar" and language == "zh") or (
            id == "vi" and language == "ar") or (id == "vi" and language == "zh") or (
            id == "zh" and language == "ar") or (id == "zh" and language == "fr")):
        if (id, language) not in model or (id, language) not in tokenizer:
            download_model(id, language)
        input_ids = tokenizer[(id, language)].encode(query, return_tensors="pt")
        gen = model[(id, language)].generate(input_ids)
        words: List[str] = tokenizer[(id, language)].batch_decode(gen, skip_special_tokens=True)
        q[language] = words

tokenized_text = {}
scores = {}
model_n = {
         "de": "deutsche-telekom/bert-multi-english-german-squad2",
         "en": "deepset/bert-base-uncased-squad2",
         "vi": "ancs21/xlm-roberta-large-vi-qa",
         "zh": "ckiplab/bert-base-chinese-qa"}

for language in names.keys():
    tokenized_text[language] = [nltk.word_tokenize(txt) for txt in names[language]["text"]]
    
    if(tokenized_text[language]==[]):
        continue

    bm25 = BM25Okapi(tokenized_text[language])
    tokenized_query = nltk.word_tokenize(str(q[language]))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores_with_indices = [(score, i) for i, score in enumerate(bm25_scores)]
    bm25_scores_with_indices.sort(key=lambda x: x[0], reverse=True)
    
    top_document_text = []
    top1 = bm25_scores_with_indices[0][1]
    print(names[language]["title"][top1])
    top_document_text.append(names[language]["text"][top1])

    if(len(bm25_scores_with_indices) >= 2 and len(bm25_scores_with_indices[1]) >= 2): 
        top2 = bm25_scores_with_indices[1][1]
        print(names[language]["title"][top2])
        top_document_text.append(names[language]["text"][top2])
    
    tokenized_text[language] = []
    for txt in top_document_text:
        l = list(filter(lambda a: a != ' ', txt.split("\n")))
        i = 0
        while (i < len(l)):
            j = l[i]
            if(len(j.replace('.',' ').replace(',',' ').split(' ')) < 500) and i+1 < len(l):
                l[i] += l[i+1]
                l.pop(i+1)
            else:
                i+=1

        for para in l:
            tokenized_text[language].append(nltk.word_tokenize(para))

    if tokenized_text[language] == []:
        continue

    tokenized_query = nltk.word_tokenize(str(q[language]))
    bm25 = BM25Okapi(tokenized_text[language])
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores_with_indices = [(score, i) for i, score in enumerate(bm25_scores)]
    bm25_scores_with_indices.sort(key=lambda x: x[0], reverse=True) 
    print(f"Language: {language}")
    
    top_para = ""
    for score, index in bm25_scores_with_indices:
#         print(f"Score: {score}, Paragraph: {' '.join(tokenized_text[language][index])}")
        top_para = ' '.join(tokenized_text[language][index])
        break
    
    model_name = model_n[language]
#     tokenizer = BertTokenizer.from_pretrained((f'Tokenizers/{language}_BERT'))
#     model = BertForQuestionAnswering.from_pretrained((f'Models/{language}_BERT'))

    if(language=="vi"):
        tokenizer_id = XLMRobertaTokenizer.from_pretrained(model_name)
        model_id = XLMRobertaForQuestionAnswering.from_pretrained(model_name)
    else:

        tokenizer_id = BertTokenizer.from_pretrained(model_name)
        model_id = BertForQuestionAnswering.from_pretrained(model_name)
        
    answer = answer_question(top_para, str(q[language]), model_id, tokenizer_id)
    print(f"Question: {str(q[language])}")
    print(f"Answer: {answer}\n")
