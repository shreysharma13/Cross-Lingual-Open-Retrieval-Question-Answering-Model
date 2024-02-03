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

from saved_model import model, tokenizer,names, target_languages,identify_language,answer_question



# sys.stdout.reconfigure(encoding='utf-8')
# nltk.download('punkt')

# print(model)

def listFiles():
    documents_folder = "/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Documents/"
    l = {"en" : [], 
         "ar": [], 
         "vi": [], 
         "de" : [], 
         "zh": [] }
    for i in l.keys():
        all_files = os.listdir(documents_folder + i)
        for filename in all_files:
            if filename.endswith(".txt"):
                l[i].append(filename)
    return l

def translate(input_str,language):
    id = identify_language(input_str)
    if language != id and not (
                (id == "ar" and language == "vi") or (id == "ar" and language == "zh") or (
                id == "vi" and language == "ar") or (id == "vi" and language == "zh") or (
                id == "zh" and language == "ar") or (id == "zh" and language == "fr")):
            input_ids = tokenizer[(id, language)].encode(input_str, return_tensors="pt")
            gen = model[(id, language)].generate(input_ids)
            words: List[str] = tokenizer[(id, language)].batch_decode(gen, skip_special_tokens=True)
            return (words)
    return input_str


def ask_query(query):
# print("Enter the query: ")
# query = input()
    # ans_arr =[]
    print(query)
    id = identify_language(query)


    # input_folder = os.path.join(os.getcwd(), "books")
    output_folder = os.path.join(os.getcwd(), r"/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1/Documents")

    # output_folder = Whaos.path.join(
    #     "/Users/shreysharma/Documents/Shrey SNU/Seventh Sem/Cross-lingual Question Answering Model/cross-lingual-frontend/major-project_trial1",
    #     "Documents"
    # )


    for language in names.keys():
        names[language]["text"] = []
        names[language]["title"] = []
        path = output_folder + '/' + language
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
        elif (language != id):
            q[language] = []

    tokenized_text = {}
    scores = {}
    model_n = {
            "de": "deutsche-telekom/bert-multi-english-german-squad2",
            "en": "deepset/bert-base-uncased-squad2",
            "vi": "ancs21/xlm-roberta-large-vi-qa",
            "zh": "ckiplab/bert-base-chinese-qa"}

    ans={}
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
        ans[language] = [str(q[language]),answer]
        print(f"Question: {str(q[language])}")
        print(f"Answer: {answer}\n")
        # print(type(answer))
        # ans_arr.append(answer)
    return ans
    
    



