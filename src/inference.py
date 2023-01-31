import os

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']="1"

import sys
sys.path.append('../')
import nltk
import numpy as np
#import tdqm
import argparse
import torch
# import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer
from datasets import load_metric
from data.dataset import SamsumDataset_total, DialogsumDataset_total, MediasumDataset_total, TweetsummDataset_total
from models.bart import BartForConditionalGeneration_DualDecoder, BartForConditionalGeneration_DualHead
from tqdm import tqdm

# Set Argument Parser
parser = argparse.ArgumentParser()
# Training hyperparameters
parser.add_argument('--dataset_name',type=str, default='samsum')
parser.add_argument('--model_checkpoint', type=str, default="./new_weights_comet/final_Trial1_context_comet")
parser.add_argument('--test_output_file_name', type=str, default='./new_weights_comet/final_Trial1_context_comet.txt')
parser.add_argument('--train_configuration',type=str,default="full")# base, context, supervision, full
parser.add_argument('--encoder_max_len', type=int, default=1024)
parser.add_argument('--decoder_max_len', type=int, default=100)
parser.add_argument('--use_paracomet',type=bool,default=False)
parser.add_argument('--use_roberta',type=bool,default=False)
parser.add_argument('--use_sentence_transformer',type=bool,default=False)
parser.add_argument('--relation',type=str,default="xReason")
parser.add_argument('--supervision_relation',type=str,default='isAfter')
parser.add_argument('--num_beams', type=int, default=20)
args = parser.parse_args()

# Set GPU
print('######################################################################')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print(torch.cuda.get_device_name())
print('######################################################################')

# Define Global Values
# model_checkpoint_list = [
#     "final_Trial2_base_BART_Samsum",
#     "final_Trial2_context_BART_Samsum",
#     "final_Trial4_supervision_BART_Samsum",
#     "final_Trial4_full_BART_Samsum",
#     "final_Trial5_supervision_BART_Samsum",
#     "final_Trial5_full_BART_Samsum",
#     "final_TRIAL3_Dialogsum_base_BART",
#     "final_TRIAL3_Dialogsum_context_BART",
#     "final_TRIAL4_Dialogsum_full_BART",
#     "final_TRIAL4_Dialogsum_supervision_BART"
#     "final_TRIAL3_Dialogsum_head_full_BART",
#     "final_TRIAL3_Dialogsum_head_supervision_BART"
# ]

model_checkpoint_list = [
    "final_Trial2_base_BART_Samsum",
    "final_Trial2_context_BART_Samsum",
    "final_Trial4_supervision_BART_Samsum",
    "final_Trial5_supervision_BART_Samsum",
    "final_Trial6_full_BART_Samsum", #xsum
    "final_Trial7_full_BART_Samsum", #xsum
    "final_TRIAL3_Dialogsum_base_BART",
    "final_TRIAL4_Dialogsum_context_BART",
    "final_TRIAL1_Dialogsum_supervision_BART",
    "final_TRIAL7_Dialogsum_full_BART"
    "final_TRIAL2_Dialogsum_head_supervision_BART",
    "final_TRIAL8_Dialogsum_head_full_BART",
    "final_Trial5_full_BART_Dialogsum", #xsum
    "final_Trial6_full_BART_Dialogsum", #xsum
]
extra_supervision = False
extra_context=False    
# Samsum
if args.train_configuration == "base":
    finetune_model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
elif args.train_configuration == "context":
    finetune_model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
    extra_context = True
elif args.train_configuration == "supervision":
    finetune_model = BartForConditionalGeneration_DualDecoder.from_pretrained(args.model_checkpoint)
    extra_supervision = True
elif args.train_configuration =="full":
    finetune_model = BartForConditionalGeneration_DualDecoder.from_pretrained(args.model_checkpoint)
    extra_supervision = True
    extra_context=True
else:
    assert "Model checkpoint is not valid"


print('######################################################################')
print("Number of Model Parameters are : ",finetune_model.num_parameters())
print('######################################################################')



# Set extra Configuration for Finetuning on Summarization Dataset
finetune_model = finetune_model.to(device)
finetune_model.eval()


# Set metric
metric = load_metric("../utils/rouge.py")
metric2 = load_metric("../utils/rouge.py")
metric3 = load_metric("../utils/rouge.py")

bertscore_metric = load_metric("bertscore",lang='en',model_type='bert-base-uncased')
if args.dataset_name=='dialogsum':
    bertscore_metric2 = load_metric("bertscore",lang='en',model_type='bert-base-uncased')
    bertscore_metric3 = load_metric("bertscore",lang='en',model_type='bert-base-uncased')


# Load Tokenizer associated to the model
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)


# Set dataset
if args.dataset_name=='samsum':
    total_dataset = SamsumDataset_total(args.encoder_max_len,args.decoder_max_len,tokenizer,extra_context=True,extra_supervision=True,paracomet=args.use_paracomet,relation=args.relation,supervision_relation=args.supervision_relation,roberta=args.use_roberta, sentence_transformer=args.use_sentence_transformer)
    test_dataset = total_dataset.getTestData()
elif args.dataset_name=='dialogsum':
    total_dataset = DialogsumDataset_total(args.encoder_max_len,args.decoder_max_len,tokenizer,extra_context=True,extra_supervision=True,paracomet=args.use_paracomet,relation=args.relation,supervision_relation=args.supervision_relation, sentence_transformer=args.use_sentence_transformer, roberta=args.use_roberta)
    test_dataset = total_dataset.getTestData()
print('######################################################################')
print('Test Dataset Size is : ')
print(len(test_dataset))
print('######################################################################')

# Build dataloader
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

total_rouge1_scores = 0.0
total_rouge2_scores = 0.0
total_rougeL_scores = 0.0
total_rougeLsum_scores = 0.0
total_bert_scores = 0.0
total_decoded_preds = []
total_decoded_labels = []

with torch.no_grad():
    for idx, data in enumerate(tqdm(test_dataloader),0):
        # if idx % 40 ==0:
        #     print(total_rouge1_scores)
        #     print(idx)
        x = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        y = data['labels'].to(device, dtype=torch.long)

        generated_ids = finetune_model.generate(
            input_ids= x,
            attention_mask=mask,
            max_length=100,
            num_beams=args.num_beams
        )

        generated_ids = generated_ids.cpu()
        y = y.cpu()

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        y = np.where(y != -100, y, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        #print('################')
        #print(decoded_preds)
        #print()
        #print()
        #print(decoded_labels)
        #print('############')
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        bertscore_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

       

        if args.dataset_name=='dialogsum':
            y2 = data['labels2'].to(device,dtype=torch.long)
            y2 = y2.cpu()
            y3 = data['labels3'].to(device,dtype=torch.long)
            y3 = y3.cpu()

            decoded_labels2 = tokenizer.batch_decode(y2, skip_special_tokens=True, clean_up_tokenization_space=True)
            decoded_labels3 = tokenizer.batch_decode(y3, skip_special_tokens=True, clean_up_tokenization_space=True)

            decoded_labels2 = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels2]
            decoded_labels3 = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels3]

            result2 = metric2.add_batch(predictions=decoded_preds, references=decoded_labels2)
            result3 = metric3.add_batch(predictions=decoded_preds, references=decoded_labels3)

            bertscore_metric2.add_batch(predictions=decoded_preds, references=decoded_labels2)
            bertscore_metric3.add_batch(predictions=decoded_preds, references=decoded_labels3)


            

  
        total_decoded_preds.append(decoded_preds)
        total_decoded_labels.append(decoded_labels)
        
       
        
            
       
bertscore_result = bertscore_metric.compute(lang='en',model_type='bert-base-uncased')
result = metric.compute(use_stemmer=True)

if args.dataset_name == "dialogsum":
    result2 = metric2.compute(use_stemmer=True)
    result3 = metric3.compute(use_stemmer=True)

    bertscore_result2 = bertscore_metric2.compute(lang='en',model_type='bert-base-uncased')
    bertscore_result3 = bertscore_metric3.compute(lang='en',model_type='bert-base-uncased')


    



result = {key: value.mid.fmeasure * 100 for key, value in result.items()}  
bertscore_result = sum(bertscore_result['f1'])/len(bertscore_result['f1'])

if args.dataset_name == "dialogsum":
    bertscore_result2 = sum(bertscore_result2['f1'])/len(bertscore_result2['f1'])
    bertscore_result3 = sum(bertscore_result3['f1'])/len(bertscore_result3['f1'])

    result2 = {key: value.mid.fmeasure * 100 for key, value in result2.items()}   
    result3 = {key: value.mid.fmeasure * 100 for key, value in result3.items()}   



NB = str(args.num_beams)
print("num_beams: " + NB)
print("bert-score")
print(bertscore_result)
print(result)

if args.dataset_name == "dialogsum":
    print(result2)
    print(result3)

with open(args.test_output_file_name,"a") as f: 
    f.write("num_beams: ")
    f.write(NB+"\n")
    f.write(" rouge1: ")
    f.write(str(result['rouge1'])+"\n")
    f.write(" rouge2: ")
    f.write(str(result['rouge2'])+"\n")
    f.write(" rougeL: ")
    f.write(str(result['rougeL'])+"\n")
    f.write(" rougeLsum: ")
    f.write(str(result['rougeLsum'])+"\n")
    f.write(" bert-score: ")
    f.write(str(bertscore_result)+"\n")
    f.write('\n')
    if args.dataset_name=='dialogsum':
        f.write("num_beams: ")
        f.write(NB+"\n")
        f.write(" rouge1: ")
        f.write(str(result2['rouge1'])+"\n")
        f.write(" rouge2: ")
        f.write(str(result2['rouge2'])+"\n")
        f.write(" rougeL: ")
        f.write(str(result2['rougeL'])+"\n")
        f.write(" rougeLsum: ")
        f.write(str(result2['rougeLsum'])+"\n")
        f.write(" bert-score: ")
        f.write(str(bertscore_result2)+"\n")
        f.write('\n')

        f.write("num_beams: ")
        f.write(NB+"\n")
        f.write(" rouge1: ")
        f.write(str(result3['rouge1'])+"\n")
        f.write(" rouge2: ")
        f.write(str(result3['rouge2'])+"\n")
        f.write(" rougeL: ")
        f.write(str(result3['rougeL'])+"\n")
        f.write(" rougeLsum: ")
        f.write(str(result3['rougeLsum'])+"\n")
        f.write(" bert-score: ")
        f.write(str(bertscore_result3)+"\n")
        f.write('\n')
    for i in total_decoded_preds:
        for sent in i:
            f.write(sent.replace("\n"," ")+"\n")
