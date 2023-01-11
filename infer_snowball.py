import argparse

import torch
from transformers import AutoTokenizer

from generator.models.relogic import RelogicModel
from evaluator.models.adversarial_evaluator import AdversarialModel
from preprocess.sql_formatter.formatting import translate_sql


def parse_arguments():
    parser = argparse.ArgumentParser(description="SQL Parser")
    parser.add_argument("--sql", type=str, help="SQL string that you want to translate into natural language description")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Pre-trained model name")
    parser.add_argument("--generator_model_save_path", type=str, default="/home/snowball/saves/spider_snow_ball_large/generator/4/pytorch_model.bin", help="SNOWBALL trained model path")
    parser.add_argument("--evaluator_model_save_path", type=str, default="/home/snowball/saves/spider_snow_ball_large/evaluator/4/pytorch_model.bin", help="SNOWBALL trained model path")
    # parser.add_argument("--model_save_path", type=str, default="/workspace/trained_models/best_model.pt/", help="SNOWBALL trained model path")
    return parser.parse_args()


def main(args):  
    def format_input_logic_string(tokenizer, logic_str):
        datastart_symbol = ["<SQL>"]
        return datastart_symbol + [tokenizer.cls_token] + tokenizer.tokenize(logic_str, add_prefix_space=True) + [tokenizer.sep_token]

    input_sql = args.sql

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    # Set and load generator model
    generator = RelogicModel(args.model_name)
    generator.load_state_dict(torch.load(args.generator_model_save_path))
    generator.eval()
    
    # Set and load evaluator model
    reranker = AdversarialModel(args.model_name)
    reranker.load_state_dict(torch.load(args.evaluator_model_save_path))
    reranker.eval()
    
    # Format input SQL string and tokenize
    _, translated_struct_sql = translate_sql(input_sql)
    formatted_input_tokens = format_input_logic_string(tokenizer, translated_struct_sql)
    formatted_input_token_ids = tokenizer.convert_tokens_to_ids(formatted_input_tokens)

    _, output_tok_ids = generator(**{
      "input_ids": torch.tensor(formatted_input_token_ids).unsqueeze(0),
      "reranker": reranker,
      "pad_token_id": 1,
      "label_bos_id": 0,
      "label_eos_id": 2,
      "label_padding_id": 1
    })
    output_nl = tokenizer.decode(output_tok_ids, skip_special_tokens=True)
    return output_nl


if __name__ == "__main__":
    args = parse_arguments()
    nl_text = main(args)
    print(f"SQL: {args.sql}")
    print(f"Translation result: {nl_text}")
