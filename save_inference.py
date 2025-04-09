import click
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


@click.command()
@click.option("--model_name", type=str, required=True)
@click.option("--data_path", type=str, required=True)
@click.option("--output_csv_path", type=str, required=True)
@click.option("--output_hidden_dir", type=str, required=True)
@click.option("--output_column", type=str, default="output")
@click.option("--save_every", type=int, default=500)
@click.option("--start_idx", type=int, default=0)
@click.option('--stop_idx', type=int, default=None)
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--stop_strings", type=str, default="[STOP]")
@click.option("--remove_stop_strings", type=bool, default=True)
@click.option("--save_hidden", type=bool, default=True)
@click.option("--track_other_layers", type=bool, default=True)
def main(model_name, data_path, output_csv_path, output_hidden_dir, output_column, save_every, start_idx, stop_idx, max_new_tokens, stop_strings, remove_stop_strings, save_hidden, track_other_layers):
    if start_idx != 0:
        assert False, "start_idx is not implemented yet because of hidden state saving"
    stop_strings = stop_strings.split(",")
    if save_hidden:
        makedirs = [output_hidden_dir]
    else:
        makedirs = []
    for output_path in [output_csv_path]:
        if os.path.dirname(output_path) != "":
            makedirs.append(os.path.dirname(output_path))
    for directory in makedirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    layer_to_track = config.num_hidden_layers // 2
    if track_other_layers:
        min_layer_to_track = 1
        max_layer_to_track = config.num_hidden_layers - 1
        other_layers_to_track = list(range(min_layer_to_track, max_layer_to_track + 1))
        if layer_to_track in other_layers_to_track:
            other_layers_to_track.remove(layer_to_track)
        other_layers_to_track = other_layers_to_track[::4]
    else:
        other_layers_to_track = []
    all_layers_to_track = [layer_to_track] + other_layers_to_track
    data_df = pd.read_csv(data_path)
    assert "text" in data_df.columns, f"Dataframe must have a 'text' column with columns {data_df.columns}"
    assert output_column not in data_df.columns, f"Dataframe already has an output column {output_column} with columns {data_df.columns}"
    data_df[output_column] = None
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    if stop_idx is not None:
        stop_idx = min(stop_idx, len(data_df))
    else:
        stop_idx = len(data_df)
    hidden_states_lists = {}
    attention_lists = {}
    for layer in all_layers_to_track:
        hidden_states_lists[layer] = []
        attention_lists[layer] = []
    for i in tqdm(range(start_idx, stop_idx)):
        prompt = data_df.loc[i, "text"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, stop_strings=stop_strings, pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_attentions=True, output_hidden_states=True, output_scores=True, return_dict_in_generate=True)
        output_sequences = output.sequences
        output_normed_perplexity = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True).mean().detach().cpu().numpy().item()
        if save_hidden:
            hidden_states = output.hidden_states[0][layer_to_track][0, -1].detach().cpu().numpy()
            for layer in all_layers_to_track:
                hidden_states = output.hidden_states[0][layer][0, -1].detach().cpu().numpy()
                hidden_states_lists[layer].append(hidden_states)
            attentions = output.attentions[0][layer_to_track][0, -1].detach().cpu().numpy()
            for layer in all_layers_to_track:
                pass
                #attentions = output.attentions[0][layer][0, -1].detach().cpu().numpy()
                #attention_lists[layer].append(attentions)
        output_only = output_sequences[0, input_length:]
        out = tokenizer.decode(output_only, skip_special_tokens=True)
        if remove_stop_strings:
            for stop_string in stop_strings:
                out = out.replace(stop_string, "")
        data_df.loc[i, output_column] = out
        data_df.loc[i, "perplexity"] = output_normed_perplexity
        if (i % save_every == 0 and i > 0) or i == stop_idx - 1:
            data_df.to_csv(output_csv_path, index=False)
            if save_hidden:
                for layer in all_layers_to_track:
                    array = np.array(hidden_states_lists[layer])
                    np.save(os.path.join(output_hidden_dir, f"hidden_states_{layer}.npy"), array)
                    #array = np.array(attention_lists[layer])
                    #np.save(os.path.join(output_hidden_dir, f"attentions_{layer}.npy"), array)
    return 

if __name__ == "__main__":
    main()