import torch as pt
device = pt.device('cpu')


def ARInference(sp, model, prompt,reptition_penalty=1.5,top_p =0.9, max_length=100, temperature=0.8)->str:
    model.eval()
    with pt.no_grad():
        decoder_input = sp.encode(prompt)
        decoder_input = pt.tensor(decoder_input, dtype=pt.long, device=device).unsqueeze(0)
        EOT_ID = sp.piece_to_id('<EOT>')
        for i in range(max_length):
            logits = model.forward(decoder_input)
            logits = logits[:, -1, :]
            logits = logits / temperature
            for token_id in set(decoder_input.squeeze(0).tolist()[-30:]):
                if token_id not in [2, 3, 4, 5]:
                    logits[0, token_id] = logits[0, token_id] / reptition_penalty
            probs= pt.softmax(logits,dim=-1)
            sorted_probs, sorted_indices = pt.sort(probs, descending=True)
            total_cum=pt.cumsum(sorted_probs, dim=-1)
            shift = total_cum-sorted_probs
            mask=shift<=top_p
            sorted_probs[~mask]=0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            sampled_index = pt.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices[0, sampled_index[0]].reshape(1, 1)
            decoder_input = pt.cat([decoder_input, next_token], dim=1)
            if next_token.item() == EOT_ID:
                break
        prompt_length = len(sp.encode(prompt))
        generated_tokens = decoder_input.squeeze(0).tolist()[prompt_length:]
        if generated_tokens and generated_tokens[-1] == EOT_ID:
            generated_tokens = generated_tokens[:-1]
        return sp.decode(generated_tokens)
