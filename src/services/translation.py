import torch as pt
device=pt.device('cpu')


def translate(model,tokenizer,english_sentence,top_p=0.9,max_len=128,temperature =0.8,device=device)->str:
    #tokenize input
     english_tokens=tokenizer.encode(english_sentence,out_type=int,add_eos=True)
     english_tokens = pt.tensor(english_tokens, dtype=pt.long,device=device).unsqueeze(0)
   
    # adding a false mask because the tranformer expects some masking
     src_mask = pt.zeros(1, english_tokens.shape[1], dtype=pt.bool, device=device)
     enc_embed = model.encoder_embedding.embedding(english_tokens)
     enc_embed = model.encoder_embedding.positional_encoding(enc_embed)
     enc_out = model.Encoderblocks.encoder_forward(enc_embed, src_mask)
  

     #decoder section
     bos_id = tokenizer.bos_id()
     eos_id = tokenizer.eos_id()
     decoder_input = pt.tensor([[bos_id]], dtype=pt.long, device=device)
   
     for _ in range(max_len):
         trg_mask = pt.zeros(1, decoder_input.shape[1], dtype=pt.bool, device=device)
         dec_embed = model.decoder_embedding.embedding(decoder_input)
         dec_embed = model.decoder_embedding.positional_encoding(dec_embed)
         dec_out = model.Decoderblocks(dec_embed,enc_out,src_mask,trg_mask)
         logits = model.linear(dec_out)
         logits =logits[:, -1, :]
         logits = logits/temperature
         probs= pt.softmax(logits,dim=-1)
         sorted_probs, sorted_indices = pt.sort(probs, descending=True)
         total_cum=pt.cumsum(sorted_probs, dim=-1)
         shift = total_cum-sorted_probs
         mask=shift<=top_p
         sorted_probs[~mask]=0.0
         sorted_probs = sorted_probs / sorted_probs.sum()
         sampled_index = pt.multinomial(sorted_probs, num_samples=1)
         next_token = sorted_indices[0, sampled_index[0]].reshape(1, 1)
         decoder_input = pt.cat([decoder_input,next_token],dim=1)
         if next_token == eos_id:
             break
     generated_ids = decoder_input.squeeze(0).tolist()[1:]  # skip <s>
     french = tokenizer.decode(generated_ids)
     return french

