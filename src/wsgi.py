# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
import torch
import torch.distributed as dist

from eva_interactive import (
    get_inference_batch,
    generate_beam, 
    generate_no_beam,
    # Global vars
    model,
    tokenizer,
    args
)

app = Flask(__name__)


@app.route('/api/inference', methods=['POST'])
def inference():
    global models, tokenizer, args
    device = torch.cuda.current_device()

    data = request.get_json()
    context = data.get('query_context')

    with torch.no_grad():
        if dist.get_rank() == 0:
            full_context_list = [
                tokenizer.encode(line) + [tokenizer.sep_id]
                for line in context
            ]
            full_context = [x for y in full_context_list for x in y]
            trunc_context = []
            for utt in full_context_list[:-9:-1]:
                if len(trunc_context) + len(utt) + 1 <= 128:
                    trunc_context = utt + trunc_context
            trunc_context.append(tokenizer.get_sentinel_id(0))
            length_tensor = torch.tensor([len(trunc_context), len(full_context)], dtype=torch.long).to(device)
            trunc_context = torch.tensor(trunc_context, dtype=torch.long).to(device)
            full_context = torch.tensor(full_context, dtype=torch.long).to(device)
                
        else:
            length_tensor = torch.zeros(2, dtype=torch.long).to(device)

        dist.barrier()
        dist.broadcast(length_tensor, 0)
        if length_tensor[0] < 0:
            # continue
            return None
        if dist.get_rank() != 0:
            trunc_context = torch.zeros(int(length_tensor[0]), dtype=torch.long).to(device)
            full_context = torch.zeros(int(length_tensor[1]), dtype=torch.long).to(device)
        dist.broadcast(trunc_context, 0)
        dist.broadcast(full_context, 0)


        # encoder tensor
        trunc_context = trunc_context.unsqueeze(0).repeat(args.batch_size, 1) # repeat
        full_context = full_context.unsqueeze(0).repeat(args.batch_size, 1) 
        model_batch = get_inference_batch(trunc_context, device, args.batch_size, tokenizer, args)

        if args.num_beams == 1:
            generation_str_list, generation_id_list = generate_no_beam(model_batch, full_context, model, tokenizer, args, device)
        else:
            generation_str_list, generation_id_list  = generate_beam(model_batch, full_context, model, tokenizer, args, device)

        full_context_list.append(generation_id_list[0] + [tokenizer.sep_id])

        if dist.get_rank() == 0:
            # print("Sys >>> {}".format(generation_str_list[0]))
            answer = generation_str_list[0]
            print(f"Context: {context} Ans: {answer}")
            # answer = "DEBUG测试"
            return jsonify({
                'code': 200,
                'message': 'ok',
                'answer': answer
            })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6666)
