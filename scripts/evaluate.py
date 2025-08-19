import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_text(
    model, tokenizer, block_size, max_length=100, device='cpu',
    start_text="\n", temperature=1.0, top_k=None, beam_width=None
):
    model.eval()

    if beam_width is None:
        # === Sampling Mode ===
        context = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)
        for _ in range(max_length):
            if context.size(1) > block_size:
                context = context[:, -block_size:]
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)
        return tokenizer.decode(context[0].tolist())

    else:
        # === Beam Search Mode ===
        beam_width = int(beam_width)
        beams = [(tokenizer.encode(start_text), 0.0)]  # (tokens, score)

        for _ in range(max_length):
            candidates = []
            for tokens, score in beams:
                context = torch.tensor([tokens], dtype=torch.long).to(device)
                if context.size(1) > block_size:
                    context = context[:, -block_size:]
                logits = model(context)
                logits = logits[:, -1, :] / temperature
                probs = F.log_softmax(logits, dim=-1)

                top_probs, top_idxs = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    new_token = top_idxs[0, i].item()
                    new_score = score + top_probs[0, i].item()
                    candidates.append((tokens + [new_token], new_score))

            # Keep top-k beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        best_seq = beams[0][0]
        return tokenizer.decode(best_seq)

