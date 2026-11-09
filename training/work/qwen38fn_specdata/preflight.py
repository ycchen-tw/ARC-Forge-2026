"""Record token-level smoke requests and compare MTP against a recurrent baseline."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


async def main(args):
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        "What is 17 times 23? Give the answer.",
        "Write a Python function that returns all primes up to n using a sieve. Explain its complexity.",
        "Find the sum of all positive integers below 1000 divisible by 3 or 5. Explain the calculation.",
        ("Reference notes: the blue box has 7 marbles, the red box has 11 marbles.\n" * 750)
        + "How many marbles are in the two boxes together? Give the total.",
    ]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=1800) as client:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            try:
                health=await client.get(args.server+'/health',timeout=5)
                if health.status_code==200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(5)
        else:
            raise RuntimeError('server readiness deadline exceeded')
        r = await client.get(args.server + "/get_server_info")
        r.raise_for_status()
        (out / "server.json").write_text(json.dumps(r.json(), indent=2))
        for i, prompt in enumerate(prompts):
            for effort in ("low", "medium", "xhigh"):
                ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, reasoning_effort=effort)
                payload = {"input_ids": ids, "sampling_params": {"temperature": 0.0,
                    "top_p": 1.0, "top_k": -1, "max_new_tokens": args.max_new,
                    "skip_special_tokens": False}, "return_logprob": True, "top_logprobs_num": 5,
                    "logprob_start_len": -1}
                r = await client.post(args.server + "/generate", json=payload)
                r.raise_for_status()
                response = r.json()
                assert response['meta_info']['prompt_tokens'] == len(ids)
                assert response['meta_info']['completion_tokens'] == len(response['output_ids'])
                record = {"input_ids": ids, "response": response}
                name = f"{i}-{effort}.json"
                (out / name).write_text(json.dumps(record))
                summary = {"case": name, "n_prompt": len(ids), "n_output": len(response['output_ids']),
                           "finish": response['meta_info']['finish_reason']}
                if args.compare:
                    baseline = json.loads((args.compare / name).read_text())
                    assert baseline['input_ids'] == ids
                    a, b = baseline['response']['output_ids'], response['output_ids']
                    common = next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
                    summary.update(equal=a==b, common_prefix=common)
                    if common < min(len(a),len(b)):
                        left=baseline['response']['meta_info']['output_top_logprobs'][common]
                        right=response['meta_info']['output_top_logprobs'][common]
                        lm={x[1]:x[0] for x in left}
                        rm={x[1]:x[0] for x in right}
                        shared=lm.keys() & rm.keys()
                        summary.update(first_divergence_logprob_max_delta=max(abs(lm[t]-rm[t]) for t in shared),
                            baseline_top2_gap=left[0][0]-left[1][0], replay_top2_gap=right[0][0]-right[1][0])
                    (out / (name+'.comparison.json')).write_text(json.dumps(summary))
                print(json.dumps(summary), flush=True)
    if args.compare:
        matches = []
        for p in out.glob('[0-9]-*.json'):
            a, b = json.loads((args.compare / p.name).read_text()), json.loads(p.read_text())
            matches.append(a['response']['output_ids'] == b['response']['output_ids'])
        report = {"cases": len(matches), "exact": sum(matches)}
        (out / 'comparison.json').write_text(json.dumps(report))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--server', default='http://127.0.0.1:30471')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--max-new', type=int, default=2048)
    asyncio.run(main(p.parse_args()))
