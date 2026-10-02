"""Recover an optional audit by matching original evaluation batch boundaries."""
from pathlib import Path
import ast
import hashlib

HERE=Path(__file__).resolve().parent
source=(HERE/'aggregate_results.py').read_text(encoding='utf-8')
replacements={
    "(out/'validation_predictions').mkdir(); (out/'attention').mkdir()":
    "(out/'validation_predictions').mkdir(exist_ok=True); (out/'attention').mkdir(exist_ok=True)",
    "with (out/'validation_predictions'/f'{key}.npz').open('xb') as f: np.savez_compressed(f,**subset)":
    "validation_file=out/'validation_predictions'/f'{key}.npz'\n        if validation_file.exists():\n            with np.load(validation_file,allow_pickle=False) as prior:\n                require(tr.recursive_content_hash(dict(prior))==tr.recursive_content_hash(subset),'Preserved prior partial export mismatch')\n        else:\n            with validation_file.open('xb') as f: np.savez_compressed(f,**subset)",
    "q,a=tr.evaluate(model,data,vi,'cuda',audit_attention=True)\n            require(np.array_equal(q,pred['q_normal'][vi]),'Attention export changed predictions')":
    "q_all,a_all=tr.evaluate(model,data,device='cuda',audit_attention=True)\n            require(np.array_equal(q_all,pred['q_normal']),'Batch-aligned attention export changed predictions')\n            q,a=q_all[vi],a_all[vi]",
    "Local input/code/upstream preservation is verified when downloaded evidence is imported.":
    "The first optional audit attempt stopped because validation-only evaluation rebatched saved full-dataset predictions. The failed attempt is retained. Attention audit was recovered by using the original full-dataset batch boundaries and then selecting validation rows; all predictions were required to match bit-for-bit. No new tolerance, training, scientific source change or output replacement occurred. Local input/code/upstream preservation is verified when downloaded evidence is imported."
}
for before,after in replacements.items():
    assert source.count(before)==1, before
    source=source.replace(before,after)
ast.parse(source)
target=HERE/'aggregate_results_batch_aligned_v2.py'
with target.open('x',encoding='utf-8',newline='\n') as f: f.write(source)
print(hashlib.sha256(target.read_bytes()).hexdigest())
