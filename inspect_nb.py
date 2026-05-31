import json
with open('ab_test_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

meta = nb.get('metadata', {})
print("nbformat:", nb.get('nbformat'), nb.get('nbformat_minor'))
print("kernelspec:", meta.get('kernelspec'))
li = meta.get('language_info', {})
print("language_info keys:", list(li.keys()))
print("version:", li.get('version'))
print("codemirror_mode:", li.get('codemirror_mode'))

# Check all outputs
for i, cell in enumerate(nb['cells']):
    outputs = cell.get('outputs', [])
    for o in outputs:
        keys = list(o.get('data', {}).keys())
        otype = o.get('output_type', '')
        if keys:
            print(f"Cell {i}: output_type={otype}, data keys={keys}")

print("Total cells:", len(nb['cells']))
