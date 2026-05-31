"""Post-process executed notebook: strip text/html outputs that block GitHub rendering."""
import json

with open('ab_test_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

stripped = 0
for cell in nb['cells']:
    for output in cell.get('outputs', []):
        data = output.get('data', {})
        if 'text/html' in data:
            del data['text/html']
            stripped += 1

with open('ab_test_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Stripped {stripped} text/html outputs — notebook ready for GitHub rendering")
