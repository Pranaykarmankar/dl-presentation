import json
with open('notebookc5cb3f14ca.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open('extracted_code.py', 'w', encoding='utf-8') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            f.write(''.join(cell['source']) + '\n\n')
