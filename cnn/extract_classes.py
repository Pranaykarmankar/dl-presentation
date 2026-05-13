import json
with open('notebookc5cb3f14ca.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

classes = ""
for cell in nb['cells']:
    if 'outputs' in cell:
        for out in cell['outputs']:
            if out.get('name') == 'stdout':
                for text in out['text']:
                    if 'Classes detected' in text:
                        classes += text

with open('classes.txt', 'w', encoding='utf-8') as f:
    f.write(classes)
