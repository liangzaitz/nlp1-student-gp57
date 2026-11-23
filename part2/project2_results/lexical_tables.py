import pickle
from pathlib import Path

base = Path('/home/scur0565/nlp1-student-gp57/part2/project2_results')
out = base / 'lexical_tables'
out.mkdir(parents=True, exist_ok=True)

with open(base / 'RESULTS_selected' / 'lexical_probes.pkl', 'rb') as f:
    probes = pickle.load(f)

for name, table in probes.items():
    full_table = table.round(3)
    full_table.to_csv(out / f'lexical_{name}.csv', index=False)
    full_table.style.hide(axis='index').to_html(out / f'lexical_{name}.html')
    print(f'Lexical probe: {name} ({len(full_table)} rows)')
    print(full_table.to_string(index=False))
