
with open('/workspaces/quantoniumos/docs/NOVEL_ALGORITHMS.md', 'r') as f:
    content = f.read()
    if 'resonant_operator_variants.py' in content:
        print("FOUND")
    else:
        print("NOT FOUND")
