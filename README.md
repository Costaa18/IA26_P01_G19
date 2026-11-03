# IA26_P01_G19

## Organização

O projeto implementa um agente para o CPDPTW de distribuição local:

- `model.py`: entidades do domínio, verificação de viabilidade e custo;
- `informed_search.py`: Greedy Best-First, A* e heurísticas;
- `hill_climbing.py`: pesquisa local com realocação e troca de pedidos;
- `csp.py`: CSP com backtracking, MRV e contagem de recuos;
- `main.ipynb`: experiências, validações, rotas, gráficos e comparação final.

## Executar

```powershell
uv run python -m unittest discover -s src -p "test*.py"
```

O notebook deve ser executado a partir da raiz do repositório. A pesquisa informada usa um limite de expansões e pode usar o fallback construtivo; essa situação é indicada por `used_constructive_fallback`. O Hill-Climbing parte da solução viável e regista a evolução do custo. O CSP reporta `backtracks`, atribuições, tempo e custo recalculado pelo modelo.