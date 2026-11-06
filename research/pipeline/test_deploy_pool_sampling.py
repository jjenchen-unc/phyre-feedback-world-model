"""Pure sampling contract checks without importing CUDA or remote dependencies."""
import ast
from pathlib import Path
import numpy as np


def main():
    source=Path(__file__).with_name('deploy_pool_updater.py')
    tree=ast.parse(source.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['draw_episode','query_indices']]
    namespace={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    tasks=[f'{i//100:05d}:{i%100:03d}' for i in range(1200)]
    for i in range(20000):
        task,cap,permutation,times=namespace['draw_episode'](tasks,i)
        assert task in tasks and cap in [1,2,4,8]
        assert len(set(permutation))==1024 and set(permutation)==set(range(1024))
        assert len(times)==8 and all(len(t)==8 and len(set(t))==8 and min(t)>=1 and max(t)<=31 for t in times)
        tried=permutation[:9].tolist()
        queries=namespace['query_indices'](permutation,tried)
        assert len(set(queries))==8 and not set(queries)&set(tried)
        assert queries==permutation[9:17].tolist()
    # Empty and successful-last-action episodes must also exclude EVERY trial.
    assert namespace['query_indices'](np.arange(1024),[0,1,2])==list(range(3,11))
    assert namespace['query_indices'](np.arange(1024),[])==list(range(8))
    print('PASS 20000 deterministic scene/cap/query/time schedules; all tried actions excluded')


if __name__=='__main__':main()
