import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1
from pymoo.visualization.scatter import Scatter
from pysamoo.experimental.ParDen import ParDen

problem = ZDT1(n_var=10)

algorithm = NSGA2(pop_size=20, n_offsprings=10)

algorithm = ParDen(algorithm,                                      
                  n_max_doe=100,
                  n_max_infills=np.inf,
                  )

algorithm.setup(problem, seed=2, termination=('n_evals', 200))

for k in range(5):
    algorithm.next()
    print(algorithm.n_gen)

res = algorithm.result()

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

