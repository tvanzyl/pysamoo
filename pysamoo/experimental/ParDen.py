import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.selection.tournament import compare
from pymoo.optimize import default_termination
from pymoo.util.display import Display
from pymoo.util.dominator import get_relation
from pymoo.util.misc import norm_eucl_dist, cdist
from pymoo.util.optimum import filter_optimum
from pymoo.util.termination.no_termination import NoTermination
from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.visualization.video.callback_video import AnimationCallback

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
# =========================================================================================================
# Display
# =========================================================================================================
from pysamoo.core.knockout import noisy


class ParDenDisplay(Display):

    def __init__(self, display, **kwargs):
        super().__init__(**kwargs)
        self.display = display

    def _do(self, problem, evaluator, gpasf):
        self.display.do(problem, evaluator, gpasf.algorithm, show=False)
        self.output = self.display.output

        if gpasf.n_gen > 1:
            surr_infills = Population.create(*gpasf.infills.get("created_by"))
            n_influenced = sum(surr_infills.get("type") == "trace")
            self.output.append("n_influenced", f"{n_influenced}/{len(surr_infills)}")
        else:
            self.output.append("n_influenced", "-")

        if problem.n_obj == 1 and problem.n_constr == 0:
            if len(gpasf.surrogate.targets) >= 1:
                target = gpasf.surrogate.targets[0]
                self.output.append("mae", target.performance("mae"))
                self.output.append("model", target.best)

        elif problem.n_obj == 2 and problem.n_constr == 0:
            if len(gpasf.surrogate.targets) >= 2:
                perf = gpasf.surrogate.performance("mae")
                if ("F", 0) in perf:
                    self.output.append("mae f1", perf[("F", 0)])
                if ("F", 1) in perf:
                    self.output.append("mae f2", perf[("F", 1)])


# =========================================================================================================
# Animation
# =========================================================================================================

class ParDenAnimation(AnimationCallback):

    def __init__(self,
                 nth_gen=1,
                 n_samples_for_surface=200,
                 dpi=200,
                 **kwargs):

        super().__init__(nth_gen=nth_gen, dpi=dpi, **kwargs)
        self.n_samples_for_surface = n_samples_for_surface
        self.last_pop = None

    def do(self, problem, algorithm):

        if problem.n_var != 2 or problem.n_obj != 1:
            raise Exception(
                "This visualization can only be used for problems with two variables and one objective!")

        # draw the problem surface
        doe = algorithm.surrogate.targets["F"].doe
        if doe is not None:
            problem = algorithm.surrogate

        plot = FitnessLandscape(problem, _type="contour", kwargs_contour=dict(alpha=0.5))
        plot.do()

        if doe is not None:
            plt.scatter(doe.get("X")[:, 0], doe.get("X")[:, 1], color="black", alpha=0.3)

        for k, sols in enumerate(algorithm.trace_assigned):
            if len(sols) > 0:
                pop = Population.create(*sols)
                plt.scatter(pop.get("X")[:, 0], pop.get("X")[:, 1], color="blue", alpha=0.3)

                x = algorithm.influenced[k].X
                for sol in sols:
                    plt.plot((x[0], sol.X[0]), (x[1], sol.X[1]), alpha=0.1, color="black")

        plt.scatter(algorithm.influenced.get("X")[:, 0], algorithm.influenced.get("X")[:, 1], color="red", marker="*",
                    alpha=0.7,
                    label="influenced")

        _biased = Population.create(
            *[e for e in algorithm.biased if e is not None])
        plt.scatter(_biased.get("X")[:, 0], _biased.get("X")[:, 1], color="orange", marker="s", label="Selected",
                    alpha=0.8,
                    s=100)

        plt.legend()


# =========================================================================================================
# Algorithm
# =========================================================================================================
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from pymoo.core.problem import Problem
from pymoo.core.evaluator import set_cv
from pymoo.util.termination.no_termination import NoTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import logging

def nds_score(estimator, X, y_grd):
    logging.debug(estimator, X, y_grd)
    nds_sorter = NonDominatedSorting()
    y_est = estimator.predict(X)
    _, y_nds_est = nds_sorter.do(y_est,
                                 only_non_dominated_front=False,
                                 return_rank=True)
    _, y_nds_grd = nds_sorter.do(y_grd,
                                 only_non_dominated_front=False,
                                 return_rank=True)
    logging.debug(y_nds_est, y_nds_grd)
    return metrics.accuracy_score(y_nds_grd, y_nds_est)


class ParDen(SurrogateAssistedAlgorithm):

    def __init__(self,
                 algorithm,
                 skip_already_evaluated=True,
                 surrogate=RandomForestRegressor(),
                 n_max_infills=np.inf,
                 **kwargs):

        SurrogateAssistedAlgorithm.__init__(self, **kwargs)

        self.proto = deepcopy(algorithm)
        self.algorithm = None

        # the control parameters for the surrogate assistance
        self.isfitted = False        
        self.nds_sorter = NonDominatedSorting()
        self.opt = None
        self.ndscore = 0.0

        # the maximum number of infill solutions
        self.n_max_infills = n_max_infills
        

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)

        # set the default termination to the proto type
        self.proto.termination = NoTermination()

        # setup the underlying algorithm with no problem
        self.proto.setup(Problem(), seed=self.seed, **kwargs)

        # copy the algorithm object to get started
        self.algorithm = deepcopy(self.proto)

        # customize the display to show the surrogate influence
        self.display = ParDenDisplay(self.algorithm.display)

    def _evalpop(self, pop, nds, problem):
        P = pop.get("X")        
        pop[nds].set("estimated", False)
        F = pop[nds].get("F")
        more_truth = np.atleast_2d(np.c_[F, P[nds]])
        if self.ground_truth is None:
            self.ground_truth = more_truth
        else:
            self.ground_truth = np.concatenate((self.ground_truth, more_truth))
        if self.isfitted:
            self.estimates[nds] = np.inf
        val = pop[nds]
        # can remove all of this if we pass the algorithm in to keep track of the opt
        if self.opt is not None:
            val = Population.merge(val, self.opt)
        self.opt = filter_optimum(val, least_infeasible=True)
        logging.debug("Surrogate: opt")
        logging.debug(self.opt.get("F"))


    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills))


    def _infill(self):
        # this would be the default behavior of the algorithm        
        # Generate new candidates */
        candidates = self.algorithm.infill()
        pop_size = candidates.shape[0]
        X_c = candidates.get("X")

        # Estimate candidates’ fitness with surrogate */
        candidates.set("estimated", True)
        Y_c = self.surrogate.predict(X_c)

        # Join candidates to non-dominated set */
        opt = self.opt.get("F")
        P_c = np.r_[self.estimates, opt]

        # Pretenders are non-dominated candidates */
        # get positons of the non-dominated set in P_c
        front = self.nds_sorter.do(P_c, only_non_dominated_front=True)        
        if front.min() < pop_size:       
            #positions of non-dominated candidates only
            nds = front[(front < pop_size)]            
        else:
            nds = None        

        # Acceptance sampling with NDScore as threshold to add additional pretenders */
        extra = np.arange(0, pop_size)
        extra = extra[~np.isin(extra, nds)]
        if len(extra) > 0:
            randoms = extra[(np.random.random(extra.shape[0]) > self.ndscore)]
            
            if len(randoms) > 0:
                extra = extra[~np.isin(extra, randoms)]
            
        pretenders = candidates[np.r_[nds,randoms]]
        pretenders.set("estimated", False)
        candidates[extra].set("F", self.estimates[extra])
            

        # Actual pretenders’ fitness */

        # Add to ground-truth */

        # Update non-dominated set */

        # Update non-dominated score */

        # Train new surrogate on ground-truth with loss L */

        # validate the current model
        self.surrogate.validate(trn=self.doe, tst=infills)

        # make a step in the main algorithm with high-fidelity solutions
        self.algorithm.advance(infills=infills, **kwargs)


        # now by default the infills are the surrogate-influenced solutions
        infills = influenced
        infills.set("type", "influenced")

        # now copy over the infills and set them to have never been evaluated
        ret = infills.copy(deep=True)
        for e in ret:
            e.reset(data=False)
        ret.set("X", infills.get("X"), "created_by", infills)

        return ret    

    def _advance(self, infills=None, **kwargs):                
        # merge the offsprings with the current population

        # execute the survival to find the fittest solutions

        super()._advance(infills=infills, **kwargs)

    def _set_optimum(self):
        sols = self.algorithm.opt
        if self.opt is not None:
            sols = Population.merge(sols, self.opt)
        self.opt = filter_optimum(sols, least_infeasible=True)
