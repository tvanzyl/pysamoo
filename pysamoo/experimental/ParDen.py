import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.population import Population
from pymoo.util.optimum import filter_optimum
from pymoo.util.termination.no_termination import NoTermination

from pysamoo.core.algorithm import SurrogateAssistedAlgorithm
# =========================================================================================================
# Display
# =========================================================================================================
from pymoo.util.display import MultiObjectiveDisplay 

class ParDenDisplay(MultiObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        opt = algorithm.opt.get("F")

        # if algorithm.n_gen > 1:
        #     surr_infills = Population.create(*algorithm.infills.get("created_by"))
        #     n_influenced = sum(surr_infills.get("type") == "trace")
        #     self.output.append("n_influenced", f"{n_influenced}/{len(surr_infills)}")
        # else:
        #     self.output.append("n_influenced", "-")
        self.output.append("beta0", algorithm.beta0)
        self.output.append("nds_score", algorithm.ndscore)
        # self.output.append("mae", algorithm.mae)
        self.output.append("n_front", len(opt))



# =========================================================================================================
# Algorithm
# =========================================================================================================
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from scipy.stats import kendalltau, spearmanr

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.evaluator import set_cv
from pymoo.util.termination.no_termination import NoTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination
from pymoo.indicators.gd_plus import GDPlus

import logging

def nds_score(estimator, X, y_grd):
    nds_sorter = NonDominatedSorting()
    y_est = estimator.predict(X)
    _, y_nds_est = nds_sorter.do(y_est,
                                 only_non_dominated_front=False,
                                 return_rank=True)
    _, y_nds_grd = nds_sorter.do(y_grd,
                                 only_non_dominated_front=False,
                                 return_rank=True)    
    # return 1.0 - np.mean(np.abs(y_nds_grd - y_nds_est)/np.max((1.0,np.max(y_nds_est),np.max(y_nds_est))))
    # return metrics.accuracy_score(y_nds_grd, y_nds_est)
    tau, p_value = kendalltau(y_nds_grd, y_nds_est)
    # rho, p_value = spearmanr(y_nds_grd, y_nds_est)
    return max(0.0, tau*(1.0-p_value))


class SurrogateToProblem(Problem):

    def __init__(self, problem, surrogate):
        super().__init__(n_var=problem.n_var,
                         n_obj=problem.n_obj,
                         n_constr=problem.n_constr,
                         xl=problem.xl,
                         xu=problem.xu)
        self.surrogate = surrogate
        self.problem = problem

    def _evaluate(self, x, out, *args, **kwargs):        
        out["F"] = self.surrogate.predict(x)

from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination

class ParDen(SurrogateAssistedAlgorithm):

    def __init__(self,
                 algorithm,
                 twopoint0=True,
                 nondominated_ranks=1,                 
                 look_ahead=False,
                 surrogate=RandomForestRegressor(),
                 n_max_infills=np.inf,
                 **kwargs):
        
        SurrogateAssistedAlgorithm.__init__(self, **kwargs)
        
        self.proto = deepcopy(algorithm)
        self.algorithm = None
        self.look_ahead = look_ahead        
        self.twopoint0 = twopoint0
        self.ndrs = nondominated_ranks

        # the control parameters for the surrogate assistance        
        self.nds_sorter = NonDominatedSorting()        

        # the maximum number of infill solutions
        self.n_max_infills = n_max_infills
        
        self.surrogate = surrogate

        self.beta0 = -1
        

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)

        # setup the surrogate as a problem for future use as a beta
        self.surrogateproblem = SurrogateToProblem(problem, self.surrogate)

        self.algorithm = deepcopy(self.proto)
        self.algorithm.setup(problem, 
                             seed=self.seed, 
                             termination=NoTermination())

        # customize the display to show the surrogate influence
        self.display = ParDenDisplay()

    def _infill(self):
        # Doing a look ahead
        algorithm = deepcopy(self.algorithm)
        algorithm.problem = self.surrogateproblem
        algorithm.termination = MultiObjectiveSpaceToleranceTermination(tol=0.00001,
                                                      n_last=1,
                                                      nth_gen=1,
                                                      n_max_gen=None,
                                                      n_max_evals=None)

        if self.look_ahead:
            # do beta loops
            # fill the reservoir array
            candidates = algorithm.infill()
            k = len(candidates)
            i = len(candidates)
            self.beta0 = 0
            while algorithm.has_next():
                i += 1
                algorithm.next()
                j = np.random.randint(1,i)
                if j <= k and np.random.randn() <= self.ndscore:                    
                    opt_i = np.random.randint(0,len(algorithm.opt))
                    opt_X = algorithm.opt[opt_i].get("X")                    
                    candidates[j-1].set("X", opt_X)
                    self.beta0 += 1
        else:
            # this would be the default behavior of the algorithm        
            # Generate new candidates */        
            # get the infill solutions
            candidates = algorithm.infill()
        
        X_c = candidates.get("X")

        # Estimate candidates’ fitness with surrogate */        
        candidates.set("estimated", True)
        pop_size = candidates.shape[0]
        Y_c = self.surrogate.predict(X_c)

        # Join candidates to non-dominated set */
        opt = self.opt.get("F")
        P_c = np.r_[Y_c, opt]

        # Pretenders are non-dominated candidates */
        # get positions of the non-dominated set in P_c
        fronts = self.nds_sorter.do(P_c, only_non_dominated_front=False)
        nds = np.arange(0)
        for front in fronts[:min(self.ndrs, len(fronts))]:            
            if front.min() < pop_size:
                #positions of non-dominated candidates only                
                nds = np.append(nds, front[(front < pop_size)])

        pretenders = candidates[np.r_[nds]]
        candidates = self.algorithm.infill()        
        if self.twopoint0:
            pretenders = Population.merge(pretenders, candidates[len(pretenders):])
        else:
            # Acceptance sampling with NDScore as threshold to add additional pretenders */    
            # get positions of the dominated candidates
            extra = np.arange(0, pop_size)
            extra = extra[~np.isin(extra, nds)]
            if len(extra) > 0:
                #importance sampling 
                randoms = extra[(np.random.random(extra.shape[0]) > self.ndscore)]
            else:
                randoms = np.arange(0)
            pretenders = Population.merge(pretenders, candidates[randoms])
        
        infills = pretenders
        infills.set("type", "trace")
        
        # Actual pretenders’ fitness */
        #Done in the next step of the outer algorithm
        ret = infills.copy(deep=True)
        ret.set("X", infills.get("X"), "created_by", infills)
        return ret

    def _all_advance(self, infills=None, **kwargs):
        infills.set("estimated", False)
        # Update the sampler M        
        # self.algorithm.advance(infills=Population.merge(self.validation, infills), **kwargs)
        self.algorithm.advance(infills=infills, **kwargs)

        # Update non-dominated score */
        X = self.archive.get("X")
        y = self.archive.get("F")        
        self.ndscores = cross_val_score(self.surrogate,
                                        X,
                                        y,
                                        cv=np.minimum(5, y.shape[0]),
                                        scoring=nds_score,
                                        n_jobs=-1)
        self.ndscore = np.mean(self.ndscores)
        
        # Train new surrogate on ground-truth with loss L */
        # here we update our surrogate model
        self.surrogate.fit(X, y)
        
        # Update non-dominated set */
        #Done in next step of the outer algorithm
        self.pop = infills

    def _initialize_advance(self, infills=None, **kwargs):
        # Add to ground-truth */
        super()._initialize_advance(infills=infills, **kwargs)
        self._all_advance(infills=infills)

    def _advance(self, infills=None, **kwargs):
        # Add to ground-truth */
        super()._advance(infills=infills, **kwargs)
        self._all_advance(infills=infills)

    def _set_optimum(self):        
        sols = self.pop
        if self.opt is not None:
            sols = Population.merge(sols, self.opt)            
            self.opt_old = self.opt
        else:
            self.opt_old = None
        
        self.opt = filter_optimum(sols, least_infeasible=True)
        
        if self.opt_old is not None:
            self.gd_old = self.gd if self.gd > 0 else self.gd_old
            self.gd = GDPlus(normalize=False, pf=self.opt.get("F")).do(self.opt_old.get("F"))
        else:
            self.gd = 0.0
            self.gd_old = 0.0            
        
            
        
        

