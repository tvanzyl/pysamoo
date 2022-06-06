from pymoo.indicators.hv import Hypervolume

def SuccessRate(checkpoints, pf, at=0.99, indicator=Hypervolume(normalize=False, ref_point=[0.0, 100.0])):
    success = 0.0
    hv_opt = indicator.do(pf)
    for checkpoint in checkpoints:
        hv = indicator.do(checkpoint.opt.get("F"))
        if hv >= hv_opt * at:
            success += 1.0
    return success / len(checkpoints)


def AESR(checkpoints, pf, at=0.99, indicator=Hypervolume(normalize=False, ref_point=[0.0, 100.0])):
    success = 0.0
    evaluations = 0.0
    hv_opt = indicator.do(pf)
    for checkpoint in checkpoints:
        for hist in checkpoint.history:
            hv = indicator.do(hist.opt.get("F"))
            if hv >= hv_opt * at:
                success += 1.0
                evaluations += hist.evaluator.n_eval
                break
    return evaluations / success if success > 0 else 'NaN'

def AGSR(checkpoints, pf, at=0.99, indicator=Hypervolume(normalize=False, ref_point=[0.0, 100.0])):
    success = 0.0
    generations = 0.0
    hv_opt = indicator.do(pf)
    for checkpoint in checkpoints:
        generation = 0.0
        for hist in checkpoint.history:
            generation += 1.0         
            hv = indicator.do(hist.opt.get("F"))
            if hv >= hv_opt * at:
                success += 1.0
                generations += generation
                break
    return generations / success if success > 0 else 'NaN'