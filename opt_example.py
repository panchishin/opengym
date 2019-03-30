# define an objective function
def objective(args):
    case, val, useless = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1),hp.uniform('c3', 0, 10)),
        ('case 2', hp.uniform('c2', 0, 10),hp.uniform('c4', 0, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval, Trials
trials = Trials()

print("With an iterative call")
for i in range(1,20) :
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=i*5, show_progressbar=False)
    args = space_eval(space, best)
    print(i,args,"-->",objective(args)) 
