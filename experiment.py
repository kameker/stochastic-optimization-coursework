from random import Random
from statistics import mean, stdev
from math import sqrt

budgets = (16, 64, 256)
runs = 2000
modes = ((1, False), (8, False), (1, True), (8, True))
errors = {}

for seed in range(runs):
    rng = Random(seed)
    data = [2 + rng.choice((-1, 1)) for _ in range(max(budgets))]
    for batch, decay in modes:
        x = 0.0
        for start in range(0, len(data), batch):
            k = start // batch + 1
            step = 1.0 / k if decay else 0.1
            target = mean(data[start:start + batch])
            x += step * (target - x)
            n = start + batch
            if n in budgets:
                key = (batch, decay, n)
                errors.setdefault(key, []).append((x - 2)**2)

for key, values in errors.items():
    print(key, round(mean(values), 6),
          round(stdev(values) / sqrt(runs), 6))

