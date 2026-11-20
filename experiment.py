from math import sqrt
from random import Random
from statistics import mean, stdev

BUDGETS = (16, 64, 256)
RUNS = 2000


def estimates(data):
    # Базовые режимы SGD, включая пакетные обновления.
    for batch in (1, 8):
        for decay in (False, True):
            x = 0.0
            rule = "1/k" if decay else "0.1"
            label = f"SGD {rule}, b={batch}"
            for start in range(0, len(data), batch):
                k = start // batch + 1
                step = 1 / k if decay else 0.1
                target = mean(data[start:start + batch])
                x += step * (target - x)
                n = start + batch
                if n in BUDGETS:
                    yield label, n, x

    # Три метода получают те же наблюдения в том же порядке.
    x_sgd = average = 0.0
    x_momentum = velocity = 0.0
    x_adagrad = sum_squares = 0.0
    for k, target in enumerate(data, start=1):
        x_sgd -= 0.1 * (x_sgd - target)
        average += (x_sgd - average) / k

        gradient = x_momentum - target
        velocity = 0.9 * velocity + gradient
        x_momentum -= 0.01 * velocity

        gradient = x_adagrad - target
        sum_squares += gradient ** 2
        x_adagrad -= 0.5 * gradient / (sqrt(sum_squares) + 1e-8)
        if k in BUDGETS:
            yield "Averaged SGD", k, average
            yield "Momentum", k, x_momentum
            yield "AdaGrad", k, x_adagrad


def experiment():
    errors = {}
    for seed in range(RUNS):
        rng = Random(seed)
        data = [2 + rng.choice((-1, 1)) for _ in range(max(BUDGETS))]
        for method, n, estimate in estimates(data):
            key = (method, n)
            errors.setdefault(key, []).append((estimate - 2) ** 2)
    print("method;observations;mse;standard_error")
    for (method, n), values in errors.items():
        se = stdev(values) / sqrt(RUNS)
        print(f"{method};{n};{mean(values):.8f};{se:.8f}")


if __name__ == "__main__":
    experiment()

