import numpy as np
import json

def generate_dataset(distributions_params: list[tuple],) -> tuple[list[float], list[int], list[int]]:
    distributions, labels, lengths = [], [], []
    r_ranges = np.arange(-50, 50, 1)
    r_i = np.random.choice(r_ranges, size=len(distributions_params))
    t_i = 2000 + r_i
    for i, (loc, scale) in enumerate(distributions_params):
        d = np.random.uniform(low=loc - scale, high=loc + scale, size=t_i[i])
        # d = np.random.normal(loc=loc, scale=scale, size=t_i[i])
        label = np.zeros(t_i[i])
        theshold = d.mean()
        label[d >= theshold] = 1
        distributions.extend(d.tolist())
        labels.extend(label)
        lengths.append(len(d))
    return distributions, labels, lengths

if __name__ == '__main__':
    distributions_params = [
        (0.6768, 0.0240),  (0.4815, 0.0384), (0.2533, 0.0414), (0.5237, 0.0419), (0.2692, 0.0268),
    ] * 2 # 10_000 samples but from 5 distributions
    print('Generating dataset...')
    distributions, labels, lengths = generate_dataset(distributions_params)
    print('Saving dataset...')
    with open('data/dataset_uniform.json', 'w') as f:
        ds_dict = {
            'distributions': distributions,
            'labels': labels,
            'distributions_params': distributions_params,
            'lengths': lengths,
        }
        json.dump(ds_dict, f)
    print('Done!')
