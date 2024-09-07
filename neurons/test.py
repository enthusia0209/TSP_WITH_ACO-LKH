import numpy as np
import tqdm

from neurons.compare_solutions import compare

if __name__ == '__main__':
    data = [compare() for i in tqdm.tqdm(range(1))]
    data = np.array(data)

    print("BEAM:", data[:, 0].mean())
    print("BASELINE:", data[:, 1].mean())
    print("NNS_VALI :", data[:, 2].mean())
    # print("HPN:", data[:, 3].mean())
    print("CHRIST:", data[:, 3].mean())
    print("MIN 0:", data[:, 4].mean())

    print("ENHANCE:", data[:, 5].mean())
    print("MIN 1:", data[:, 6].mean())

    print("OR:", data[:, 7].mean())
    print("MIN 2:", data[:, 8].mean())

    print("LKH:", data[:, 9].mean())
    print("MIN 3:", data[:, 10].mean())

    print("ACO:", data[:, 11].mean())
    print("MIN 4:", data[:, 12].mean())

    # print("CON:", data[:, 13].mean())
    # print("MIN 5:", data[:, 14].mean())
