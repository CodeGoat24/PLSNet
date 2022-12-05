import argparse
import re
import numpy as np
def main(args):
    table = []
    with open(args.path, 'r') as f:
        lines = f.readlines()

        for l in lines:
            value = re.findall(r'.*Epoch\[(\d+)/500\].*Train Loss: (\d+\.\d+).*Test Loss: (\d+\.\d+)', l)
            table.append(value[0])

    s = f'|Epoch|'
    for i in range(0, 500, 50):
        s += f'{i}|'
    print(s)

    for j, name in enumerate(['Train Loss', "Test Loss"]):
        s = f'|{name}|'
        for i in range(0, 500, 50):
            s += f'{table[i][j+1]}|'
        print(s)

    # data = np.load(args.path, allow_pickle=True).item()
    # final_fc = data["timeseires"]
    # final_pearson = data["corr"]
    # labels = data["label"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/star/CodeGoat24/FBNETGEN/result/ABIDE_AAL_71.43%/learnable_matrix.npy', type=str,
                        help='Log file path.')
    args = parser.parse_args()
    main(args)