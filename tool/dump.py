import argparse
import os
import pickle
import torch


def load_python2(data_path):
    from functools import partial
    import pickle as pickle_py2
    pickle_py2.load = partial(pickle_py2.load, encoding="latin1")
    pickle_py2.Unpickler = partial(pickle_py2.Unpickler, encoding="latin1")
    model = torch.load(
        data_path,
        map_location=lambda storage, loc: storage,
        pickle_module=pickle_py2)
    return model


def dump_python3(data, out_path):
    torch.save(data, out_path, pickle_module=pickle)


def dump_one_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    model = load_python2(args.model_path)
    dump_python3(model, args.out_path)


def dump_data():
    src_dir = 'data'
    dst_dir = 'data_py3'
    os.mkdir(dst_dir)
    for file in os.listdir(src_dir):
        data = load_python2(os.path.join(src_dir, file))
        dump_python3(data, os.path.join(dst_dir, file))


if __name__ == "__main__":
    dump_data()
    # dump_one_file()
