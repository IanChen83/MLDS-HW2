import config
import param
import model
__author__ = 'patrickchen'

update_pair_w = []
update_pair_b = []


def update_w():
    for i in range(config.layer_num):
        ret = param.Wi[i]
        for j in config.batch_num:
            ret += model.grad[j]
        update_pair_w.append(
            (param.Wi[i], (ret / config.batch_num))
        )


def update_b():
    for i in range(config.layer_num):
        ret = param.Bh[i]
        for j in config.batch_num:
            ret += model.grad[config.layer_num + j]
        update_pair_b.append(
            (param.Bh[i], (ret / config.batch_num))
        )
