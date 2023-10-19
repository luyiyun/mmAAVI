import collections as ct

import torch

from mmAAVI.model.encoders import AttGMMEncoder


def get_encoder_and_input(ss=True):
    omics = ct.OrderedDict(atac=10, rna=5)
    encoder = AttGMMEncoder(omics, 2, 2, 2, semi_supervised=ss)

    inpts, masks = ct.OrderedDict(), ct.OrderedDict()
    for k, v in omics.items():
        inpts[k] = torch.zeros(100, v).float()
    masks["atac"] = torch.randint(1, size=(100,))
    masks["rna"] = 1 - masks["atac"]

    if not ss:
        return encoder, inpts, masks

    sslabel = torch.randint(3, size=(100,))
    sslabel[sslabel == 2] = -1

    return encoder, inpts, masks, sslabel


class TestEncoder:

    def test_encode_ss_output(self):
        encoder, inpts, masks, sslabel = get_encoder_and_input(True)
        _, losses = encoder.step(inpts, masks, sslabel=sslabel)
        assert list(losses.keys()) == ["kl_z", "kl_c", "kl_u", "ss"]
