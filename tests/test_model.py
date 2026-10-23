import torch

from kneemri.data import gpu_augment, mirror, study_fold, zoom
from kneemri.model import KneeNet

SLOTS = ["sag_fluid", "cor_fluid"]


def tiny_model():
    return KneeNet("resnet18", n_slots=2, n_labels=12, pretrained=False, d=32)


def test_forward_shapes_all_and_sampled_windows():
    m = tiny_model().eval()
    x = torch.rand(2, 2, 4, 32, 32)
    present = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    assert m(x, present).shape == (2, 12)
    assert m(x, present, k=3).shape == (2, 12)


def test_missing_series_do_not_change_output():
    m = tiny_model().eval()
    x = torch.rand(1, 2, 4, 32, 32)
    present = torch.tensor([[1.0, 0.0]])
    x2 = x.clone()
    x2[:, 1] = torch.rand_like(x2[:, 1])           # change only the absent series
    with torch.no_grad():
        assert torch.allclose(m(x, present), m(x2, present), atol=1e-5)


def test_study_with_no_series_is_finite():
    m = tiny_model().eval()
    with torch.no_grad():
        assert torch.isfinite(m(torch.rand(1, 2, 4, 32, 32), torch.zeros(1, 2))).all()


def test_mirror_is_an_involution():
    x = torch.rand(2, 2, 4, 16, 16)
    assert torch.equal(mirror(mirror(x, SLOTS), SLOTS), x)
    assert torch.equal(mirror(x, SLOTS)[:, 0], x[:, 0].flip(1))     # sagittal: reverse slice order
    assert torch.equal(mirror(x, SLOTS)[:, 1], x[:, 1].flip(-1))    # coronal: flip image width


def test_augment_and_zoom_keep_shape_and_range():
    x = torch.rand(2, 2, 4, 16, 16)
    y = gpu_augment(x, SLOTS, mirror_p=0.5)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert zoom(x).shape == x.shape


def test_folds_are_stable_and_balanced():
    uids = [f"1.2.826.{i}" for i in range(2000)]
    folds = [study_fold(u) for u in uids]
    assert folds == [study_fold(u) for u in uids]
    assert min(folds.count(k) for k in range(5)) > 300
